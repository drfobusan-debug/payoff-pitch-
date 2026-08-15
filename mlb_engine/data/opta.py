"""VSIN/Opta AI player prop projections: an outside model to be judged against.

VSIN publishes Opta's MLB prop projections at ``data.vsin.com/projections``,
with DraftKings prices beside them and, once a slate finishes, the graded
outcome of every call. The page is a shell; the rows come from an undocumented
fragment endpoint that needs no login:

    props_ajax.php?sport=MLB&tab={ms,ou}&stat={Hits,2B,...}&page=N&day={-1,0,1}

Two reasons to keep it, neither of which is pricing -- the odds shown are
DraftKings', which the Odds API already gives us properly:

1. It is an *independent* model. Where it and the engine disagree, one of them
   is wrong, and the ledger can eventually say which. Grading ourselves against
   ourselves is the same weakness CLV exists to fix.
2. ``day`` clamps at yesterday -- asking for -7 or -30 returns yesterday's slate
   -- so there is no archive to backfill. A benchmark only exists if it is
   captured nightly, starting now.

The endpoint returns HTML with no contract behind it, so every field is parsed
defensively and a row that does not make sense is dropped rather than guessed
at. Losing a benchmark row costs nothing; a wrong one is worse than none.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import date as Date
from html import unescape
from pathlib import Path

import requests

from mlb_engine.data import http
from mlb_engine.market import keys

log = logging.getLogger(__name__)

BASE = "https://data.vsin.com/projections"
_FRAGMENT = f"{BASE}/props_ajax.php"
_PAGE = f"{BASE}/?sport=mlb"
_HEADERS = {"User-Agent": "Mozilla/5.0 (mlb-prediction-engine)", "Referer": _PAGE}

# VSIN's stat code -> (engine market, the stat symbol used in a selection key).
# Their "HA" is hits allowed and their "Hits" is the batter's; the two collide
# on any lazier mapping.
_STATS: dict[str, tuple[str, str]] = {
    "Hits": ("batter_h", "H"),
    "2B": ("batter_2b", "2B"),
    "HR": ("batter_hr", "HR"),
    "TB": ("batter_tb", "TB"),
    "HRR": ("batter_hrr", "H+R+RBI"),
    "K": ("pitcher_k", "Ks"),
    "Outs": ("pitcher_outs", "Outs"),
    "ER": ("pitcher_er", "ER"),
    "HA": ("pitcher_h", "Hits"),
}
_PITCHER_STATS = frozenset({"K", "Outs", "ER", "HA"})
# VSIN's team codes differ from MLB's for seven clubs. Left unmapped, every
# Padres/Rays/Nationals/Royals/White Sox/Diamondbacks/Giants row would fail to
# join the engine's slate.
_TEAMS = {
    "KAN": "KC", "WAS": "WSH", "CHW": "CWS", "ARI": "AZ",
    "TAM": "TB", "SDG": "SD", "SFO": "SF",
}
_MAX_PAGES = 40  # a slate runs to ~11 pages per stat; this is a runaway guard

_TAG = re.compile(r"<[^>]+>")
_ROW = re.compile(r"(?s)<tr[^>]*>(.*?)</tr>")
_CELL = re.compile(r"(?s)<t[dh][^>]*>(.*?)</t[dh]>")
_PLAYER_ID = re.compile(r"loadPlayer\((\d+)\)")
_PAGES = re.compile(r"Page\s+(\d+)\s+of\s+(\d+)")
_MATCHUP = re.compile(r"([A-Z]{2,3})\s*(@|vs)\s*([A-Z]{2,3})")
_DAY_LABEL = re.compile(r'data-day="(-?\d+)"[^>]*>([^<]+)<')
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_ODDS = re.compile(r"([+-]\d+)")
_SIDE_ODDS = re.compile(r"([OU])\s*([+-]\d+)")
_MILESTONE = re.compile(r"\d\+")


@dataclass(frozen=True)
class OptaRow:
    """One Opta projection, its DraftKings price, and how it turned out."""

    date: str
    matchup: str  # engine form, "AWAY @ HOME"
    market: str
    selection: str  # the engine's quote key, when the row maps onto one
    player: str
    player_id: int | None
    stat: str
    line: float
    projection: float | None
    over_odds: float | None
    under_odds: float | None
    over_prob: float | None  # Opta's probability of the over, comparable to ours
    edge: float | None  # Opta's edge over the book, as a fraction
    bet: str | None  # "over", "under", or None where it found no value
    confidence: int  # 0-3 stars
    result: str | None  # "hit", "miss", or None before the game is graded
    actual: float | None

    @property
    def key(self) -> str:
        return "|".join((self.date, self.matchup, self.market, self.selection))


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(_TAG.sub(" ", html))).strip()


def _number(s: str) -> float | None:
    m = _NUMBER.search(s.replace(",", ""))
    return float(m.group()) if m else None


def _american(s: str) -> float | None:
    m = _ODDS.search(s.replace(" ", ""))
    return float(m.group(1)) if m else None


def _implied(american: float) -> float:
    return 100.0 / (american + 100.0) if american > 0 else -american / (-american + 100.0)


def _line(cell: str) -> float | None:
    """The threshold being bet.

    The over/under tab writes it plainly ("1.5"); the milestone tab writes a
    target ("2+"), which is the over on the half-point below it, so 2+ and o1.5
    are the same bet and must land on the same selection key.
    """
    txt = _text(cell)
    n = _number(txt)
    if n is None:
        return None
    # The "+" sits against the digits, but a trend arrow may follow it.
    return n - 0.5 if _MILESTONE.search(txt) else n


def _confidence(cell: str) -> int:
    return _text(cell).count("\u2605")


def _result(cell: str) -> tuple[str | None, float | None]:
    txt = _text(cell).upper()
    if "HIT" in txt:
        return "hit", _number(txt)
    if "MISS" in txt:
        return "miss", _number(txt)
    return None, None


class OptaClient:
    """Reads VSIN's Opta projection tables. No credentials, no credits."""

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    def _get(self, url: str, **params: object) -> str:
        try:
            resp = http.get(url, params=params, headers=_HEADERS, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            # A benchmark feed going dark must never take a slate down with it.
            log.warning("VSIN Opta request failed (%s): %s", url, exc)
            return ""

    def slate_dates(self) -> dict[int, str]:
        """Which calendar date each ``day`` offset means, read off the page.

        Worth the extra request: the offset does not roll at midnight. At 3am ET
        the page still called the previous evening's slate ``day=0``, so
        assuming ``day=0`` is today silently files a capture under tomorrow.
        """
        html = self._get(_PAGE)
        out: dict[int, str] = {}
        for off, label in _DAY_LABEL.findall(html):
            parsed = _parse_day_label(label)
            if parsed is not None:
                out[int(off)] = parsed.isoformat()
        return out

    def fetch(self, day: int = 0, date: str | None = None) -> list[OptaRow]:
        """Every projection VSIN shows for one slate, across both tabs.

        The two tabs are different bets on the same players -- milestones
        ("2+ hits") and over/unders ("o1.5") -- and both are kept, deduplicated
        on the selection they resolve to.
        """
        if date is None:
            date = self.slate_dates().get(day, "")
        rows: dict[str, OptaRow] = {}
        for tab in ("ou", "ms"):
            for stat in _STATS:
                for row in self._fetch_stat(tab, stat, day, date):
                    # The over/under tab is read first and wins: it carries both
                    # sides of the price, where a milestone row has only the over.
                    rows.setdefault(row.key, row)
        log.info("VSIN Opta %s (day=%d): %d projections", date or "?", day, len(rows))
        return list(rows.values())

    def _fetch_stat(self, tab: str, stat: str, day: int, date: str) -> Iterator[OptaRow]:
        page, last = 1, 1
        while page <= min(last, _MAX_PAGES):
            html = self._get(
                _FRAGMENT, sport="MLB", tab=tab, stat=stat, page=page,
                filter="all", sort="edge", dir="desc", day=day,
            )
            if not html:
                return
            found = _PAGES.search(html)
            if found:
                last = int(found.group(2))
            seen = 0
            for raw in _ROW.findall(html):
                row = _parse_row(raw, stat, date)
                if row is not None:
                    seen += 1
                    yield row
            if seen == 0:
                return
            page += 1


def _parse_day_label(label: str) -> Date | None:
    """"Sat Aug 8" -> a date. The year is absent, so it is inferred."""
    m = re.search(r"([A-Z][a-z]{2})\s+(\d{1,2})", label.strip())
    if m is None:
        return None
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if m.group(1) not in months:
        return None
    month, dom = months.index(m.group(1)) + 1, int(m.group(2))
    today = Date.today()
    year = today.year
    # A December label read in January belongs to the year just gone.
    if month == 12 and today.month == 1:
        year -= 1
    elif month == 1 and today.month == 12:
        year += 1
    try:
        return Date(year, month, dom)
    except ValueError:
        return None


def _parse_row(raw: str, stat: str, date: str) -> OptaRow | None:
    cells = _CELL.findall(raw)
    if len(cells) < 8:
        return None
    who = _text(cells[0])
    game = _MATCHUP.search(who)
    if game is None:  # a header or a spacer row
        return None
    away, home = _teams(game)
    player = who[: game.start()].strip(" \u00b7-")
    if not player:
        return None
    line = _line(cells[2])
    if line is None:
        # Without a line there is no bet to key on, and every such row would
        # collapse onto the same key and evict a real one.
        return None
    market, symbol = _STATS[stat]
    over, under = _prices(cells[3])
    result, actual = _result(cells[8]) if len(cells) > 8 else (None, None)
    pid = _PLAYER_ID.search(cells[0])
    selection = (
        keys.pitcher_prop(player, symbol, line) if stat in _PITCHER_STATS
        else keys.batter_prop(player, symbol, line)
    )
    return OptaRow(
        date=date,
        matchup=f"{away} @ {home}",
        market=market,
        selection=selection,
        player=player,
        player_id=int(pid.group(1)) if pid else None,
        stat=symbol,
        line=line,
        projection=_number(_text(cells[1])),
        over_odds=over,
        under_odds=under,
        over_prob=_over_prob(cells[4]),
        edge=_percent(cells[5]),
        bet=_bet(cells[6]),
        confidence=_confidence(cells[7]),
        result=result,
        actual=actual,
    )


def _teams(game: re.Match[str]) -> tuple[str, str]:
    """VSIN writes the row's own player first, so "@" and "vs" swap the sides."""
    first, sep, second = game.group(1), game.group(2), game.group(3)
    first, second = _TEAMS.get(first, first), _TEAMS.get(second, second)
    return (first, second) if sep == "@" else (second, first)


def _prices(cell: str) -> tuple[float | None, float | None]:
    """Both sides where the tab gives them, else the over alone."""
    txt = _text(cell)
    sides = dict(_SIDE_ODDS.findall(txt.replace(" ", "")))
    if sides:
        over = float(sides["O"]) if "O" in sides else None
        under = float(sides["U"]) if "U" in sides else None
        return over, under
    return _american(txt), None


def _over_prob(cell: str) -> float | None:
    """Opta's probability of the over.

    It publishes its view as a price on whichever side it likes ("U +130"), so
    half the rows state the under. Both are flipped to the over here: the
    engine's own ``model_prob`` is always the over on an ``o{line}`` key, and a
    benchmark that silently changes which side it is quoting cannot be compared
    to anything.
    """
    m = _SIDE_ODDS.search(_text(cell).replace(" ", ""))
    if m is None:
        return None
    prob = _implied(float(m.group(2)))
    return round(prob if m.group(1) == "O" else 1.0 - prob, 6)


def _percent(cell: str) -> float | None:
    n = _number(_text(cell))
    return None if n is None else round(n / 100.0, 6)


def _bet(cell: str) -> str | None:
    txt = _text(cell).upper()
    if "OVER" in txt:
        return "over"
    if "UNDER" in txt:
        return "under"
    return None


def _match_key(market: str, selection: str) -> str:
    """Join key tolerant of the spelling differences between two feeds.

    Both sides build the selection with ``market.keys``, so the shape already
    agrees; what does not is punctuation in a name -- "Luis Robert Jr." against
    "Luis Robert Jr", "Andres" against "Andrés".
    """
    plain = unicodedata.normalize("NFKD", selection.casefold())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return f"{market}|" + re.sub(r"[^a-z0-9]+", "", plain)


def annotate(recs: list, rows: list[OptaRow]) -> int:
    """Stamp each recommendation with Opta's read on the same prop.

    A second model's probability beside our own is the one thing the card
    cannot get from the price: the market quotes what it chooses to, and CLV
    only grades the number we paid. Where Opta has an opinion on the same
    selection, the sheet can show whether an outside forecast agrees.

    Opta's own star rating is carried through rather than a threshold of our
    invention, and it is only shown against the side Opta actually bet -- three
    stars on the under says nothing good about our over.
    """
    by_key: dict[str, OptaRow] = {}
    for entry in rows:
        by_key.setdefault(_match_key(entry.market, entry.selection), entry)
    hits = 0
    for rec in recs:
        row = by_key.get(_match_key(rec.market, rec.selection))
        if row is None:
            continue
        prob = row.over_prob
        if prob is not None and rec.side == "under":
            prob = 1.0 - prob
        rec.opta_prob = prob
        rec.opta_stars = row.confidence
        rec.opta_agrees = None if row.bet is None else row.bet == rec.side
        hits += 1
    return hits


def save_rows(path: Path, rows: list[OptaRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows], indent=1), encoding="utf-8")


def load_rows(path: Path) -> list[OptaRow]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return [OptaRow(**r) for r in raw if isinstance(r, dict)]


def merge_rows(old: list[OptaRow], new: list[OptaRow]) -> list[OptaRow]:
    """Union two captures of a slate, a graded view of a prop beating an open one.

    "Later wins" is the obvious rule and the wrong one, because the two captures
    travel between machines and arrive in either order: the morning's projection
    reaches the branch after the evening's result as often as before it. Grading
    is the one thing that only moves forwards, so it decides.
    """
    merged: dict[str, OptaRow] = {r.key: r for r in old}
    for row in new:
        seen = merged.get(row.key)
        if seen is None or seen.result is None or row.result is not None:
            merged[row.key] = row
    return [merged[k] for k in sorted(merged)]
