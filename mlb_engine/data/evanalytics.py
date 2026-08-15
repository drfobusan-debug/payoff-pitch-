"""EV Analytics' player-prop board, read from a saved page.

Their table publishes THE BAT X's projection for a prop next to the book's line
and price, which is the same forecast ``data.batx`` reads -- but as a *mean*
rather than a probability, and already joined to a line. That makes it a second
opinion the engine can print on almost every prop it prices, where BAT X's own
export has to be downloaded and run through ``scripts/batx_study.py`` first.

Why a saved page and not a fetch: the board is drawn from a JSON endpoint that
answers without a key, but signed out every row past the first few is replaced
with placeholder text -- 38,000 rows of the word "@EV Analytics". A subscriber's
saved page carries the real numbers, so the file on disk is the feed.

What is deliberately absent, as with every outside model here: any path into
``model_prob``, ``bet_prob``, ``edge``, the tiers or the screens. Their number is
printed beside ours and read by nothing.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# Their market names, mapped onto ours. Note that "Walks" is the batter's own
# walks and "Walks Allowed" the pitcher's, which are different rows in the same
# column. The unmapped ones are real markets we do not price (stolen bases,
# triples, runs+RBIs, pitcher win) -- dropped rather than guessed at, so a
# renamed column shows up as a fall in coverage instead of as silence.
MARKETS = {
    "Hits": "batter_h",
    "Singles": "batter_1b",
    "Doubles": "batter_2b",
    "Home Runs": "batter_hr",
    "Runs": "batter_r",
    "RBIs": "batter_rbi",
    "Hits Runs and RBIs": "batter_hrr",
    "Total Bases": "batter_tb",
    "Walks": "batter_bb",
    "Hitter Strikeouts": "batter_k",
    "Strikeouts": "pitcher_k",
    "Pitching Outs": "pitcher_outs",
    "Hits Allowed": "pitcher_h",
    "Walks Allowed": "pitcher_bb",
    "Earned Runs": "pitcher_er",
}

# "0.5 (+172)" -> line 0.5, price +172. The price is absent on some rows.
_QUOTE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:\(([+-]?\d+)\))?")
_SEL_SUFFIX = re.compile(
    r"\s+(H\+R\+RBI|1B|2B|3B|HR|TB|BB|H|R|RBI|K|Ks|Walks|Hits|ER|Outs)\s+[ou][\d.]+$"
)


@dataclass(frozen=True)
class EVProp:
    """One prop as their board publishes it.

    ``projection`` is THE BAT X's mean for the stat, which is a statement about
    the player and not about the line: it is comparable to any line the book
    hangs, and that is what lets it be joined to a prop we priced at a different
    number. ``suggestion`` is the side *they* call at *their* line.
    """

    date: str
    player: str
    team: str
    matchup: str
    market: str
    line: float | None
    projection: float
    implied: float | None
    suggestion: str

    @property
    def side(self) -> str:
        """The side their board is on: their own call, or their number's.

        Their "suggested bet" column is the published one and is used whenever
        it is filled. Where it is blank the direction is taken from their
        projection against the market's *implied* projection -- not against the
        line. Those are different questions: a 0.86 hits projection is above a
        0.5 line and yet exactly on a market implying 0.86, and the second is
        the one that says whether the price is wrong.
        """
        if self.suggestion in ("over", "under"):
            return self.suggestion
        if self.implied is None or self.projection == self.implied:
            return ""
        return "over" if self.projection > self.implied else "under"

    def reads(self, side: str, line: float) -> bool | None:
        """Whether their call backs ``side`` at *our* ``line``, or says nothing.

        Their side is stated at their own number, and it only carries to a
        different one in the direction it already points: an under at 0.5 is
        also an under at 1.5, and an over at 1.5 is also an over at 0.5, because
        the harder claim contains the easier one. Read the other way it does
        not -- liking the over 0.5 says nothing about the over 2.5 -- so those
        get no mark instead of a borrowed one.
        """
        theirs = self.side
        if not theirs:
            return None
        if self.line is not None and self.line != line:
            carries = line >= self.line if theirs == "under" else line <= self.line
            if not carries:
                return None
        return theirs == side

    @property
    def summary(self) -> str:
        """"OVER: BATX 1.47 vs 1.11 implied" -- their side and both numbers."""
        numbers = f"BATX {self.projection:.2f}"
        if self.implied is not None:
            numbers += f" vs {self.implied:.2f} implied"
        return f"{self.side.upper()}: {numbers}" if self.side else numbers


def _norm(name: str) -> str:
    plain = unicodedata.normalize("NFKD", str(name).casefold())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    plain = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", plain.strip())
    return re.sub(r"[^a-z0-9]+", "", plain)


def player_from_selection(selection: str) -> str:
    return _SEL_SUFFIX.sub("", str(selection))


def _number(text: str) -> float | None:
    try:
        return float(text.replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _quote(text: str) -> float | None:
    """The line out of "0.5 (+172)"; ``None`` when the book is not hanging one."""
    m = _QUOTE.search(text or "")
    return float(m.group(1)) if m else None


def _slate_date(text: str, year: int) -> str:
    """Their "Aug 15" carries no year, so the file's own rows date the capture."""
    try:
        parsed = Date(year, _MONTHS[text.split()[0][:3].title()], int(text.split()[1]))
    except (KeyError, IndexError, ValueError):
        return ""
    return parsed.isoformat()


_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
}


def parse_board(html: str, year: int | None = None) -> list[EVProp]:
    """Read every prop off a saved board, skipping the ones we do not price."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="dataTable")
    if table is None:
        log.warning("EV Analytics: no props table in the saved page")
        return []
    headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
    year = year or Date.today().year
    props: list[EVProp] = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells, strict=True))
        market = MARKETS.get(row.get("MARKET", ""))
        projection = _number(row.get("THE BAT X PROJECTION", ""))
        player = row.get("PLAYER", "").strip()
        if market is None or projection is None or not player:
            continue
        # Their over and under columns are quoted separately and either can be
        # blank; the line is the same number on both, so the first one wins.
        line = _quote(row.get("OVER", "")) or _quote(row.get("UNDER", ""))
        props.append(
            EVProp(
                date=_slate_date(row.get("DATE", ""), year),
                player=player,
                team=row.get("TM", "").strip(),
                matchup=row.get("GAME", "").replace("@", " @ ").strip(),
                market=market,
                line=line,
                projection=projection,
                implied=_number(row.get("IMPLIED PROJECTION", "")),
                suggestion=row.get("SUGGESTED BET", "").strip().lower(),
            )
        )
    log.info("EV Analytics: %d props parsed", len(props))
    return props


def load_board(directory: Path, date: str | None = None) -> list[EVProp]:
    """Every saved board in ``directory`` for ``date``, merged into one.

    Their table pages at 250 rows and offers no "show all", so a full slate is
    several saved pages; all of them are read and combined rather than the
    newest one winning, which would silently keep a fifth of the board.

    A file is dated by the rows inside it, never by its name or its mtime, so
    yesterday's download left in the folder is dropped rather than joined to
    tonight's bets -- the one way a benchmark can be worse than absent.
    """
    if not directory.is_dir():
        return []
    props: list[EVProp] = []
    seen: set[tuple[str, str, float | None]] = set()
    for path in sorted(directory.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.suffix.lower() not in (".html", ".htm"):
            continue
        for prop in parse_board(path.read_text(encoding="utf-8", errors="replace")):
            if date is not None and prop.date != date:
                continue
            key = (prop.market, _norm(prop.player), prop.line)
            if key in seen:
                continue
            seen.add(key)
            props.append(prop)
    log.info("EV Analytics: %d props for %s", len(props), date or "all dates")
    return props


def _key(market: str, player: str) -> str:
    return f"{market}|{_norm(player)}"


def annotate(recs: list, props: list[EVProp]) -> int:
    """Print their projection beside ours, and whether it backs our side.

    The join is on the player and the stat, not on the line. Their projection is
    a mean -- "1.47 hits" -- so it speaks to whatever number the book hung, and
    requiring an exact line match would blank the column on precisely the props
    where the engine has taken an alternate number and a second opinion is worth
    most.

    The mark compares sides, not numbers: their over against our over. A board
    row that takes no side -- no suggested bet and a projection sitting on the
    market's own implied number -- gets no mark rather than a coin flip.
    """
    by_key: dict[str, EVProp] = {}
    for prop in props:
        by_key.setdefault(_key(prop.market, prop.player), prop)
    hits = 0
    for rec in recs:
        if rec.line is None or rec.side not in ("over", "under"):
            continue
        theirs = by_key.get(_key(rec.market, player_from_selection(rec.selection)))
        if theirs is None:
            continue
        rec.ev_proj = theirs.projection
        rec.ev_pick = theirs.summary
        rec.ev_agrees = theirs.reads(rec.side, rec.line)
        hits += 1
    return hits
