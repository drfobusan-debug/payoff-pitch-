"""VSiN public betting splits for college football: where the money is, per side.

VSiN publishes each book's handle share and ticket share per side at
``data.vsin.com/betting-splits/?sport=CFB``. The quantity worth having is the
difference between them -- handle% minus tickets% -- because a side taking a
larger share of the *money* than of the *bets* is being backed by bigger
accounts, and bigger accounts are the ones a book moves a number for.

Why this feed rather than another box-score metric: in the sibling MLB engine
these two columns were the only moneyline input that separated graded winners
from losers (handle-minus-tickets AUC 0.80, p=0.027), while the engine's own
model EV was *inverted* on that market (AUC 0.33, p=0.004 over 102 graded rows).
CFB has no graded rows yet, so nothing here is claimed to have been measured on
college football -- see :mod:`cfb_engine.market.mlsharp` for what is done with
it, and why only the moneyline consumes it.

The page is public HTML, no key. Two books are read, sharpest first: Circa (a
limit-taking Las Vegas book, so its handle share is closer to a professional
read) then DraftKings (retail, but it posts far more of the board -- 306 rows
against Circa's 102 in week 1). The first book with a usable split for a side
wins; a side neither book reports has no split at all, which every consumer has
to treat as "no read" rather than as a negative one.

Rows are keyed by *date and team*, not team alone: the page lists the whole
remaining season at once, so a team appears a dozen times and matching on name
would price week 1 off some November number.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import requests

from cfb_engine.data.teamnames import school_key
from cfb_engine.market import keys
from cfb_engine.schemas import Slate
from mlb_engine.data import http

log = logging.getLogger(__name__)

SPLITS_URL = "https://data.vsin.com/betting-splits/?source={source}&sport=CFB"
_HEADERS = {"User-Agent": "Mozilla/5.0 (payoff-pitch-cfb/1.0)"}
# VSiN's ``source`` code -> book label, sharpest first.
_SOURCES: tuple[tuple[str, str], ...] = (("circa", "circa"), ("DK", "draftkings"))
_TTL = 1800.0

_ROW_RE = re.compile(r'<tr class="sp-row[^"]*">(.*?)</tr>', re.S)
_GAME_RE = re.compile(r'data-gamecode="(\d{4})(\d{2})(\d{2})CFB')
_TEAM_RE = re.compile(r'class="sp-team-link"[^>]*>([^<]+)<')
_BADGE_RE = re.compile(r'<span class="sp-badge[^"]*">(.*?)</span>', re.S)

# VSiN spells schools the way a betting screen does -- abbreviated, and without a
# state's full name -- and ``school_key`` cannot know that "N Dakota ST" is North
# Dakota State. Leading directional initials are expanded generically below; only
# the spellings that survive that need listing.
_VSIN_ALIASES: dict[str, str] = {
    "texas san antonio": "utsa",
    "middle tenn": "middle tennessee",
    "florida intl": "florida international",
}
_DIRECTIONS = {
    "n": "north", "s": "south", "e": "east", "w": "west", "c": "central",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
}


@dataclass(frozen=True)
class Split:
    """One side's share of the money and of the tickets, at one book."""

    handle_pct: float | None = None
    bets_pct: float | None = None
    book: str = ""

    @property
    def divergence(self) -> float | None:
        """Handle share minus ticket share, in percentage points.

        Positive means the money is heavier on this side than the ticket count is
        -- fewer, larger bets. ``None`` when either share is missing, which is the
        only honest reading of a side the book reported nothing for.
        """
        if self.handle_pct is None or self.bets_pct is None:
            return None
        return round(self.handle_pct - self.bets_pct, 1)


# (matchup, market, side) -> Split. The side is handicap-free
# (``keys.side_of``), so a split read at VSiN's main number still resolves for a
# selection the engine priced at a different one -- the money is on the team, not
# on the half-point.
SplitBook = dict[tuple[str, str, str], Split]


def vsin_key(name: str) -> str:
    """``school_key`` after expanding the abbreviations VSiN's screen uses."""
    words = school_key(name).split()
    expanded = [_DIRECTIONS.get(w, w) if i == 0 else w for i, w in enumerate(words)]
    joined = " ".join(expanded)
    return _VSIN_ALIASES.get(joined, joined)


@dataclass(frozen=True)
class SplitRow:
    """One team's line of the splits table, on one date."""

    game_date: Date
    name: str
    spread_handle: float | None
    spread_bets: float | None
    total_handle: float | None
    total_bets: float | None
    ml_american: float | None
    ml_handle: float | None
    ml_bets: float | None


class SplitsProvider:
    """Fetches and caches the VSiN CFB splits pages, keyed to a slate."""

    def __init__(self, cache_dir: Path, ttl: float = _TTL, timeout: int = 25) -> None:
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.timeout = timeout

    def fetch(self, slate: Slate) -> SplitBook:
        """Splits for every side of ``slate`` either book reports.

        Returns an empty book rather than raising when the pages are
        unavailable: a public-money read is an input to a screen, and losing it
        should cost the screen its opinion, not the slate its card.
        """
        sides: dict[tuple[Date, str], tuple[str, str, bool]] = {}
        for game in slate.games:
            for team, is_home in ((game.home, True), (game.away, False)):
                sides[(game.game_date, vsin_key(team.name))] = (
                    game.matchup(), team.abbrev, is_home
                )

        book: SplitBook = {}
        for source, label in _SOURCES:
            for row in self._rows(source):
                match = sides.get((row.game_date, vsin_key(row.name)))
                if match is None:
                    continue
                matchup, abbrev, is_home = match
                if row.ml_american is not None:
                    _put(book, (matchup, "game_ml", abbrev),
                         Split(row.ml_handle, row.ml_bets, label))
                _put(book, (matchup, "game_ats", abbrev),
                     Split(row.spread_handle, row.spread_bets, label))
                # VSiN lists the visitor's row against the Over and the home
                # team's against the Under, so a total's split belongs to a side
                # of the number rather than to a team.
                _put(book, (matchup, "game_total", "Under" if is_home else "Over"),
                     Split(row.total_handle, row.total_bets, label))
        log.info("VSiN splits: %d sides read for %d games", len(book), len(slate.games))
        return book

    # -- HTTP --------------------------------------------------------------
    def _rows(self, source: str) -> list[SplitRow]:
        text = self._page(source)
        return parse_splits(text) if text else []

    def _page(self, source: str) -> str | None:
        cached = self._cache_read(source)
        if cached is not None:
            return cached
        try:
            resp = http.get(
                SPLITS_URL.format(source=source), headers=_HEADERS, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("VSiN splits fetch failed (%s): %s", source, exc)
            return None
        self._cache_write(source, resp.text)
        return resp.text

    def _cache_path(self, source: str) -> Path:
        return self.cache_dir / "vsin" / f"splits_{source}.html"

    def _cache_read(self, source: str) -> str | None:
        path = self._cache_path(source)
        if not path.exists():
            return None
        if self.ttl > 0 and time.time() - path.stat().st_mtime > self.ttl:
            return None
        try:
            return path.read_text()
        except OSError:
            return None

    def _cache_write(self, source: str, text: str) -> None:
        path = self._cache_path(source)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        except OSError as exc:
            log.warning("could not cache VSiN splits (%s): %s", source, exc)


def parse_splits(html: str) -> list[SplitRow]:
    """Team rows from a VSiN splits page (two rows per game, road team first).

    Read from the markup rather than with ``pandas.read_html`` for the date: the
    page's date headings are ``<thead>`` rows that a dataframe hoists into column
    labels, discarding which game each row belongs to, while every row carries
    its own ``data-gamecode`` stamped with the kickoff date.
    """
    rows: list[SplitRow] = []
    for chunk in _ROW_RE.findall(html):
        stamp = _GAME_RE.search(chunk)
        team = _TEAM_RE.search(chunk)
        badges = _BADGE_RE.findall(chunk)
        if stamp is None or team is None or len(badges) < 9:
            continue
        year, month, day = (int(g) for g in stamp.groups())
        rows.append(
            SplitRow(
                game_date=Date(year, month, day),
                name=team.group(1).strip(),
                spread_handle=_pct(badges[1]),
                spread_bets=_pct(badges[2]),
                total_handle=_pct(badges[4]),
                total_bets=_pct(badges[5]),
                ml_american=_american(badges[6]),
                ml_handle=_pct(badges[7]),
                ml_bets=_pct(badges[8]),
            )
        )
    return rows


def lookup(book: SplitBook, matchup: str, market: str, selection: str) -> Split | None:
    """The split for a selection, resolved through its handicap-free side."""
    return book.get((matchup, market, keys.side_of(selection)))


def _put(book: SplitBook, key: tuple[str, str, str], split: Split) -> None:
    """File a usable split, leaving a reading from a sharper book in place.

    A market the book has taken no action in is printed as a placeholder rather
    than left blank -- ``0%``/``0%`` on a week-1 moneyline against an FCS visitor,
    ``50%``/``50%`` on both sides of a Circa game, ``100%``/``100%`` against a
    side that is blank -- and all of them say the same identical share of handle
    and tickets. None of them can be told apart from a genuine dead heat, so the
    whole equal-shares case is discarded, which costs nothing: a divergence of
    exactly zero is the gate's boundary and changes no verdict, while discarding
    it lets a book that *did* publish a split be read instead. (Circa prints
    50/50 on moneylines it has no split for, so keeping them would hide
    DraftKings' +23 behind a fabricated dead heat.)
    """
    if not split.divergence:
        return
    book.setdefault(key, split)


def _pct(v: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", v)
    return float(m.group()) if m else None


def _american(v: str) -> float | None:
    m = re.search(r"[+-]?\d+", v.replace(" ", ""))
    return float(m.group()) if m else None
