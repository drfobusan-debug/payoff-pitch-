"""TeamRankings' daily MLB picks: a second outside model, on our own markets.

TeamRankings publishes a picks grid at ``teamrankings.com/mlb-betting-picks/``
-- one row per game, four columns: the projected winner, a run-line value pick,
a total, and a money-line value pick, each carrying their star rating. Two of
those columns say "Lay Off" when the model finds no value, which is itself an
opinion worth recording.

Why this one, when :mod:`mlb_engine.data.opta` already gives us an outside model:
Opta prices *props*, which is the half of our book that is measurably biased,
and it does not touch the game markets at all. TeamRankings covers exactly the
markets where our own ledger says the engine is unbiased (moneyline, run line,
totals), so it is the first benchmark that can disagree with us where we are
strongest -- which is where a disagreement is informative.

The stars are theirs, carried through as published rather than rethresholded
into a tier of ours. The free grid shows one- and two-star picks; the higher
ratings sit behind their subscription, so an absent three-star pick means "not
published to us", not "not made". ``stars`` is the rating as printed and no
inference is drawn from its ceiling.

The page is plain server-rendered HTML with no contract behind it, so every
field is parsed defensively and a row that does not make sense is dropped
rather than guessed at, as with Opta: losing a benchmark row costs nothing, a
wrong one is worse than none.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path

import requests

from mlb_engine.config import Credentials
from mlb_engine.data import http
from mlb_engine.market import keys
from mlb_engine.recommendations import Recommendation

log = logging.getLogger(__name__)

PICKS_URL = "https://www.teamrankings.com/mlb-betting-picks/"
LOGIN_URL = "https://www.teamrankings.com/login/"
_HEADERS = {"User-Agent": "Mozilla/5.0 (mlb-prediction-engine)"}

# The cookie that means *subscriber*, and the only one that does. ``tr_session``
# is handed to anonymous visitors by the login page itself, so checking for it
# accepted every rejected login: the account that could not sign in captured the
# free grid -- the last slate already played -- and reported success. A wrong
# password has to be louder than a missing benchmark, because it looks exactly
# like one.
SUBSCRIBER_COOKIE = "tru"

# Their team rating tables. Only the two that are not already inside the market
# price we anchor to are worth storing, plus the headline rating for context:
#
#   luck         wins above what the run differential implies -- an outside
#                reading of the season luck gap ``features.team_form`` scaffolds
#   consistency  game-to-game variability, *lower is steadier*. Nothing in the
#                engine models the spread of a team's performance, only its
#                mean, and the spread is what a total and a run line are priced
#                off -- a volatile club blows out and gets blown out.
#   predictive   their power rating, kept only as the yardstick the other two
#                are read against.
#
# Ratings are as-of-today with no archive, so like the picks they exist only if
# they are captured nightly.
RATING_URL = "https://www.teamrankings.com/mlb/ranking/{}-by-other"
RATINGS = ("luck", "consistency", "predictive")

# TeamRankings writes a club's name its own way in the grid's team column, and
# its own abbreviation inside a pick cell. Both are mapped to the engine's code
# so a pick joins the slate. The abbreviations agree with ours for most clubs;
# the ones that do not are the same seven Opta gets wrong, plus Sacramento for
# the Athletics.
TEAM_NAMES: dict[str, str] = {
    "arizona": "AZ", "atlanta": "ATL", "baltimore": "BAL", "boston": "BOS",
    "chi cubs": "CHC", "chi sox": "CWS", "cincinnati": "CIN", "cleveland": "CLE",
    "colorado": "COL", "detroit": "DET", "houston": "HOU", "kansas city": "KC",
    "la angels": "LAA", "la dodgers": "LAD", "miami": "MIA", "milwaukee": "MIL",
    "minnesota": "MIN", "ny mets": "NYM", "ny yankees": "NYY", "philadelphia": "PHI",
    "pittsburgh": "PIT", "sacramento": "ATH", "san diego": "SD", "seattle": "SEA",
    "sf giants": "SF", "st. louis": "STL", "tampa bay": "TB", "texas": "TEX",
    "toronto": "TOR", "washington": "WSH",
}
TEAM_CODES: dict[str, str] = {
    "ARI": "AZ", "CHW": "CWS", "KAN": "KC", "SAC": "ATH",
    "SDG": "SD", "SFO": "SF", "TAM": "TB", "WAS": "WSH",
}

# The engine markets their grid speaks to. Their projected winner is game-level
# context, so it is shown on any of the three, not only on the moneyline.
GAME_MARKETS = frozenset({"game_ml", "game_rl", "game_total"})

_ROW = re.compile(r"(?s)<tr[^>]*>(.*?)</tr>")
_CELL = re.compile(r"(?s)<t[dh]([^>]*)>(.*?)</t[dh]>")
# Each pick cell sorts on "{stars}-{number}", and the number is the figure the
# column is about: a win probability in the winner and total columns, the value
# TeamRankings sees in the price in the run-line and money-line columns.
_SORT = re.compile(r'data-sort="(\d+)-(\d+(?:\.\d+)?)"')
_TAG = re.compile(r"<[^>]+>")
_STARS = re.compile(r"tr_stars_(\d+)")
_SLUG_DATE = re.compile(r"/mlb/matchup/[a-z0-9-]+?-(\d{4}-\d{2}-\d{2})")
_RUNLINE = re.compile(r"^([A-Z]{2,3})\s*([+-]\d+(?:\.\d+)?)\s*([+-]\d+)?$")
_MONEYLINE = re.compile(r"^([A-Z]{2,3})\s*([+-]\d+)?$")
_TOTAL = re.compile(r"^(Over|Under)\s+(\d+(?:\.\d+)?)$", re.I)
_RECORD = re.compile(r"\(\d+-\d+\)")

# Column order in the grid. The winner column is a straight projection with no
# price; the other two bet columns can decline to bet.
_WINNER, _RUNLINE_COL, _TOTAL_COL, _MONEYLINE_COL = 2, 3, 4, 5


@dataclass(frozen=True)
class Cell:
    """One pick cell: its text, its star rating, and the number it sorts on."""

    text: str
    stars: int
    figure: float | None


@dataclass(frozen=True)
class TRPick:
    """One TeamRankings call on one market of one game."""

    date: str
    matchup: str  # engine form, "AWAY @ HOME"
    market: str  # game_ml | game_rl | game_total | game_winner
    selection: str
    line: float | None
    side: str  # "over"/"under" for totals, "" otherwise
    team: str  # engine team code the pick backs, "" for a total
    team_side: str  # "home"/"away", "" for a total
    american: float | None
    stars: int
    # Their published number for this call: the winner and total columns sort on
    # a win probability, the two value columns on the edge they see in the price.
    # Kept as a fraction, and never mixed with ours -- the point of a benchmark
    # is that it is computed somewhere else.
    win_prob: float | None = None
    value: float | None = None

    @property
    def key(self) -> str:
        """One call per game market, so re-capturing a slate cannot double it.

        Deliberately excludes the selection: the grid is captured several times
        before first pitch, and if they move a total from 8.0 to 8.5 that is the
        same call revised, not a second bet. Keying on the selection would leave
        both in the ledger and pay their benchmark twice on one market.
        """
        return "|".join((self.date, self.matchup, self.market))

    @property
    def summary(self) -> str:
        """Their call as they publish it, with their own number attached.

        The winner and total columns carry a win probability and the two value
        columns the edge they see in the price, so each is printed in the unit
        the column is actually about rather than being coerced into one field.
        """
        stars = f" {self.stars}\u2605" if self.stars else ""
        if self.win_prob is not None:
            return f"{self.selection} {self.win_prob * 100:.0f}%{stars}"
        if self.value is not None:
            return f"{self.selection} +{self.value * 100:.1f}% val{stars}"
        return f"{self.selection}{stars}"


@dataclass(frozen=True)
class TeamRating:
    """Their ratings for one club on one day, as published.

    Kept as their number and their rank, unnormalised: the rank is the only part
    that is comparable across days, because the rating's scale drifts as the
    season fills in. Nothing here is a probability and none of it is blended
    with ours -- these are stored to be measured against our own errors first.
    """

    date: str
    team: str
    luck: float | None = None
    luck_rank: int | None = None
    # Lower is steadier: their table ranks the least volatile club first.
    consistency: float | None = None
    consistency_rank: int | None = None
    predictive: float | None = None
    predictive_rank: int | None = None

    @property
    def key(self) -> str:
        return f"{self.date}|{self.team}"


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(_TAG.sub(" ", html))).strip()


def _cell(attrs: str, html: str) -> Cell:
    sort = _SORT.search(attrs)
    stars = _STARS.search(html)
    return Cell(
        text=_text(html),
        # The star markup is the published rating; the sort key repeats it and is
        # the fallback for a row whose stars are rendered some other way.
        stars=int(stars.group(1)) if stars else (int(sort.group(1)) if sort else 0),
        figure=float(sort.group(2)) if sort else None,
    )


def _code(token: str) -> str:
    return TEAM_CODES.get(token.upper(), token.upper())


def _teams(cell: str) -> tuple[str, str] | None:
    """The two clubs of a row, away first, as engine codes.

    The grid stacks the visitor above the host in one cell, which survives the
    tag strip as two names separated by whitespace -- and both are multi-word
    ("Chi Cubs", "St. Louis"), so they cannot be split on the space. The known
    names are matched instead, longest first, and anything unrecognised drops
    the row rather than half-reading it.
    """
    text = _text(cell).lower()
    found: list[tuple[int, str]] = []
    for name, code in TEAM_NAMES.items():
        at = text.find(name)
        if at >= 0:
            found.append((at, code))
    if len(found) != 2:
        return None
    found.sort()
    return found[0][1], found[1][1]


class TeamRankingsClient:
    """Reads the TeamRankings MLB picks grid, signed in when we have an account.

    Signed out, the grid serves the *last completed* slate with the results
    already filled in -- their model's calls become public only once they can no
    longer be bet. Tonight's grid needs the subscriber cookie, so a run without
    credentials captures nothing rather than filing yesterday's board as today's.
    """

    def __init__(self, creds: Credentials | None = None, timeout: float = 20.0) -> None:
        self.timeout = timeout
        self.creds = creds or Credentials()
        self._session: requests.Session | None = None

    def _signed_in(self) -> requests.Session | None:
        """A logged-in session, or ``None`` when we have no account or it failed."""
        if self._session is not None:
            return self._session
        if not self.creds.has_teamrankings():
            return None
        session = requests.Session()
        try:
            session.get(LOGIN_URL, headers=_HEADERS, timeout=self.timeout)
            resp = session.post(
                LOGIN_URL,
                data={
                    "email": self.creds.teamrankings_user,
                    "password": self.creds.teamrankings_pass,
                },
                headers={**_HEADERS, "Referer": LOGIN_URL},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("TeamRankings login failed: %s", exc)
            return None
        if SUBSCRIBER_COOKIE not in session.cookies:
            log.warning(
                "TeamRankings login rejected the credentials (no %s cookie): tonight's "
                "grid will not be published to us and the capture will file the last "
                "slate that was played",
                SUBSCRIBER_COOKIE,
            )
            return None
        self._session = session
        return session

    def _get(self, url: str) -> str:
        session = self._signed_in()
        try:
            if session is not None:
                resp = session.get(url, headers=_HEADERS, timeout=self.timeout)
            else:
                resp = http.get(url, headers=_HEADERS, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            # A benchmark feed going dark must never take a slate down with it.
            log.warning("TeamRankings request failed (%s): %s", url, exc)
            return ""

    def fetch(self, date: str | None = None) -> list[TRPick]:
        """Every pick on the published grid, optionally filtered to one slate.

        Only the current grid is served -- there is no date parameter and no
        archive -- so, exactly as with Opta, a benchmark exists only if it is
        captured nightly.
        """
        picks = parse_picks(self._get(PICKS_URL))
        if date is not None:
            picks = [p for p in picks if p.date == date]
        log.info("TeamRankings %s: %d picks", date or "all", len(picks))
        return picks

    def fetch_ratings(self, date: str) -> list[TeamRating]:
        """Their luck, consistency and predictive ratings, one row per club.

        A table that fails to load leaves its fields empty rather than dropping
        the club: the ratings are independent of each other, and a day with one
        of the three missing is still worth having.
        """
        tables = {
            name: parse_ratings(self._get(RATING_URL.format(name))) for name in RATINGS
        }
        teams = sorted({t for table in tables.values() for t in table})
        out: list[TeamRating] = []
        for team in teams:
            luck_rank, luck = _rating_of(tables["luck"], team)
            con_rank, con = _rating_of(tables["consistency"], team)
            pred_rank, pred = _rating_of(tables["predictive"], team)
            out.append(
                TeamRating(
                    date=date, team=team,
                    luck=luck, luck_rank=luck_rank,
                    consistency=con, consistency_rank=con_rank,
                    predictive=pred, predictive_rank=pred_rank,
                )
            )
        log.info("TeamRankings ratings %s: %d clubs", date, len(out))
        return out


def parse_picks(html: str) -> list[TRPick]:
    out: list[TRPick] = []
    for raw in _ROW.findall(html):
        out.extend(_parse_row(raw))
    return out


def _parse_row(raw: str) -> list[TRPick]:
    raw_cells = _CELL.findall(raw)
    if len(raw_cells) <= _MONEYLINE_COL:
        return []
    cells = [_cell(attrs, html) for attrs, html in raw_cells]
    teams = _teams(raw_cells[1][1])
    slug = _SLUG_DATE.search(raw)
    if teams is None or slug is None:  # a header, a spacer, or an unknown club
        return []
    away, home = teams
    matchup = f"{away} @ {home}"
    date = slug.group(1)
    picks = [
        _winner(cells[_WINNER], date, matchup, away, home),
        _runline(cells[_RUNLINE_COL], date, matchup, away, home),
        _total(cells[_TOTAL_COL], date, matchup),
        _moneyline(cells[_MONEYLINE_COL], date, matchup, away, home),
    ]
    return [p for p in picks if p is not None]


def _side_of(team: str, away: str, home: str) -> str:
    return "home" if team == home else "away" if team == away else ""


def _winner(cell: Cell, date: str, matchup: str, away: str, home: str) -> TRPick | None:
    """The projected winner, which is a forecast rather than a bet.

    It is kept on its own ``game_winner`` market rather than folded into
    ``game_ml``: the money-line column is where TeamRankings says a side is
    *worth backing at the price*, and it says "Lay Off" for most games. Merging
    the two would credit the model with a bet it declined to make.
    """
    team = _team_named(cell.text)
    if team is None:
        return None
    return TRPick(
        date=date, matchup=matchup, market="game_winner",
        selection=keys.game_ml(team), line=None, side="", team=team,
        team_side=_side_of(team, away, home), american=None, stars=cell.stars,
        win_prob=_fraction(cell.figure),
    )


def _fraction(figure: float | None) -> float | None:
    """A published percentage as a fraction, ignoring anything out of range."""
    if figure is None or not 0.0 <= figure <= 100.0:
        return None
    return figure / 100.0


def _team_named(text: str) -> str | None:
    """The club a cell names, whether written long ("Chi Cubs") or short ("STL")."""
    lowered = text.lower()
    for name, code in TEAM_NAMES.items():
        if name in lowered:
            return code
    token = text.split()[0] if text.split() else ""
    return _code(token) if token.isalpha() and 2 <= len(token) <= 3 else None


def _runline(cell: Cell, date: str, matchup: str, away: str, home: str) -> TRPick | None:
    m = _RUNLINE.match(cell.text)
    if m is None:  # "Lay Off", or a shape we do not recognise
        return None
    team, point = _code(m.group(1)), float(m.group(2))
    side = _side_of(team, away, home)
    if not side:
        return None
    return TRPick(
        date=date, matchup=matchup, market="game_rl",
        selection=keys.game_rl(team, point), line=point, side="", team=team,
        team_side=side, american=float(m.group(3)) if m.group(3) else None,
        stars=cell.stars, value=_fraction(cell.figure),
    )


def _total(cell: Cell, date: str, matchup: str) -> TRPick | None:
    m = _TOTAL.match(cell.text)
    if m is None:
        return None
    over = m.group(1).lower() == "over"
    line = float(m.group(2))
    return TRPick(
        date=date, matchup=matchup, market="game_total",
        selection=keys.game_total(over, line), line=line,
        side="over" if over else "under", team="", team_side="",
        american=None, stars=cell.stars, win_prob=_fraction(cell.figure),
    )


def _moneyline(cell: Cell, date: str, matchup: str, away: str, home: str) -> TRPick | None:
    m = _MONEYLINE.match(cell.text)
    if m is None:
        return None
    team = _code(m.group(1))
    side = _side_of(team, away, home)
    if not side:
        return None
    return TRPick(
        date=date, matchup=matchup, market="game_ml",
        selection=keys.game_ml(team), line=None, side="", team=team,
        team_side=side, american=float(m.group(2)) if m.group(2) else None,
        stars=cell.stars, value=_fraction(cell.figure),
    )


def annotate(recs: list[Recommendation], picks: list[TRPick]) -> int:
    """Stamp each game-market recommendation with TeamRankings' calls on it.

    All four of their columns are carried, because they are four different
    statements. The run line, total and money-line *value* picks each answer the
    bet we are making and are joined market to market. The projected winner is
    not a bet -- it names a side on every game, including the ones where the
    value columns say "Lay Off" -- so it rides along on every row of the game as
    context rather than being folded into the money-line comparison, where it
    would silently turn "no value here" into a pick they never made.

    Props are untouched: TeamRankings does not price them.
    """
    by_market: dict[str, TRPick] = {}
    winners: dict[str, TRPick] = {}
    for pick in picks:
        if pick.market == "game_winner":
            winners[pick.matchup] = pick
        else:
            by_market[f"{pick.market}|{pick.matchup}"] = pick
    hits = 0
    for rec in recs:
        winner = winners.get(rec.matchup)
        if winner is not None and rec.market in GAME_MARKETS:
            rec.tr_winner = winner.summary
        theirs = by_market.get(f"{rec.market}|{rec.matchup}")
        if theirs is None:
            continue
        rec.tr_pick = theirs.summary
        rec.tr_stars = theirs.stars or None
        rec.tr_agrees = _agrees(rec, theirs)
        hits += 1
    return hits


def _agrees(rec: Recommendation, pick: TRPick) -> bool | None:
    """Whether TeamRankings is on our side of this bet, ``None`` when unclear.

    Totals are compared as a direction and team markets as a team, for the same
    reason as VSiN's join: two models can back the same side at two numbers, and
    which side of the game they are on is the comparison worth printing.
    """
    if pick.side in ("over", "under"):
        return pick.side == rec.side if rec.side in ("over", "under") else None
    if not pick.team:
        return None
    head = rec.selection.split()[0] if rec.selection.split() else ""
    return pick.team == head if head else None


def parse_ratings(html: str) -> dict[str, tuple[int, float]]:
    """One rating table: engine team code -> (their rank, their rating).

    The team cell carries the record ("Tampa Bay (74-48)"), which is stripped;
    a club we cannot name is skipped rather than ranked as something else.
    """
    out: dict[str, tuple[int, float]] = {}
    for raw in _ROW.findall(html):
        cells = [_text(html_) for _, html_ in _CELL.findall(raw)]
        if len(cells) < 3 or not cells[0].isdigit():  # a header or a spacer
            continue
        team = TEAM_NAMES.get(_RECORD.sub("", cells[1]).strip().lower())
        try:
            rating = float(cells[2])
        except ValueError:
            continue
        if not math.isfinite(rating):  # "nan" parses as a float and is not JSON
            continue
        if team is not None:
            out[team] = (int(cells[0]), rating)
    return out


def _rating_of(
    table: dict[str, tuple[int, float]], team: str
) -> tuple[int | None, float | None]:
    got = table.get(team)
    return (None, None) if got is None else got


def save_picks(path: Path, picks: list[TRPick]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in picks], indent=1), encoding="utf-8")


def load_picks(path: Path) -> list[TRPick]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return [TRPick(**p) for p in raw if isinstance(p, dict)]


def save_ratings(path: Path, ratings: list[TeamRating]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in ratings], indent=1), encoding="utf-8")


def load_ratings(path: Path) -> list[TeamRating]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return [TeamRating(**r) for r in raw if isinstance(r, dict)]


def merge_ratings(old: list[TeamRating], new: list[TeamRating]) -> list[TeamRating]:
    """One row per club per day, the later capture winning."""
    merged: dict[str, TeamRating] = {r.key: r for r in old}
    merged.update({r.key: r for r in new})
    return [merged[k] for k in sorted(merged)]


def merge_picks(old: list[TRPick], new: list[TRPick]) -> list[TRPick]:
    """Union two captures of a slate, the later one winning.

    Unlike Opta's rows these carry no graded outcome -- the audit grades them
    against the box score itself -- so there is no forward-only field to protect
    and the fresher capture simply wins.
    """
    merged: dict[str, TRPick] = {p.key: p for p in old}
    merged.update({p.key: p for p in new})
    return [merged[k] for k in sorted(merged)]
