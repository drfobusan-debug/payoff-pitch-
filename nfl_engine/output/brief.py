"""The context a reader wants beside a game's number, gathered and never priced.

A :class:`GameBrief` is assembled at card time from three places the engine
already has -- the nflverse schedule (records, results, rest, roof, surface,
named starters), the rating book pricing was given (opponent-adjusted EPA and
success rate, ranked across the league) and ESPN's public summary (preview,
leaders, injuries, venue, forecast) -- plus the ledger's own rows, from which the
model's spread and total are read back off the rung it priced nearest to a coin
flip. Every source is optional and best-effort: a field the source does not have
is ``None`` and the card leaves it out rather than guessing.

Nothing here feeds back. The card is built from ledger rows written by pricing,
and a brief can be absent, partial or wrong without a probability, screen or
tier moving.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from nfl_engine.audit.ledger import ENGINE, LedgerEntry
from nfl_engine.data import nflverse, weather
from nfl_engine.data.color import ColorBook, ESPNColor, GameColor, TeamColor
from nfl_engine.data.teamnames import BY_NAME, canonical, franchise
from nfl_engine.features import books as books_mod
from nfl_engine.features.adjustments import Situation
from nfl_engine.features.ratings import RatingBook

log = logging.getLogger(__name__)

_FULL_NAME = {code: name.title().replace("49Ers", "49ers") for name, code in BY_NAME.items()}


def team_name(code: str) -> str:
    return _FULL_NAME.get(canonical(code), code)


@dataclass
class TeamBrief:
    code: str
    name: str
    record: str = ""
    streak: str = ""
    last: str = ""  # "W 34-26 @ TEN"
    ats: str | None = None
    qb: str | None = None
    rest: int | None = None
    rated: bool = False
    net_rank: int | None = None
    net_rating: float | None = None  # net success rate, pct points vs league
    off_epa: float | None = None
    off_rank: int | None = None
    def_epa: float | None = None
    def_rank: int | None = None
    leaders: list[str] = field(default_factory=list)
    out: list[str] = field(default_factory=list)
    questionable: list[str] = field(default_factory=list)


@dataclass
class GameBrief:
    matchup: str
    home: TeamBrief
    away: TeamBrief
    venue: str | None = None
    city: str | None = None
    roof: str | None = None  # nflverse: outdoors / dome / closed / open
    surface: str | None = None
    grass: bool | None = None
    neutral_site: bool = False
    # ``None`` is unknown: only a stated flag lets the divisional total term speak.
    div_game: bool | None = None
    broadcast: str | None = None
    kickoff_local: str | None = None
    headline: str | None = None
    story: str | None = None
    tags: list[str] = field(default_factory=list)
    fpi_home: float | None = None
    condition: str | None = None
    temperature_f: float | None = None
    precip_pct: float | None = None
    gust_mph: float | None = None
    # Sustained kickoff wind from the Open-Meteo forecast, the reading the wind
    # term prices; ESPN's gust above is colour only.
    wind_mph: float | None = None
    # Read back from the ledger's rungs: the home handicap and total the model
    # priced nearest even money, beside the market's consensus rung.
    model_spread: float | None = None
    market_spread: float | None = None
    model_total: float | None = None
    market_total: float | None = None

    def indoors(self) -> bool:
        return self.roof in ("dome", "closed")

    def situation(self) -> Situation:
        """The game's context in the shape the adjustment layer reads."""
        return Situation(
            roof=self.roof,
            wind_mph=self.wind_mph,
            temp_f=self.temperature_f,
            home_rest=self.home.rest,
            away_rest=self.away.rest,
            neutral_site=self.neutral_site,
            div_game=self.div_game,
        )


def _text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, str)):
        try:
            if isinstance(value, float) and pd.isna(value):
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None
    return None


def _played(schedule: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    played = schedule[
        (schedule.season == season)
        & (schedule.week < week)
        & schedule.home_score.notna()
        & schedule.away_score.notna()
    ]
    return played.sort_values(["week", "gameday"])


def _form(played: pd.DataFrame, code: str) -> tuple[str, str, str]:
    """``(record, streak, last)`` for ``code`` from the season's finished games."""
    wins = losses = ties = 0
    letters: list[str] = []
    last = ""
    for row in played.itertuples():
        home, away = canonical(str(row.home_team)), canonical(str(row.away_team))
        if code not in (home, away):
            continue
        us, them = (
            (row.home_score, row.away_score) if code == home else (row.away_score, row.home_score)
        )
        opp = away if code == home else home
        letter = "W" if us > them else "L" if us < them else "T"
        wins += letter == "W"
        losses += letter == "L"
        ties += letter == "T"
        letters.append(letter)
        last = f"{letter} {int(us)}-{int(them)} {'vs' if code == home else '@'} {opp}"
    record = f"{wins}-{losses}" + (f"-{ties}" if ties else "")
    streak = ""
    if letters:
        run = 0
        for letter in reversed(letters):
            if letter != letters[-1]:
                break
            run += 1
        streak = f"{letters[-1]}{run}"
    return record, streak, last


def _ranks(book: RatingBook) -> dict[str, tuple[int, int, int]]:
    """``(net, offense, defense)`` league rank per team; a better defence is lower def_epa."""
    teams = list(book.teams.values())
    if not teams:
        return {}
    by_net = sorted(teams, key=lambda t: -t.net_success())
    by_off = sorted(teams, key=lambda t: -t.off_epa)
    by_def = sorted(teams, key=lambda t: t.def_epa)
    out: dict[str, tuple[int, int, int]] = {}
    for team in teams:
        out[team.team] = (
            by_net.index(team) + 1,
            by_off.index(team) + 1,
            by_def.index(team) + 1,
        )
    return out


def _apply_rating(
    brief: TeamBrief, book: RatingBook, ranks: dict[str, tuple[int, int, int]]
) -> None:
    code = franchise(brief.code)
    if code not in book.teams or not book.is_usable():
        return
    rating = book.teams[code]
    net, off, deff = ranks[code]
    brief.rated = True
    brief.net_rank, brief.net_rating = net, rating.net_success() * 100
    brief.off_epa, brief.off_rank = rating.off_epa, off
    brief.def_epa, brief.def_rank = rating.def_epa, deff


def _apply_color(brief: TeamBrief, color: TeamColor) -> None:
    # The schedule is the authority on form when it is there; ESPN's last five
    # reach into last season, which is not this year's streak.
    if not brief.record:
        brief.record = color.record or ""
        if color.last_five:
            brief.streak = color.streak()
            brief.last = color.last_five[0]
    brief.ats = color.ats
    brief.leaders = list(color.leaders)
    brief.out = list(color.out)
    brief.questionable = list(color.questionable)


def _game_color(brief: GameBrief, color: GameColor) -> None:
    if color.venue is not None:
        brief.venue = color.venue
    if color.city is not None:
        brief.city = color.city
    if color.grass is not None:
        brief.grass = color.grass
    if color.indoor is True and brief.roof is None:
        brief.roof = "dome"
    brief.neutral_site = color.neutral_site
    brief.broadcast = color.broadcast
    brief.headline, brief.story, brief.tags = color.headline, color.story, list(color.tags)
    brief.fpi_home = color.fpi_home
    brief.condition = color.condition
    brief.temperature_f, brief.precip_pct, brief.gust_mph = (
        color.temperature_f,
        color.precip_pct,
        color.gust_mph,
    )
    _apply_color(brief.home, color.home)
    _apply_color(brief.away, color.away)


def _nearest_even(rows: list[LedgerEntry], prob: str) -> float | None:
    """The line whose ``prob`` sits nearest 0.5 -- the number that side of the market implies."""
    scored: list[tuple[float, float]] = []
    for e in rows:
        value = e.fair_prob if prob == "fair" else e.model_prob
        if e.line is None or value is None:
            continue
        scored.append((abs(value - 0.5), e.line))
    if not scored:
        return None
    return min(scored, key=lambda x: x[0])[1]


def _lines_from_ledger(brief: GameBrief, rows: list[LedgerEntry]) -> None:
    home = brief.home.code
    spreads = [e for e in rows if e.market == "spread" and canonical(e.side) == home]
    totals = [e for e in rows if e.market == "total" and e.side == "over"]
    brief.model_spread = _nearest_even(spreads, "model")
    brief.market_spread = _nearest_even(spreads, "fair")
    brief.model_total = _nearest_even(totals, "model")
    brief.market_total = _nearest_even(totals, "fair")


def _schedule_row(
    schedule: pd.DataFrame, season: int, week: int, home: str, away: str
) -> pd.Series | None:
    rows = schedule[
        (schedule.season == season)
        & (schedule.week == week)
        & (schedule.home_team.map(lambda c: canonical(str(c))) == home)
        & (schedule.away_team.map(lambda c: canonical(str(c))) == away)
    ]
    return rows.iloc[0] if len(rows) else None


def _kickoff_local(row: pd.Series) -> str | None:
    day, clock = _text(row.get("gameday")), _text(row.get("gametime"))
    if not day:
        return None
    try:
        when = pd.Timestamp(day)
    except (TypeError, ValueError):
        return None
    label = when.strftime("%a %b %-d")
    return f"{label} {clock} ET" if clock else label


def build_briefs(
    entries: Iterable[LedgerEntry],
    *,
    season: int,
    week: int,
    schedule: pd.DataFrame | None = None,
    ratings: RatingBook | None = None,
    color: ColorBook | None = None,
    forecast: bool = False,
) -> dict[str, GameBrief]:
    """A brief per matchup priced in ``season`` week ``week``.

    Pure given its inputs unless ``forecast`` is set, which asks Open-Meteo for
    the kickoff wind; :func:`gather` is the one that goes and fetches the rest.
    """
    scope = [e for e in entries if e.season == season and e.week == week and e.source == ENGINE]
    by_game: dict[str, list[LedgerEntry]] = {}
    for entry in scope:
        by_game.setdefault(entry.matchup, []).append(entry)
    played = _played(schedule, season, week) if schedule is not None else None
    ranks = _ranks(ratings) if ratings is not None else {}
    out: dict[str, GameBrief] = {}
    stadium_ids: dict[str, str | None] = {}
    for matchup, rows in by_game.items():
        away_code, _, home_code = matchup.partition(" @ ")
        home_code, away_code = canonical(home_code.strip()), canonical(away_code.strip())
        brief = GameBrief(
            matchup=matchup,
            home=TeamBrief(code=home_code, name=team_name(home_code)),
            away=TeamBrief(code=away_code, name=team_name(away_code)),
        )
        if schedule is not None and played is not None:
            for team in (brief.home, brief.away):
                team.record, team.streak, team.last = _form(played, team.code)
            row = _schedule_row(schedule, season, week, home_code, away_code)
            if row is not None:
                brief.roof = _text(row.get("roof"))
                brief.surface = _text(row.get("surface"))
                brief.venue = _text(row.get("stadium"))
                div_flag = _int(row.get("div_game"))
                brief.div_game = None if div_flag is None else bool(div_flag)
                brief.neutral_site = bool(_int(row.get("neutral_site")) or 0)
                stadium_ids[matchup] = _text(row.get("stadium_id"))
                brief.kickoff_local = _kickoff_local(row)
                brief.home.qb = _text(row.get("home_qb_name"))
                brief.away.qb = _text(row.get("away_qb_name"))
                brief.home.rest = _int(row.get("home_rest"))
                brief.away.rest = _int(row.get("away_rest"))
        if ratings is not None:
            _apply_rating(brief.home, ratings, ranks)
            _apply_rating(brief.away, ratings, ranks)
        if color is not None and matchup in color:
            _game_color(brief, color[matchup])
        _lines_from_ledger(brief, rows)
        out[matchup] = brief
    if forecast:
        _attach_wind(out, stadium_ids, scope)
    return out


def _attach_wind(
    briefs: dict[str, GameBrief],
    stadium_ids: dict[str, str | None],
    entries: list[LedgerEntry],
) -> None:
    """Kickoff wind and temperature for the outdoor games whose venue is known."""
    kickoffs = {e.matchup: e.kickoff_utc for e in entries if e.kickoff_utc}
    points: dict[str, tuple[str, datetime]] = {}
    for matchup, brief in briefs.items():
        stadium = stadium_ids.get(matchup)
        when = weather.parse_kickoff(kickoffs.get(matchup, ""))
        if stadium is None or when is None or not weather.is_outdoors(brief.roof):
            continue
        points[matchup] = (stadium, when)
    for matchup, reading in weather.readings(points).items():
        brief = briefs[matchup]
        brief.wind_mph = reading.wind_mph
        if brief.temperature_f is None:
            brief.temperature_f = reading.temperature_f
        if brief.gust_mph is None:
            brief.gust_mph = reading.gust_mph


def gather(
    entries: list[LedgerEntry],
    *,
    season: int,
    week: int,
    cache_dir: Path | None = None,
) -> dict[str, GameBrief]:
    """Fetch what can be fetched and build the briefs; each source fails on its own.

    The card is written whether or not any of this arrives: a network that is down
    costs the reader the colour, never the plays.
    """
    matchups = sorted({e.matchup for e in entries if e.season == season and e.week == week})
    schedule: pd.DataFrame | None = None
    ratings: RatingBook | None = None
    color: ColorBook | None = None
    try:
        games = nflverse.games()
        # The loader returns an empty, schema-less frame when both the network and
        # the cache are out; that is "no schedule", not a schedule of no games.
        schedule = games if not games.empty else None
    except Exception as exc:  # noqa: BLE001 - colour is optional
        log.warning("brief: schedule unavailable (%s)", exc)
    try:
        ratings = books_mod.as_of(season, week).ratings
    except Exception as exc:  # noqa: BLE001 - colour is optional
        log.warning("brief: ratings unavailable (%s)", exc)
    try:
        color = ESPNColor(cache_dir).fetch(season, week, matchups)
    except Exception as exc:  # noqa: BLE001 - colour is optional
        log.warning("brief: espn unavailable (%s)", exc)
    return build_briefs(
        entries,
        season=season,
        week=week,
        schedule=schedule,
        ratings=ratings,
        color=color,
        forecast=True,
    )
