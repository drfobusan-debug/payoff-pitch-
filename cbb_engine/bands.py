"""Graded-only CBB bands: flat 1u on every priced main-line row in a slice of the board.

Same bar as NHL and NBA (``engine_common.bands``): a band is called only at 150
games, 2 SE past what the market expects at the card's price, with both halves
agreeing. Nothing here touches pricing, gates or tiers. A row's context is what
was known before tip: the AP rank on the day's ESPN scoreboard, season-to-date
net rating and tempo from Torvik's game logs dated before the slate, days since
each team's previous game, and the first line captured that day.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta

from cbb_engine import history
from cbb_engine.audit import pnl
from cbb_engine.data import capture, espn
from cbb_engine.output import teamstats
from engine_common.bands import BandRead
from engine_runtime import ledger, quotes

MAIN = ("game_ats", "game_total", "game_ml")
POWER = frozenset({"ACC", "B10", "B12", "SEC", "BE"})
WEAK_RANK = 200
SLOW_SHARE = 1 / 3
BIG_SPREAD = 15.0
LONG_SHOT = 0.25
HEAVY_FAV = 0.85
MOVE_PTS = 1.0
REST_LOOKBACK = 10


@dataclass(frozen=True)
class Team:
    ap: int | None = None
    net_rank: int | None = None
    slow: bool | None = None
    power: bool | None = None
    rest: int | None = None  # days since the previous game
    pace: str | None = None  # "slow" / "mid" / "fast": season-to-date tempo third


@dataclass(frozen=True)
class GameCtx:
    home: Team = Team()
    away: Team = Team()
    neutral: bool = False
    conference: bool = False
    opener: Mapping[tuple[str, str], float] = field(default_factory=dict)  # (market, side) -> first line


Pred = Callable[[ledger.Row, GameCtx], bool]


def _me(r: ledger.Row, c: GameCtx) -> Team:
    return c.home if r.side == "home" else c.away


def _them(r: ledger.Row, c: GameCtx) -> Team:
    return c.away if r.side == "home" else c.home


def _fav(r: ledger.Row) -> bool:
    if r.market == "game_ats":
        return r.line is not None and r.line < 0
    return (r.p_fair or 0.0) > 0.5


def _ats(r: ledger.Row) -> bool:
    return r.market == "game_ats"


def _moved(r: ledger.Row, c: GameCtx) -> bool:
    first = c.opener.get((r.market, r.side))
    if first is None or r.line is None:
        return False
    toward = r.line - first if r.side == "over" else first - r.line
    return toward >= MOVE_PTS


def _b2b(t: Team) -> bool:
    return t.rest == 1


def _weak(c: GameCtx) -> bool:
    return all(t.net_rank is not None and t.net_rank > WEAK_RANK for t in (c.home, c.away))


def _side(market: str, side: str, fav: bool | None = None) -> Pred:
    return lambda r, c: r.market == market and r.side == side and (fav is None or _fav(r) == fav)


BANDS: tuple[tuple[str, Pred], ...] = (
    ("home favourite ATS", _side("game_ats", "home", True)),
    ("home underdog ATS", _side("game_ats", "home", False)),
    ("away favourite ATS", _side("game_ats", "away", True)),
    ("away underdog ATS", _side("game_ats", "away", False)),
    ("home favourite ML", _side("game_ml", "home", True)),
    ("home underdog ML", _side("game_ml", "home", False)),
    ("away favourite ML", _side("game_ml", "away", True)),
    ("away underdog ML", _side("game_ml", "away", False)),
    ("ML long shot (no-vig < 25%)", lambda r, c: r.market == "game_ml" and (r.p_fair or 1.0) < LONG_SHOT),
    ("ML heavy favourite (no-vig >= 85%)", lambda r, c: r.market == "game_ml" and (r.p_fair or 0.0) >= HEAVY_FAV),
    ("AP-ranked road favourite ATS", lambda r, c: _ats(r) and r.side == "away" and not c.neutral and _me(r, c).ap is not None and _fav(r)),
    ("AP-ranked road underdog ATS", lambda r, c: _ats(r) and r.side == "away" and not c.neutral and _me(r, c).ap is not None and not _fav(r)),
    ("neutral site favourite ATS", lambda r, c: _ats(r) and c.neutral and _fav(r)),
    ("neutral site underdog ATS", lambda r, c: _ats(r) and c.neutral and not _fav(r)),
    ("line moved >= 1 toward the side ATS", lambda r, c: _ats(r) and _moved(r, c)),
    ("line moved >= 1 toward the side total", lambda r, c: r.market == "game_total" and _moved(r, c)),
    ("team on a back-to-back ATS", lambda r, c: _ats(r) and _b2b(_me(r, c)) and not _b2b(_them(r, c))),
    ("rested (2+ days) vs a back-to-back ATS", lambda r, c: _ats(r) and (_me(r, c).rest or 0) >= 2 and _b2b(_them(r, c))),
    ("a team on a back-to-back: Under", lambda r, c: r.market == "game_total" and r.side == "under" and (_b2b(c.home) or _b2b(c.away))),
    ("weak game (both net rank > 200) favourite ATS", lambda r, c: _ats(r) and _weak(c) and _fav(r)),
    ("weak game (both net rank > 200) underdog ATS", lambda r, c: _ats(r) and _weak(c) and not _fav(r)),
    ("mid-major vs power conference: mid-major ATS", lambda r, c: _ats(r) and _me(r, c).power is False and _them(r, c).power is True),
    ("conference game underdog ATS", lambda r, c: _ats(r) and c.conference and not _fav(r)),
    ("non-conference underdog ATS", lambda r, c: _ats(r) and not c.conference and not _fav(r)),
    ("conference game Under", lambda r, c: r.market == "game_total" and r.side == "under" and c.conference),
    ("non-conference Under", lambda r, c: r.market == "game_total" and r.side == "under" and not c.conference),
    ("every Over", _side("game_total", "over")),
    ("every Under", _side("game_total", "under")),
    ("both teams slow tempo: Under", lambda r, c: r.market == "game_total" and r.side == "under" and bool(c.home.slow and c.away.slow)),
    ("big spread (>= 15) favourite ATS", lambda r, c: _ats(r) and r.line is not None and r.line <= -BIG_SPREAD),
    ("big spread (>= 15) underdog ATS", lambda r, c: _ats(r) and r.line is not None and r.line >= BIG_SPREAD),
)


def one_slate_per_game(rows: Iterable[ledger.Row]) -> list[ledger.Row]:
    """Keep each (game, market, side) from one slate only, so a game captured on several daily slates counts once.

    The slate of its own US tip day wins; failing that the latest slate before it, else the earliest.
    """
    by_bet: dict[tuple[str, str, str], list[ledger.Row]] = defaultdict(list)
    for r in rows:
        by_bet[(r.event_id, r.market, r.side)].append(r)
    out: list[ledger.Row] = []
    for rs in by_bet.values():
        slates = sorted({r.slate_date for r in rs})
        tip = tip_day(max(rs, key=lambda r: r.slate_date).tip_utc)
        day = tip.isoformat() if tip else ""
        before = [d for d in slates if d <= day]
        keep = before[-1] if before else slates[0]
        out += [r for r in rs if r.slate_date == keep]
    return out


def main_rows(rows: Iterable[ledger.Row]) -> list[ledger.Row]:
    """One graded row per (game, market, side) from the game's own slate: the most-quoted line, median price."""
    groups: dict[tuple[str, str, str, str], list[ledger.Row]] = defaultdict(list)
    graded = [
        r for r in rows
        if r.bet_of_record and r.market in MAIN and r.outcome in ("win", "loss", "push") and r.pnl is not None and r.p_fair
    ]
    for r in one_slate_per_game(graded):
        groups[(r.slate_date, r.event_id, r.market, r.side)].append(r)
    out: list[ledger.Row] = []
    for rs in groups.values():
        by_line: dict[float | None, list[ledger.Row]] = defaultdict(list)
        for r in rs:
            by_line[r.line].append(r)
        pick = max(by_line.values(), key=len)
        pick.sort(key=lambda r: pnl(r, "win") or 0.0)
        out.append(pick[len(pick) // 2])
    return sorted(out, key=lambda r: (r.slate_date, r.tip_utc, r.event_id, r.market, r.side))


def market_expected(r: ledger.Row) -> float:
    """Flat 1u at the row's price settled at the no-vig consensus; a push expects nothing."""
    if r.outcome == "push" or r.p_fair is None:
        return 0.0
    return r.p_fair * (pnl(r, "win") or 0.0) - (1.0 - r.p_fair)


def read_bands(rows: Iterable[ledger.Row], ctx: Mapping[str, GameCtx]) -> list[BandRead]:
    main = [r for r in main_rows(rows) if r.event_id in ctx]
    out: list[BandRead] = []
    for name, hit in BANDS:
        rs = [r for r in main if hit(r, ctx[r.event_id])]
        game: dict[tuple[str, str], float] = {}
        exp: dict[tuple[str, str], float] = {}
        for r in rs:
            k = (r.slate_date, r.event_id)
            game[k] = game.get(k, 0.0) + (r.pnl or 0.0)
            exp[k] = exp.get(k, 0.0) + market_expected(r)
        out.append(BandRead(name, [r.pnl or 0.0 for r in rs], list(game.values()), [market_expected(r) for r in rs], list(exp.values())))
    return out


def render(reads: Sequence[BandRead]) -> str:
    lines = ["Bands (graded only; flat 1u per main-line row; excess = units less what the no-vig market expects)"]
    for b in reads:
        h1, h2 = b.halves
        lines.append(
            f"  {b.name:<48} n={b.n:<5} {b.record:<9} {b.units:+7.1f}u roi {100 * b.roi:+6.1f}%"
            f"  excess {b.excess:+6.1f} ± {b.se:4.1f}  halves {h1:+5.1f}/{h2:+5.1f}  {b.verdict}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class LogGame:
    day: Date
    team: str
    poss: float
    pts: float
    opp_pts: float


def parse_logs(logs: Iterable[list[str]]) -> list[LogGame]:
    """Torvik ``getgamestats`` rows reduced to what the bands read."""
    out: list[LogGame] = []
    for row in logs:
        if len(row) < 30:
            continue
        day = teamstats._date(row[0])
        parsed = teamstats._box(row[29], row[2])
        if day is None or parsed is None:
            continue
        own, opp, poss = parsed
        out.append(LogGame(day, row[2], poss, own["pts"], opp["pts"]))
    return out


def profile(logs: Sequence[LogGame], before: Date) -> dict[str, tuple[int | None, bool | None]]:
    """Season-to-date (net-rating rank, slowest-third tempo) per Torvik team, games before ``before``."""
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for g in logs:
        if g.day < before:
            a = acc[g.team]
            a[0] += 1
            a[1] += g.poss
            a[2] += g.pts
            a[3] += g.opp_pts
    net = teamstats._ranks({t: 100 * (a[2] - a[3]) / a[1] for t, a in acc.items() if a[1]}, high_is_good=True)
    pace = teamstats._ranks({t: a[1] / a[0] for t, a in acc.items() if a[0]}, high_is_good=False)
    slow_cut = len(pace) * SLOW_SHARE
    return {
        t: (net[t].rank if t in net else None, (pace[t].rank or 0) <= slow_cut if t in pace and pace[t].rank else None)
        for t in acc
    }


def tempo_thirds(results: Iterable[Mapping[str, str]]) -> dict[str, str]:
    """Torvik adjusted tempo (``adjt``, team_results) cut into thirds: "slow" / "mid" / "fast"."""
    pace: dict[str, float] = {}
    for r in results:
        try:
            pace[r.get("team", "")] = float(r.get("adjt", ""))
        except ValueError:
            continue
    ranked = sorted(pace, key=lambda t: pace[t])
    n = len(ranked)
    return {
        t: "slow" if i < n * SLOW_SHARE else "fast" if i >= n * (1 - SLOW_SHARE) else "mid"
        for i, t in enumerate(ranked)
    }


def torvik_ranks(results: Iterable[Mapping[str, str]]) -> dict[str, int]:
    """Torvik's own rank per team from an archived ``team_results`` table (pre-season until games land)."""
    out: dict[str, int] = {}
    for r in results:
        try:
            out[r.get("team", "")] = int(r.get("rank", ""))
        except ValueError:
            continue
    return out


def rest_days(team: str, day: Date, boards: Mapping[Date, Sequence[espn.Game]]) -> int | None:
    """Days since ``team``'s (ESPN name) previous game; None when no board in the lookback has one."""
    for back in range(1, REST_LOOKBACK + 1):
        games = boards.get(day - timedelta(days=back))
        if games and any(team in (g.home, g.away) for g in games):
            return back
    return None


def openers(shards: Iterable[quotes.Quote]) -> dict[str, dict[tuple[str, str], float]]:
    """Each event's first captured consensus line per (market, side), from the earliest pass quoting it."""
    by_pass: dict[str, list[quotes.Quote]] = defaultdict(list)
    for q in shards:
        if q.market in ("game_ats", "game_total"):
            by_pass[q.pass_id].append(q)
    out: dict[str, dict[tuple[str, str], float]] = {}
    for qs in sorted(by_pass.values(), key=lambda v: min(q.captured_at for q in v)):
        for eid in sorted({q.event_id for q in qs} - set(out)):
            out[eid] = main_lines([q for q in qs if q.event_id == eid])
    return out


def main_lines(qs: Sequence[quotes.Quote]) -> dict[tuple[str, str], float]:
    """Median posted line per (market, side) across books."""
    lines: dict[tuple[str, str], list[float]] = defaultdict(list)
    for q in qs:
        if q.line is not None:
            lines[(q.market, q.side)].append(q.line)
    return {k: sorted(v)[len(v) // 2] for k, v in lines.items()}


def tip_day(tip_utc: str) -> Date | None:
    """The US calendar day a game tips on, so pre-tip stats can stop the day before it."""
    day = capture.slate_date_of(tip_utc)
    return Date.fromisoformat(day) if day else None


def context(
    teams: Mapping[str, tuple[str, str]],
    day: Date,
    boards: Mapping[Date, Sequence[espn.Game]],
    logs: Sequence[LogGame],
    confs: Mapping[str, str],
    opener: Mapping[str, Mapping[tuple[str, str], float]] | None = None,
    tempo: Mapping[str, str] | None = None,
    ranks: Mapping[str, int] | None = None,
) -> dict[str, GameCtx]:
    """Pre-tip context per event from the day's ESPN board (rank, venue, conference flag),
    earlier boards (rest), Torvik logs before ``day`` (net rank, tempo), conference membership and
    ``tempo`` (each Torvik team's adjusted-tempo third as of ``day``). ``ranks`` is Torvik's archived
    rank as of ``day``; it stands in for the net rank while a team has no logged game yet."""
    prof = profile(logs, day)
    thirds = tempo or {}
    tv_rank = ranks or {}
    tv = sorted(set(prof) | set(confs) | set(thirds) | set(tv_rank))
    games = boards.get(day, ())
    out: dict[str, GameCtx] = {}

    def team(name: str, espn_name: str | None, ap: int | None) -> Team:
        t = teamstats.match_team(name, tv)
        net, slow = prof.get(t, (None, None)) if t else (None, None)
        if net is None and t:
            net = tv_rank.get(t)
        conf = confs.get(t) if t else None
        rest = rest_days(espn_name, day, boards) if espn_name else None
        return Team(ap, net, slow, (conf in POWER) if conf else None, rest, thirds.get(t) if t else None)

    for eid, (home, away) in teams.items():
        hit = history.match_oriented(home, away, games)
        g, swapped = hit if hit else (None, False)
        h_name, a_name = (g.away, g.home) if g and swapped else (g.home, g.away) if g else (None, None)
        h_rank, a_rank = (g.away_rank, g.home_rank) if g and swapped else (g.home_rank, g.away_rank) if g else (None, None)
        out[eid] = GameCtx(
            home=team(home, h_name, h_rank),
            away=team(away, a_name, a_rank),
            neutral=bool(g and g.neutral),
            conference=bool(g and g.conference_game),
            opener=dict((opener or {}).get(eid, {})),
        )
    return out


__all__ = [
    "BANDS", "GameCtx", "LogGame", "Team", "context", "main_lines", "main_rows", "market_expected", "one_slate_per_game",
    "openers", "parse_logs", "profile", "read_bands", "render", "rest_days", "tempo_thirds", "tip_day", "torvik_ranks",
]
