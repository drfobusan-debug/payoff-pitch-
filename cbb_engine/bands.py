"""Graded-only CBB bands: flat 1u on every priced main-line row in a slice of the board.

Same bar as NHL and NBA (``engine_common.bands``): a band is called only at 150
games, 2 SE past what the market expects at the card's price, with both halves
agreeing. Nothing here touches pricing, gates or tiers. A row's context is what
was known before tip: the AP rank on the day's ESPN scoreboard, Torvik's rank and
adjusted-tempo third from the ``team_results`` table archived on or before the
slate, whether each team's last logged game (dated before the slate) was a loss,
days since its previous game, the first line captured that day, and each team's
season-to-date 2P% and 3P% gaps (made less allowed) once it has ``SHOOT_MIN_GAMES`` logged,
and its Torvik z-score (adjusted efficiency margin, AdjOE less AdjDE, in standard deviations
from the mean of that same archived table).

The z-score bands were cut after looking at the 11-13 ledger, so they count games from
``Z_SINCE`` only: the games that suggested them cannot also confirm them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date as Date
from datetime import timedelta

from cbb_engine import history
from cbb_engine.audit import pnl
from cbb_engine.data import capture, espn
from cbb_engine.data.vsin_splits import Public
from cbb_engine.output import teamstats
from engine_common.bands import BandRead
from engine_common.promotion import close_roi
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
SHOOT_MIN_GAMES = 10
Z_SINCE = "2026-11-14"
Z_AVG = 0.5
Z_AVG_WIDE = 0.75
Z_AVG_WIDEST = 1.0
Z_HIGH = 1.0
Z_ELITE = 2.0


@dataclass(frozen=True)
class Team:
    ap: int | None = None
    net_rank: int | None = None  # Torvik's rank as archived on or before the day
    slow: bool | None = None  # slowest third by Torvik adjusted tempo as of the day
    off_loss: bool | None = None  # last logged game before the day was a loss
    power: bool | None = None
    rest: int | None = None  # days since the previous game
    pace: str | None = None  # "slow" / "mid" / "fast": season-to-date tempo third
    two: float | None = None  # 2P% made less 2P% allowed, season to date (None under SHOOT_MIN_GAMES)
    three: float | None = None  # 3P% made less 3P% allowed, same window
    z: float | None = None  # Torvik AdjEM z-score in the table archived on or before the day


@dataclass(frozen=True)
class GameCtx:
    home: Team = Team()
    away: Team = Team()
    neutral: bool = False
    conference: bool = False
    opener: Mapping[tuple[str, str], float] = field(default_factory=dict)  # (market, side) -> first line
    public: Mapping[tuple[str, str], Public] = field(default_factory=dict)  # (market, side) -> VSiN shares before tip
    ats_fav: str | None = None  # the game's spread favourite across both sides' main rows (set by read_bands)


Pred = Callable[[ledger.Row, GameCtx], bool]


def _me(r: ledger.Row, c: GameCtx) -> Team:
    return c.home if r.side == "home" else c.away


def _them(r: ledger.Row, c: GameCtx) -> Team:
    return c.away if r.side == "home" else c.home


def _fav(r: ledger.Row, c: GameCtx) -> bool:
    if r.market == "game_ats":
        if c.ats_fav is not None:
            return r.side == c.ats_fav
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


def _rlm(r: ledger.Row, c: GameCtx) -> bool:
    """The number moved a point or more toward this side while the tickets leaned the other way.

    Both halves are required: a move with no split on record is steam, not a reverse move,
    and a ticket minority with no move is just the quiet side."""
    pub = c.public.get((r.market, r.side))
    return pub is not None and pub.majority is False and _moved(r, c)


def _b2b(t: Team) -> bool:
    return t.rest == 1


def _weak(c: GameCtx) -> bool:
    return all(t.net_rank is not None and t.net_rank > WEAK_RANK for t in (c.home, c.away))


def _both_off_loss(c: GameCtx) -> bool:
    return bool(c.home.off_loss and c.away.off_loss)


def _inside_not_outside(r: ledger.Row, c: GameCtx) -> bool:
    """This side's 2P% gap beats the opponent's while its 3P% gap trails: a better inside team that has shot worse from three."""
    me, them = _me(r, c), _them(r, c)
    if None in (me.two, me.three, them.two, them.three):
        return False
    return me.two > them.two and me.three < them.three  # type: ignore[operator]


def _avg(t: Team, width: float = Z_AVG) -> bool:
    return t.z is not None and abs(t.z) <= width


def _avg_game(c: GameCtx, width: float) -> bool:
    return _avg(c.home, width) and _avg(c.away, width)


def _high(t: Team, lo: float, hi: float = float("inf")) -> bool:
    return t.z is not None and lo <= t.z < hi


def _high_vs_avg(c: GameCtx, lo: float, hi: float = float("inf"), width: float = Z_AVG) -> str | None:
    """The side ("home"/"away") of a team with ``lo <= z < hi`` facing a team within ``width`` of average."""
    if _high(c.home, lo, hi) and _avg(c.away, width):
        return "home"
    if _high(c.away, lo, hi) and _avg(c.home, width):
        return "away"
    return None


def _z(pred: Pred) -> Pred:
    """A z-score band reads only games from ``Z_SINCE``."""
    return lambda r, c: r.slate_date >= Z_SINCE and pred(r, c)


def _under(r: ledger.Row) -> bool:
    return r.market == "game_total" and r.side == "under"


def _over(r: ledger.Row) -> bool:
    return r.market == "game_total" and r.side == "over"


def _side(market: str, side: str, fav: bool | None = None) -> Pred:
    return lambda r, c: r.market == market and r.side == side and (fav is None or _fav(r, c) == fav)


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
    ("AP-ranked road favourite ATS", lambda r, c: _ats(r) and r.side == "away" and not c.neutral and _me(r, c).ap is not None and _fav(r, c)),
    ("AP-ranked road underdog ATS", lambda r, c: _ats(r) and r.side == "away" and not c.neutral and _me(r, c).ap is not None and not _fav(r, c)),
    ("neutral site favourite ATS", lambda r, c: _ats(r) and c.neutral and _fav(r, c)),
    ("neutral site underdog ATS", lambda r, c: _ats(r) and c.neutral and not _fav(r, c)),
    ("line moved >= 1 toward the side ATS", lambda r, c: _ats(r) and _moved(r, c)),
    ("line moved >= 1 toward the side total", lambda r, c: r.market == "game_total" and _moved(r, c)),
    ("reverse line move (>= 1 against the ticket majority) ATS", lambda r, c: _ats(r) and _rlm(r, c)),
    ("reverse line move (>= 1 against the ticket majority) favourite ATS", lambda r, c: _ats(r) and _rlm(r, c) and _fav(r, c)),
    ("reverse line move (>= 1 against the ticket majority) underdog ATS", lambda r, c: _ats(r) and _rlm(r, c) and not _fav(r, c)),
    ("reverse line move (>= 1 against the ticket majority) total", lambda r, c: r.market == "game_total" and _rlm(r, c)),
    ("team on a back-to-back ATS", lambda r, c: _ats(r) and _b2b(_me(r, c)) and not _b2b(_them(r, c))),
    ("rested (2+ days) vs a back-to-back ATS", lambda r, c: _ats(r) and (_me(r, c).rest or 0) >= 2 and _b2b(_them(r, c))),
    ("a team on a back-to-back: Under", lambda r, c: r.market == "game_total" and r.side == "under" and (_b2b(c.home) or _b2b(c.away))),
    ("weak game (both Torvik rank > 200) favourite ATS", lambda r, c: _ats(r) and _weak(c) and _fav(r, c)),
    ("weak game (both Torvik rank > 200) underdog ATS", lambda r, c: _ats(r) and _weak(c) and not _fav(r, c)),
    ("bounce back: side off a loss ATS", lambda r, c: _ats(r) and bool(_me(r, c).off_loss) and not _them(r, c).off_loss),
    ("bounce back: both off a loss, favourite ATS", lambda r, c: _ats(r) and _both_off_loss(c) and _fav(r, c)),
    ("bounce back: both off a loss, underdog ATS", lambda r, c: _ats(r) and _both_off_loss(c) and not _fav(r, c)),
    ("mid-major vs power conference: mid-major ATS", lambda r, c: _ats(r) and _me(r, c).power is False and _them(r, c).power is True),
    ("conference game underdog ATS", lambda r, c: _ats(r) and c.conference and not _fav(r, c)),
    ("non-conference underdog ATS", lambda r, c: _ats(r) and not c.conference and not _fav(r, c)),
    ("conference game Under", lambda r, c: r.market == "game_total" and r.side == "under" and c.conference),
    ("non-conference Under", lambda r, c: r.market == "game_total" and r.side == "under" and not c.conference),
    ("every Over", _side("game_total", "over")),
    ("every Under", _side("game_total", "under")),
    ("both teams slow tempo: Under", lambda r, c: r.market == "game_total" and r.side == "under" and bool(c.home.slow and c.away.slow)),
    ("big spread (>= 15) favourite ATS", lambda r, c: _ats(r) and r.line is not None and r.line <= -BIG_SPREAD),
    ("big spread (>= 15) underdog ATS", lambda r, c: _ats(r) and r.line is not None and r.line >= BIG_SPREAD),
    ("exploratory: underdog better 2P%, worse 3P% (10+ games) ATS", lambda r, c: _ats(r) and not _fav(r, c) and _inside_not_outside(r, c)),
    ("z: average game (both |z| <= 0.5) underdog ATS", _z(lambda r, c: _ats(r) and _avg_game(c, Z_AVG) and not _fav(r, c))),
    ("z: average game (both |z| <= 0.5) home underdog ATS",
     _z(lambda r, c: _ats(r) and r.side == "home" and _avg_game(c, Z_AVG) and not _fav(r, c))),
    ("z: average game (both |z| <= 0.75) underdog ATS", _z(lambda r, c: _ats(r) and _avg_game(c, Z_AVG_WIDE) and not _fav(r, c))),
    ("z: average game (both |z| <= 1) underdog ATS", _z(lambda r, c: _ats(r) and _avg_game(c, Z_AVG_WIDEST) and not _fav(r, c))),
    ("z: average game (both |z| <= 0.5) Under", _z(lambda r, c: _under(r) and _avg_game(c, Z_AVG))),
    ("z: high (1 <= z < 2) vs average (|z| <= 0.5): average team ATS",
     _z(lambda r, c: _ats(r) and _high_vs_avg(c, Z_HIGH, Z_ELITE) not in (None, r.side))),
    ("z: high (1 <= z < 2) vs average (|z| <= 0.5): high team ML",
     _z(lambda r, c: r.market == "game_ml" and _high_vs_avg(c, Z_HIGH, Z_ELITE) == r.side)),
    ("z: high (1 <= z < 2) vs average (|z| <= 0.5): average team ML",
     _z(lambda r, c: r.market == "game_ml" and _high_vs_avg(c, Z_HIGH, Z_ELITE) not in (None, r.side))),
    ("z: high (1 <= z < 2) vs average (|z| <= 0.5): Under", _z(lambda r, c: _under(r) and _high_vs_avg(c, Z_HIGH, Z_ELITE) is not None)),
    ("z: high (z >= 1) vs average (|z| <= 0.75): Under",
     _z(lambda r, c: _under(r) and _high_vs_avg(c, Z_HIGH, width=Z_AVG_WIDE) is not None)),
    ("z: elite (z >= 2) vs average (|z| <= 0.5): elite team ATS", _z(lambda r, c: _ats(r) and _high_vs_avg(c, Z_ELITE) == r.side)),
    ("z: elite (z >= 2) vs average (|z| <= 0.5): Over", _z(lambda r, c: _over(r) and _high_vs_avg(c, Z_ELITE) is not None)),
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


def ats_favourites(main: Iterable[ledger.Row]) -> dict[str, str]:
    """Each game's spread favourite from its two main rows, so a game has one favourite and one underdog.

    Two execution books can disagree on who is favoured (DraftKings Albany -1.5, BetMGM Dartmouth -1.5),
    and each side's best line is then +1.5. The lower line is the favourite; at equal lines the side
    with the higher no-vig cover chance is. A game with one side recorded, or a dead tie, is left out."""
    by_game: dict[str, dict[str, ledger.Row]] = defaultdict(dict)
    for r in main:
        if r.market == "game_ats" and r.line is not None and r.side in ("home", "away"):
            by_game[r.event_id][r.side] = r
    out: dict[str, str] = {}
    for eid, sides in by_game.items():
        if len(sides) < 2:
            continue
        h, a = sides["home"], sides["away"]
        key_h, key_a = (h.line, -(h.p_fair or 0.0)), (a.line, -(a.p_fair or 0.0))
        if key_h != key_a:
            out[eid] = "home" if key_h < key_a else "away"
    return out


def read_bands(rows: Iterable[ledger.Row], ctx: Mapping[str, GameCtx]) -> list[BandRead]:
    main = [r for r in main_rows(rows) if r.event_id in ctx]
    favs = ats_favourites(main)
    ctx = {eid: replace(c, ats_fav=favs.get(eid)) for eid, c in ctx.items()}
    out: list[BandRead] = []
    for name, hit in BANDS:
        rs = [r for r in main if hit(r, ctx[r.event_id])]
        game: dict[tuple[str, str], float] = {}
        exp: dict[tuple[str, str], float] = {}
        for r in rs:
            k = (r.slate_date, r.event_id)
            game[k] = game.get(k, 0.0) + (r.pnl or 0.0)
            exp[k] = exp.get(k, 0.0) + market_expected(r)
        closes = [c for c in (close_implied(r) for r in rs) if c is not None]
        out.append(BandRead(name, [r.pnl or 0.0 for r in rs], list(game.values()), [market_expected(r) for r in rs], list(exp.values()), closes))
    return out


def close_implied(r: ledger.Row) -> float | None:
    """Close-implied ROI of the row's price at a pre-tip close; None for pushes or no close."""
    if r.outcome == "push" or r.close_consensus is None or r.close_status != "pre_tip":
        return None
    if r.american is None:
        return None
    return close_roi(r.close_consensus, r.american)


def render(reads: Sequence[BandRead]) -> str:
    lines = ["Bands (graded only; flat 1u per main-line row; excess = units less what the no-vig market expects)"]
    for b in reads:
        h1, h2 = b.halves
        lines.append(
            f"  {b.name:<48} n={b.n:<5} {b.record:<9} {b.units:+7.1f}u roi {100 * b.roi:+6.1f}%"
            f"  excess {b.excess:+6.1f} ± {b.se:4.1f}  halves {h1:+5.1f}/{h2:+5.1f}  {b.price_read}  {b.verdict}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class LogGame:
    day: Date
    team: str
    poss: float
    pts: float
    opp_pts: float
    two: tuple[float, float] = (0.0, 0.0)  # 2PM, 2PA
    three: tuple[float, float] = (0.0, 0.0)  # 3PM, 3PA
    opp_two: tuple[float, float] = (0.0, 0.0)
    opp_three: tuple[float, float] = (0.0, 0.0)


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
        out.append(LogGame(day, row[2], poss, own["pts"], opp["pts"], *(_shots(own) + _shots(opp))))
    return out


def _shots(box: Mapping[str, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    return (box["fgm"] - box["tpm"], box["fga"] - box["tpa"]), (box["tpm"], box["tpa"])


def shooting(logs: Sequence[LogGame], before: Date) -> dict[str, tuple[float, float]]:
    """Per Torvik team with ``SHOOT_MIN_GAMES`` logged before ``before``: (2P% gap, 3P% gap), each made less allowed,
    pooled over shots."""
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0] * 9)
    for g in logs:
        if g.day < before:
            a = acc[g.team]
            a[0] += 1
            for i, v in enumerate((*g.two, *g.three, *g.opp_two, *g.opp_three), 1):
                a[i] += v
    out: dict[str, tuple[float, float]] = {}
    for t, (n, m2, a2, m3, a3, om2, oa2, om3, oa3) in acc.items():
        if n >= SHOOT_MIN_GAMES and a2 and a3 and oa2 and oa3:
            out[t] = (m2 / a2 - om2 / oa2, m3 / a3 - om3 / oa3)
    return out


def last_results(logs: Sequence[LogGame], before: Date) -> dict[str, bool]:
    """Per Torvik team, whether its last logged game dated before ``before`` was a loss."""
    last: dict[str, LogGame] = {}
    for g in logs:
        if g.day < before and (g.team not in last or g.day > last[g.team].day):
            last[g.team] = g
    return {t: g.pts < g.opp_pts for t, g in last.items()}


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


def torvik_z(results: Iterable[Mapping[str, str]]) -> dict[str, float]:
    """Each team's adjusted efficiency margin (``adjoe`` less ``adjde``) in standard deviations from the table's mean."""
    em: dict[str, float] = {}
    for r in results:
        try:
            em[r.get("team", "")] = float(r.get("adjoe", "")) - float(r.get("adjde", ""))
        except ValueError:
            continue
    if len(em) < 2:
        return {}
    mean = sum(em.values()) / len(em)
    sd = (sum((v - mean) ** 2 for v in em.values()) / len(em)) ** 0.5
    return {t: (v - mean) / sd for t, v in em.items()} if sd else {}


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
    public: Mapping[str, Mapping[tuple[str, str], Public]] | None = None,
    zs: Mapping[str, float] | None = None,
) -> dict[str, GameCtx]:
    """Pre-tip context per event from the day's ESPN board (AP rank, venue, conference flag),
    earlier boards (rest), Torvik logs before ``day`` (last game a loss), conference membership,
    ``tempo`` (each Torvik team's adjusted-tempo third as of ``day``) and ``ranks`` (Torvik's rank
    archived on or before ``day``) and ``public`` (each event's VSiN ticket/handle shares from the
    last capture before its tip) and ``zs`` (each Torvik team's AdjEM z-score, ``torvik_z``, from the same
    archived table as ``ranks``). A season-to-date rank from one or two logged games is not a
    rank, so nothing here is computed from the logs' margins."""
    last = last_results(logs, day)
    shoot = shooting(logs, day)
    thirds = tempo or {}
    tv_rank = ranks or {}
    tv_z = zs or {}
    tv = sorted(set(last) | set(confs) | set(thirds) | set(tv_rank) | set(tv_z))
    games = boards.get(day, ())
    out: dict[str, GameCtx] = {}

    def team(name: str, espn_name: str | None, ap: int | None) -> Team:
        t = teamstats.match_team(name, tv)
        pace = thirds.get(t) if t else None
        conf = confs.get(t) if t else None
        rest = rest_days(espn_name, day, boards) if espn_name else None
        two, three = shoot.get(t, (None, None)) if t else (None, None)
        return Team(ap=ap, net_rank=tv_rank.get(t) if t else None, slow=(pace == "slow") if pace else None,
                    off_loss=last.get(t) if t else None, power=(conf in POWER) if conf else None, rest=rest, pace=pace,
                    two=two, three=three, z=tv_z.get(t) if t else None)

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
            public=dict((public or {}).get(eid, {})),
        )
    return out


__all__ = [
    "BANDS", "GameCtx", "LogGame", "Team", "context", "main_lines", "main_rows", "market_expected", "one_slate_per_game",
    "ats_favourites", "last_results", "openers", "parse_logs", "read_bands", "render", "rest_days", "shooting", "tempo_thirds", "tip_day",
    "torvik_ranks", "torvik_z",
]
