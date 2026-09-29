"""Lineup-driven 5v5 strength (§5.7): isolated player impacts and the rebuild.

Three pure pieces, no I/O:

1. ``build_stints`` -- turn one game's shift chart + roster + MoneyPuck shot
   log into 5v5 stints: maximal intervals where the ten skaters (and both
   goalies) on the ice are unchanged, with the xG each side generated.
2. ``fit_rapm`` -- weighted ridge regression of stint xGF/60 on the ten
   skaters (an offense column for the attacking five, a defense column for the
   defending five, a home-ice intercept). The coefficient is a player's
   isolated impact in xG/60, net of linemates and opponents; that is what stops
   a centre's value living on in his wingers' on-ice numbers after he is
   scratched. Ridge ``lam`` and the extra EB ``k`` come from
   ``scripts/nhl/rapm_fit.py`` and are quoted in ``config.LineupParams``.
3. ``rebuild`` -- tonight's 5v5 xGF/60 and xGA/60 for a team from the players
   expected to dress::

       lam = team_rate + sum_dressed(share_i * impact_i) - sum_baseline(share_i * impact_i)

   ``team_rate`` is the §5.1 EB posterior and ``baseline`` is the TOI-share
   roster that produced it, so an unchanged healthy lineup returns the team rate
   exactly and a departed player simply drops out of the sum (no rescale, no
   decay that has to "notice" a deadline sell-off). Impacts are EB-shrunk by the
   player's 5v5 exposure before entering the sum, so a thin call-up moves
   nothing and a star moves a few percent, not a goal.

Nothing here prices a market or hand-codes a "star out" delta; the size of
every absence is whatever the fitted impacts say.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine_common.shrink import reliability_at
from nhl_engine.data.nhlapi import RosterSpot, Shift

PER_60 = 3600.0
REG_PERIOD = 1200


@dataclass(frozen=True)
class Stint:
    game_id: int
    period: int
    start: int
    end: int
    home: tuple[int, ...]
    away: tuple[int, ...]
    home_xg: float = 0.0
    away_xg: float = 0.0
    venue: str = ""  # home team code = arena; absorbs rink tracking bias in the fit

    @property
    def seconds(self) -> int:
        return self.end - self.start


def build_stints(
    game_id: int,
    shifts: Sequence[Shift],
    roster: Sequence[RosterSpot],
    shots: pd.DataFrame,
    *,
    home: str,
    away: str,
) -> list[Stint]:
    """5v5 stints (5 skaters + 1 goalie a side) in regulation, with xG per side.

    ``shots`` is this game's slice of the MoneyPuck shot log (``period``,
    ``time`` in game seconds, ``isHomeTeam``, ``xGoal``). A shot at second
    ``t`` belongs to the stint with ``start <= t < end``.
    """
    goalies = {r.player_id for r in roster if r.position == "G"}
    by_period: dict[int, list[Shift]] = defaultdict(list)
    for s in shifts:
        if 1 <= s.period <= 3:
            by_period[s.period].append(s)

    shot_rows = _shots_by_period(shots)
    out: list[Stint] = []
    for period, plist in sorted(by_period.items()):
        cuts = sorted({s.start for s in plist} | {s.end for s in plist})
        for t0, t1 in zip(cuts, cuts[1:], strict=False):
            on = [s for s in plist if s.start <= t0 and s.end >= t1]
            h_sk = tuple(
                sorted(s.player_id for s in on if s.team == home and s.player_id not in goalies)
            )
            a_sk = tuple(
                sorted(s.player_id for s in on if s.team == away and s.player_id not in goalies)
            )
            h_g = sum(1 for s in on if s.team == home and s.player_id in goalies)
            a_g = sum(1 for s in on if s.team == away and s.player_id in goalies)
            if len(h_sk) != 5 or len(a_sk) != 5 or h_g != 1 or a_g != 1:
                continue
            hx = ax = 0.0
            for t, is_home, xg in shot_rows.get(period, ()):
                if t0 <= t < t1:
                    if is_home:
                        hx += xg
                    else:
                        ax += xg
            out.append(Stint(game_id, period, t0, t1, h_sk, a_sk, hx, ax, home))
    return out


def _shots_by_period(shots: pd.DataFrame) -> dict[int, list[tuple[int, bool, float]]]:
    rows: dict[int, list[tuple[int, bool, float]]] = defaultdict(list)
    if shots.empty:
        return rows
    for period, t, is_home, xg in zip(
        shots["period"].astype(int),
        shots["time"].astype(int),
        shots["isHomeTeam"].astype(float),
        shots["xGoal"].astype(float),
        strict=True,
    ):
        if 1 <= period <= 3 and np.isfinite(xg):
            rows[period].append((t - (period - 1) * REG_PERIOD, bool(is_home), xg))
    return rows


# -- ridge / RAPM -----------------------------------------------------------------


@dataclass(frozen=True)
class Impact:
    player_id: int
    off: float  # isolated xGF/60 vs an average skater
    dfn: float  # isolated xGA/60 vs an average skater (negative = suppresses)
    toi: float  # 5v5 seconds in the fit

    @property
    def net(self) -> float:
        return self.off - self.dfn


@dataclass(frozen=True)
class RapmFit:
    intercept: float
    home_ice: float
    impacts: dict[int, Impact]
    lam: float
    stints: int
    seconds: float
    venues: dict[str, float] = field(default_factory=dict)  # xG/60 both sides, per arena


def fit_rapm(
    stints: Sequence[Stint],
    *,
    lam: float,
    min_seconds: float = 0.0,
    venue_effects: bool = False,
    venue_lam: float = 1000.0,
) -> RapmFit:
    """Weighted ridge of stint xG/60 on on-ice skaters; one row per stint per side.

    Design row for the home side attacking: +1 in each home skater's offense
    column, +1 in each away skater's defense column, home indicator 1, target
    ``home_xg / seconds * 3600``, weight ``seconds``. The away side mirrors it
    with home indicator 0. Intercept and home-ice are unpenalised; the player
    columns are shrunk by ``lam`` toward zero (= an average skater). Players with
    fewer than ``min_seconds`` are pooled into a "replacement" column so they
    cannot absorb variance.

    ``venue_effects`` adds one column per arena, +1 on *both* sides' rows of a
    stint played there, lightly penalised by ``venue_lam`` (they are collinear
    with the intercept otherwise; 1000 s against ~2e5 s per arena-season is a
    <1% shrink). Whatever a rink's
    scorers add to or shave off every shot's xG lands there instead of on the
    home team's skaters, who are identified from their road games
    (``scripts/nhl/arena_bias_study.py`` is the study that says whether it is
    needed; ``config.LineupParams`` records the choice).
    """
    if not stints:
        return RapmFit(0.0, 0.0, {}, lam, 0, 0.0)
    toi: dict[int, float] = defaultdict(float)
    for s in stints:
        for pid in s.home + s.away:
            toi[pid] += s.seconds
    players = sorted(pid for pid, sec in toi.items() if sec >= min_seconds)
    idx = {pid: i for i, pid in enumerate(players)}
    p = len(players)
    # columns: off[0:p] | def[p:2p] | replacement off, def | intercept | home
    n_off = p
    rep_off, rep_def = 2 * p, 2 * p + 1
    venues = sorted({s.venue for s in stints if s.venue}) if venue_effects else []
    v_idx = {v: 2 * p + 2 + i for i, v in enumerate(venues)}
    ncol = 2 * p + 2 + len(venues) + 2
    c_int, c_home = ncol - 2, ncol - 1

    xtwx = np.zeros((ncol, ncol))
    xtwy = np.zeros(ncol)
    total = 0.0
    for s in stints:
        w = float(s.seconds)
        if w <= 0:
            continue
        total += w
        for attackers, defenders, xg, is_home in (
            (s.home, s.away, s.home_xg, 1.0),
            (s.away, s.home, s.away_xg, 0.0),
        ):
            cols = [c_int]
            if is_home:
                cols.append(c_home)
            if s.venue in v_idx:
                cols.append(v_idx[s.venue])
            for pid in attackers:
                cols.append(idx[pid] if pid in idx else rep_off)
            for pid in defenders:
                cols.append(n_off + idx[pid] if pid in idx else rep_def)
            y = xg / w * PER_60
            c = np.array(cols)
            xtwx[np.ix_(c, c)] += w
            xtwy[c] += w * y
    penalty = np.full(ncol, lam)
    penalty[[c_int, c_home]] = 0.0
    for vc in v_idx.values():
        penalty[vc] = venue_lam
    beta = np.linalg.solve(xtwx + np.diag(penalty), xtwy)
    impacts = {
        pid: Impact(pid, float(beta[i]), float(beta[n_off + i]), toi[pid]) for pid, i in idx.items()
    }
    return RapmFit(
        float(beta[c_int]),
        float(beta[c_home]),
        impacts,
        lam,
        len(stints),
        total,
        {v: float(beta[vc]) for v, vc in v_idx.items()},
    )


# -- lineup rebuild -----------------------------------------------------------------


@dataclass(frozen=True)
class Lineup:
    """TOI shares of the skaters expected to dress; ``sum(shares) ~= 5``."""

    team: str
    shares: dict[int, float]
    source: str  # confirmed | projected | last_game | team_rate
    out: frozenset[int] = frozenset()

    @property
    def dressed(self) -> frozenset[int]:
        return frozenset(self.shares)


def toi_shares(stints: Iterable[Stint], team_is_home: Mapping[int, bool]) -> dict[int, float]:
    """Each skater's share of the team's 5v5 skater-seconds (sums to 5).

    ``team_is_home`` maps game id -> whether the team was home in that game.
    """
    sec: dict[int, float] = defaultdict(float)
    total = 0.0
    for s in stints:
        side = s.home if team_is_home.get(s.game_id) else s.away
        for pid in side:
            sec[pid] += s.seconds
        total += s.seconds
    if total <= 0:
        return {}
    return {pid: v / total for pid, v in sec.items()}


def drop_players(
    lineup: Mapping[int, float],
    out: Iterable[int],
    *,
    positions: Mapping[int, str] | None = None,
) -> dict[int, float]:
    """Remove ``out`` and hand their share pro rata to the remaining skaters.

    With ``positions`` (player -> C/L/R/D) the share stays inside the same
    group (forwards or defence) -- a missing centre is covered by forwards.
    Returns a new mapping; the total share is unchanged.
    """
    out = set(out)
    kept = {pid: sh for pid, sh in lineup.items() if pid not in out}
    lost = {pid: sh for pid, sh in lineup.items() if pid in out}
    if not lost or not kept:
        return kept

    def group(pid: int) -> str:
        pos = (positions or {}).get(pid, "")
        return "D" if pos == "D" else "F"

    for gone, share in lost.items():
        pool = [pid for pid in kept if positions is None or group(pid) == group(gone)] or list(kept)
        base = sum(kept[pid] for pid in pool)
        for pid in pool:
            kept[pid] += share * (kept[pid] / base if base > 0 else 1.0 / len(pool))
    return kept


@dataclass(frozen=True)
class Rebuild:
    team: str
    xgf60: float
    xga60: float
    team_xgf60: float
    team_xga60: float
    source: str
    turnover: float
    contributions: dict[int, tuple[float, float]] = field(default_factory=dict)

    @property
    def delta_for(self) -> float:
        return self.xgf60 - self.team_xgf60

    @property
    def delta_against(self) -> float:
        return self.xga60 - self.team_xga60


def shrunk(imp: Impact | None, k: float) -> tuple[float, float]:
    if imp is None:
        return 0.0, 0.0
    r = reliability_at(imp.toi, k)
    return imp.off * r, imp.dfn * r


def rebuild(
    team: str,
    *,
    team_xgf60: float,
    team_xga60: float,
    dressed: Mapping[int, float],
    baseline: Mapping[int, float],
    impacts: Mapping[int, Impact],
    k: float,
    source: str,
) -> Rebuild:
    """Tonight's 5v5 rates from the dressed lineup vs the baseline that made the team rate."""
    contributions: dict[int, tuple[float, float]] = {}
    d_for = d_against = 0.0
    for pid, share in dressed.items():
        off, dfn = shrunk(impacts.get(pid), k)
        contributions[pid] = (off * share, dfn * share)
        d_for += off * share
        d_against += dfn * share
    for pid, share in baseline.items():
        off, dfn = shrunk(impacts.get(pid), k)
        d_for -= off * share
        d_against -= dfn * share
    return Rebuild(
        team=team,
        xgf60=team_xgf60 + d_for,
        xga60=team_xga60 + d_against,
        team_xgf60=team_xgf60,
        team_xga60=team_xga60,
        source=source,
        turnover=turnover_share(baseline, dressed, impacts, k),
        contributions=contributions,
    )


def turnover_share(
    baseline: Mapping[int, float],
    dressed: Mapping[int, float],
    impacts: Mapping[int, Impact],
    k: float,
) -> float:
    """Diagnostic: fraction of the baseline's isolated |impact| mass not dressed tonight.

    Printed on the card and cut in the audit so a deadline sell-off can be seen
    repricing; it triggers nothing.
    """
    total = gone = 0.0
    for pid, share in baseline.items():
        off, dfn = shrunk(impacts.get(pid), k)
        mass = share * (abs(off) + abs(dfn))
        total += mass
        if pid not in dressed:
            gone += mass
    return gone / total if total > 0 else 0.0


__all__ = [
    "Impact",
    "Lineup",
    "RapmFit",
    "Rebuild",
    "Stint",
    "build_stints",
    "drop_players",
    "fit_rapm",
    "rebuild",
    "shrunk",
    "toi_shares",
    "turnover_share",
]
