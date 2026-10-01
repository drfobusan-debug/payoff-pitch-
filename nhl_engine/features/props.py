"""Per-player rates for tonight from NHL API game logs (master plan §5.5).

Everything is a shrunk rate per 60 of the player's own ice time, so the game
sim (score state, opponent) can scale it, and a shrunk TOI to multiply it by::

    toi      = w_recent x mean(last 10 TOI) + (1 - w_recent) x season mean,
               the season mean itself shrunk to last season's over k_toi games
    rate/60  = (events + k x prior/60) / (minutes + k)        k in minutes
    prior/60 = last season's rate shrunk the same way to the league positional mean

The league positional means are pooled from whoever is on the slate (hundreds
of players), falling back to 2024-25 constants on opening night. Nothing is
conditioned on the opponent here -- that is the model's job.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date as Date

from nhl_engine.data.skaters import GoalieGame, SkaterGame

PER_60 = 3600.0

# 2023-24 pooled rates per 60 of own TOI (NHL API game logs, top-450 TOI skaters;
# scripts/nhl/props_study.py).
LEAGUE_F = {"sog60": 7.71, "g60": 0.96, "a60": 1.32, "pts60": 2.28, "ppp_pg": 0.16}
LEAGUE_D = {"sog60": 4.32, "g60": 0.24, "a60": 0.90, "pts60": 1.14, "ppp_pg": 0.08}
LEAGUE_SV = 0.903


@dataclass(frozen=True)
class PropShrink:
    """Minutes (or games) of prior weight per rate; study-fitted defaults."""

    k_toi_games: float = 8.0
    recent_games: int = 10
    w_recent: float = 0.35
    k_sog_min: float = 350.0
    k_g_min: float = 1500.0
    k_a_min: float = 1500.0
    k_pts_min: float = 1000.0
    k_ppp_games: float = 25.0
    k_sv_shots: float = 1200.0


DEFAULT_SHRINK = PropShrink()


@dataclass(frozen=True)
class SkaterProjection:
    player_id: int
    name: str
    team: str
    position: str  # F | D
    toi: float  # projected seconds tonight
    sog60: float
    g60: float
    a60: float
    pts60: float
    ppp_pg: float
    games: int  # current-season games in the log
    prior_games: int

    @property
    def minutes(self) -> float:
        return self.toi / 60.0

    def mean(self, stat: str) -> float:
        if stat == "ppp":
            return self.ppp_pg
        rate = {"sog": self.sog60, "g": self.g60, "a": self.a60, "pts": self.pts60}[stat]
        return rate * self.toi / PER_60


@dataclass(frozen=True)
class GoalieProjection:
    player_id: int
    name: str
    team: str
    sv_pct: float
    shots: int
    games: int


def _before(log: Sequence[SkaterGame], slate: Date) -> list[SkaterGame]:
    return [g for g in log if g.game_date < slate and g.toi > 0]


def _rate(events: float, seconds: float, prior60: float, k_min: float) -> float:
    minutes = seconds / 60.0
    return (events + k_min * prior60 / 60.0) / (minutes + k_min) * 60.0


def league_means(logs: Iterable[tuple[str, Sequence[SkaterGame]]]) -> dict[str, dict[str, float]]:
    """Pooled per-60 rates by position from ``(position, log)`` pairs; constants where thin."""
    out: dict[str, dict[str, float]] = {"F": dict(LEAGUE_F), "D": dict(LEAGUE_D)}
    acc: dict[str, dict[str, float]] = {
        p: {"toi": 0.0, "sog": 0.0, "g": 0.0, "a": 0.0, "pts": 0.0, "ppp": 0.0, "gp": 0.0}
        for p in ("F", "D")
    }
    for pos, log in logs:
        a = acc["D" if pos == "D" else "F"]
        for g in log:
            a["toi"] += g.toi
            a["sog"] += g.sog
            a["g"] += g.goals
            a["a"] += g.assists
            a["pts"] += g.points
            a["ppp"] += g.ppp
            a["gp"] += 1
    for pos, a in acc.items():
        if a["toi"] < 200 * 3600:  # ~200 player-games before the pool replaces the constants
            continue
        out[pos] = {
            "sog60": a["sog"] / a["toi"] * PER_60,
            "g60": a["g"] / a["toi"] * PER_60,
            "a60": a["a"] / a["toi"] * PER_60,
            "pts60": a["pts"] / a["toi"] * PER_60,
            "ppp_pg": a["ppp"] / a["gp"],
        }
    return out


def project_skater(
    *,
    player_id: int,
    name: str,
    team: str,
    position: str,
    current: Sequence[SkaterGame],
    previous: Sequence[SkaterGame],
    slate: Date,
    league: dict[str, float],
    shrink: PropShrink = DEFAULT_SHRINK,
) -> SkaterProjection:
    pos = "D" if position == "D" else "F"
    cur = _before(current, slate)
    prev = [g for g in previous if g.toi > 0]
    n_cur, n_prev = len(cur), len(prev)

    # TOI: season mean shrunk to last season over k games, blended with the last 10.
    prev_toi = sum(g.toi for g in prev) / n_prev if n_prev else 15.5 * 60
    season_toi = (sum(g.toi for g in cur) + shrink.k_toi_games * prev_toi) / (
        n_cur + shrink.k_toi_games
    )
    recent = cur[-shrink.recent_games :]
    recent_toi = sum(g.toi for g in recent) / len(recent) if recent else season_toi
    w = shrink.w_recent * min(1.0, len(recent) / shrink.recent_games)
    toi = w * recent_toi + (1 - w) * season_toi

    def two_stage(stat: str, key: str, k_min: float) -> float:
        prev_ev = sum(getattr(g, stat) for g in prev)
        prev_sec = sum(g.toi for g in prev)
        prior60 = _rate(prev_ev, prev_sec, league[key], k_min)
        return _rate(sum(getattr(g, stat) for g in cur), sum(g.toi for g in cur), prior60, k_min)

    prev_ppp = (sum(g.ppp for g in prev) + shrink.k_ppp_games * league["ppp_pg"]) / (
        n_prev + shrink.k_ppp_games
    )
    ppp = (sum(g.ppp for g in cur) + shrink.k_ppp_games * prev_ppp) / (n_cur + shrink.k_ppp_games)
    return SkaterProjection(
        player_id=player_id,
        name=name,
        team=team,
        position=pos,
        toi=toi,
        sog60=two_stage("sog", "sog60", shrink.k_sog_min),
        g60=two_stage("goals", "g60", shrink.k_g_min),
        a60=two_stage("assists", "a60", shrink.k_a_min),
        pts60=two_stage("points", "pts60", shrink.k_pts_min),
        ppp_pg=ppp,
        games=n_cur,
        prior_games=n_prev,
    )


def project_goalie(
    *,
    player_id: int,
    name: str,
    team: str,
    current: Sequence[GoalieGame],
    previous: Sequence[GoalieGame],
    slate: Date,
    shrink: PropShrink = DEFAULT_SHRINK,
    league_sv: float = LEAGUE_SV,
) -> GoalieProjection:
    cur = [g for g in current if g.game_date < slate and g.shots_against > 0]
    prev = [g for g in previous if g.shots_against > 0]
    k = shrink.k_sv_shots
    psa = sum(g.shots_against for g in prev)
    psv = sum(g.saves for g in prev)
    prior = (psv + k * league_sv) / (psa + k)
    sa = sum(g.shots_against for g in cur)
    sv = sum(g.saves for g in cur)
    return GoalieProjection(player_id, name, team, (sv + k * prior) / (sa + k), sa, len(cur))


__all__ = [
    "LEAGUE_D",
    "LEAGUE_F",
    "LEAGUE_SV",
    "DEFAULT_SHRINK",
    "GoalieProjection",
    "PropShrink",
    "SkaterProjection",
    "league_means",
    "project_goalie",
    "project_skater",
]
