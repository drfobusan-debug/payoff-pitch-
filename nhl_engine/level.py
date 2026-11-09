"""Nightly goal level fitted to the market's no-vig Over chance at the main total.

``SimParams.goal_level`` scales every xG-derived goal rate. The fixed 1.035 came
from 2022-25 league scoring; in October 2026 a league-average sim sat about 3
points of Over chance under the no-vig consensus at the main line, the same
across weeks and at 5.5 / 6 / 6.5. One multiplier closes a gap shaped like that,
so the card refits it each night, aimed at the market rather than at the scores
(the market has been the better price; the market-vs-results gap is a separate
question, left alone).

The fit, per night:

* Use earlier nights only: the main-total Over row of each game (the line whose
  consensus is nearest a coin flip), from the last ``WINDOW_NIGHTS`` slates of this
  season. Each row carries the level it was priced at (``LedgerRow.goal_level``;
  rows from before the stamp ran at ``base``).
* Put every row on the previous night's level with the sim's own sensitivity
  ``s = dP(over) / d ln(level)`` (a league-average game simulated just above and
  below the level on common draws), then ``gap = mean(market - sim)``.
* Target ``previous x exp(gap / s)``, held within ``MAX_DRIFT`` of ``base`` and
  moved at most ``MAX_STEP`` per night. The first night of a season starts from
  ``base``; under ``MIN_GAMES`` rows the level is held.

The level changes the sim only; with totals graded-only at full market weight it
does not change a stake. ``NHLE_LEVEL_FIT=0`` prices at the fixed level.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date as Date
from pathlib import Path

from nhl_engine.audit.ledger import LedgerRow, load_rows
from nhl_engine.config import SimParams
from nhl_engine.data.moneypuck import season_of
from nhl_engine.models import markets
from nhl_engine.models.goals import game_rates
from nhl_engine.models.periods import simulate

# ~3 weeks, ~150-200 games: the market-vs-sim gap has a per-game SD of about 4.5
# points of Over chance, so the window's SE is ~0.35 pts against a ~3-pt gap, and
# it still follows the market when league scoring moves within a season.
WINDOW_NIGHTS = 21
MIN_GAMES = 30
MAX_STEP = 0.01  # |d ln level| per night (1%)
MAX_DRIFT = 0.10  # |ln level - ln base| (about +/-10%)
BUMP = 0.03  # +/- ln level used to measure the sensitivity
SENS_DRAWS = 40000
SENS_SEED = 20261022


@dataclass(frozen=True)
class MainTotal:
    night: str
    matchup: str
    line: float
    p_sim: float  # the sim's own Over chance (no push), as priced
    p_market: float  # no-vig consensus Over chance
    level: float  # goal level the row was priced at


@dataclass(frozen=True)
class LevelFit:
    level: float
    base: float
    previous: float
    n: int
    first: str
    last: str
    gap: float  # mean(market - sim) on the previous night's level, prob points
    sensitivity: float
    target: float
    reason: str

    def describe(self) -> str:
        if self.n == 0:
            return f"goal level {self.level:.3f} (fixed; {self.reason})"
        held = f"; {self.reason}" if self.reason else ""
        return (
            f"goal level {self.level:.3f} (was {self.previous:.3f}, fixed {self.base:.3f}): "
            f"sim Over {self.gap * -100:+.1f} pts vs the no-vig market on {self.n} games "
            f"{self.first}..{self.last}; target {self.target:.3f}{held}"
        )


def main_totals(rows: Iterable[LedgerRow], *, base: float) -> list[MainTotal]:
    """One main-total Over row per game and night (consensus nearest a coin flip)."""
    by: dict[tuple[str, str], LedgerRow] = {}
    for r in rows:
        if r.market != "game_total" or r.side != "over" or r.line is None:
            continue
        k = (r.slate_date, r.matchup)
        if k not in by or abs(r.consensus - 0.5) < abs(by[k].consensus - 0.5):
            by[k] = r
    return [
        MainTotal(
            r.slate_date,
            r.matchup,
            float(r.line or 0.0),
            r.model_prob,
            r.consensus,
            r.goal_level if r.goal_level is not None else base,
        )
        for _, r in sorted(by.items())
    ]


def sensitivity(league: dict[str, float], params: SimParams, *, line: float = 6.0) -> float:
    """``dP(over line) / d ln(level)`` for a league-average game, on common draws."""
    probs = []
    for sign in (1.0, -1.0):
        p = replace(params, goal_level=params.goal_level * math.exp(sign * BUMP))
        rates = game_rates(
            league, league, league, home_goalie_gsax60=0.0, away_goalie_gsax60=0.0, params=p
        )
        sim = simulate(rates, p, draws=SENS_DRAWS, seed=SENS_SEED)
        probs.append(
            markets.price(
                sim,
                market="game_total",
                side="over",
                entity="",
                line=line,
                ot_rule="incl_ot_so",
                home="H",
                away="A",
            ).win
        )
    return (probs[0] - probs[1]) / (2 * BUMP)


def window(obs: Sequence[MainTotal], slate: Date) -> list[MainTotal]:
    """This season's rows from the last ``WINDOW_NIGHTS`` slates before ``slate``."""
    season = season_of(slate)
    prior = [
        o
        for o in obs
        if o.night < slate.isoformat() and season_of(Date.fromisoformat(o.night)) == season
    ]
    nights = sorted({o.night for o in prior})[-WINDOW_NIGHTS:]
    keep = set(nights)
    return [o for o in prior if o.night in keep]


def fit(obs: Sequence[MainTotal], *, slate: Date, base: float, sens: float) -> LevelFit:
    rows = window(obs, slate)
    if not rows:
        return LevelFit(
            base, base, base, 0, "", "", 0.0, sens, base, "no earlier night this season"
        )
    last = max(o.night for o in rows)
    previous = next(o.level for o in rows if o.night == last)
    first = min(o.night for o in rows)
    if len(rows) < MIN_GAMES or not sens > 0:
        why = f"held: {len(rows)} games < {MIN_GAMES}" if sens > 0 else "held: no sensitivity"
        return LevelFit(previous, base, previous, len(rows), first, last, 0.0, sens, previous, why)
    ln_prev = math.log(previous)
    adj = [o.p_sim + sens * (ln_prev - math.log(o.level)) for o in rows]
    gap = sum(o.p_market - a for o, a in zip(rows, adj, strict=True)) / len(rows)
    ln_target = ln_prev + gap / sens
    lo, hi = math.log(base) - MAX_DRIFT, math.log(base) + MAX_DRIFT
    ln_t = min(max(ln_target, lo), hi)
    ln_new = min(max(ln_t, ln_prev - MAX_STEP), ln_prev + MAX_STEP)
    reasons = []
    if ln_t != ln_target:
        reasons.append(f"target held to {MAX_DRIFT:.0%} of fixed")
    if ln_new != ln_t:
        reasons.append(f"step capped at {MAX_STEP:.0%}")
    return LevelFit(
        math.exp(ln_new),
        base,
        previous,
        len(rows),
        first,
        last,
        gap,
        sens,
        math.exp(ln_target),
        "; ".join(reasons),
    )


def prior_rows(data_dir: Path, slate: Date) -> list[LedgerRow]:
    """Main-total candidates from this season's predictions ledger before ``slate``."""
    out: list[LedgerRow] = []
    season = season_of(slate)
    for p in sorted((data_dir / "ledger").glob("predictions_*.json")):
        day = p.stem.removeprefix("predictions_")
        try:
            d = Date.fromisoformat(day)
        except ValueError:  # versioned ``--force`` copies are not the ledger
            continue
        if d < slate and season_of(d) == season:
            out.extend(r for r in load_rows(p) if r.market == "game_total" and r.slate_date == day)
    return out


@dataclass(frozen=True)
class NightLevel:
    night: str
    level: float
    games: int
    p_sim: float
    p_market: float

    @property
    def gap(self) -> float:
        return self.p_sim - self.p_market


def by_night(rows: Iterable[LedgerRow], *, base: float) -> list[NightLevel]:
    """Per night: level priced at, and the sim vs the no-vig market at the main total."""
    nights: dict[str, list[MainTotal]] = {}
    for o in main_totals(rows, base=base):
        nights.setdefault(o.night, []).append(o)
    out = []
    for night, obs in sorted(nights.items()):
        n = len(obs)
        out.append(
            NightLevel(
                night,
                obs[0].level,
                n,
                sum(o.p_sim for o in obs) / n,
                sum(o.p_market for o in obs) / n,
            )
        )
    return out


__all__ = [
    "LevelFit",
    "MainTotal",
    "NightLevel",
    "by_night",
    "fit",
    "main_totals",
    "prior_rows",
    "sensitivity",
    "window",
]
