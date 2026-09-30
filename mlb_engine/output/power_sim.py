"""Turn the power screen's matchup into a probability, by simulating the game.

The screen ends at a sort: it ranks hitters by scored metrics and prints the
exposure it expects them to get. A sort cannot be bet. What a bettor needs from
"good bat, soft arm, 4.5 plate appearances in a doubles park" is the probability
of a double, and the only honest way from one to the other is to play the game
out -- the outcome distribution of a hitter's night is the convolution of his
per-PA rates over an uncertain number of turns, against two different pitching
staffs, and that has no closed form worth trusting.

So this module runs the engine's own simulator on the screen's own matchup. It
adds no new model: the per-PA rates are ``features.rolling``'s, the log5
combination is ``models.matchup``'s, the park multipliers are the measured
component factors in ``data.parks``, the plate-appearance-by-plate-appearance
game is ``models.montecarlo``'s, and the market probabilities come out of
``models.props``. The screen was already computing every input and throwing the
distribution away.

Four things it does *not* do, deliberately.

It reads no market. The screen's contract is that every figure in it is observed
or modelled and none of it is priced, and a simulated probability is a model
output like any other. Comparing it to a price happens in the board section,
which reads the card's own devigged rows off disk, and per #259 a bet blends the
model toward the devigged price before staking. A raw simulated probability is
not a recommendation.

It does not re-neutralise the hitter for his home park. An earlier prototype
divided each hitter's rate vector by a half-weight park factor before applying
tonight's, on the theory that his own rates already contain his home park. The
stored factors make that double correction wrong: ``singles_factor`` and
``xbh_factor`` are each measured as the park's rate *against the rate the same
hitters posted everywhere else*, and then shrunk to the share of the deviation
that repeats across split halves, so they are already a differential and already
conservative. Dividing by them again subtracts an effect that was never added.

It does not pretend the opposing offense is modelled. Only the screened team's
nine hitters carry real rates; the other side bats league-average, because it
enters our hitter's line only through the score margin (which decides whether he
faces leverage arms) and the length of the game. That is a real approximation and
it is the reason the runs and RBI columns are wider than the hits column.

And it fixes the seed. Two runs of the same screen on the same day must print the
same probability, or the number is not a number.
"""

from __future__ import annotations

import math
from datetime import date as Date

import numpy as np
import pandas as pd

from mlb_engine.data.parks import Park
from mlb_engine.features.rolling import (
    LEAGUE_RATES,
    OutcomeRates,
    build_batter_profile,
    build_bullpen_profile,
    build_pitcher_profile,
)
from mlb_engine.models.matchup import apply_multipliers, combine
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig
from mlb_engine.output.power_screen import Distribution, MatchupSection, SimLine

#: Sims per matchup. 20k puts the Monte Carlo error on a .300 probability at
#: about 0.3 percentage points, which is an order of magnitude inside the model
#: error and cheap enough to run for every section of the note.
N_SIMS = 20_000

#: Fixed, so the note is reproducible. Not a knob worth exposing.
SEED = 20260401

#: Markets the note prints, in the order it prints them. Keys are the simulator's
#: own per-slot batting stats (total bases is derived) and the lines are the ones
#: books actually hang.
MARKET_LINES: dict[str, tuple[float, ...]] = {
    "1B": (0.5,),
    "2B": (0.5,),
    "H": (0.5, 1.5),
    "HR": (0.5,),
    "TB": (1.5, 2.5),
    "R": (0.5,),
    "RBI": (0.5,),
}

LEAGUE_LINEUP_SLOTS = 9


def _distribution(arr: np.ndarray, lines: tuple[float, ...]) -> Distribution:
    """Mean, median, mode and threshold probabilities of a simulated stat.

    The mode is the most common single night, which for almost every batter
    market is 0 or 1 -- worth printing precisely because it is the number a
    reader's intuition gets wrong when shown a mean of 1.3 hits.
    """
    values = arr.astype(float)
    counts = np.bincount(np.rint(values).astype(int).clip(min=0))
    return Distribution(
        mean=float(values.mean()),
        median=float(np.median(values)),
        mode=float(int(counts.argmax())),
        over={line: float((values > line).mean()) for line in lines},
    )


def _park_multipliers(park: Park | None) -> dict[str, float]:
    """Tonight's ballpark, on the hit types it has a measured effect on.

    Singles and extra-base hits only. The home-run factor lives in the runs park
    factor and the weather term, neither of which this module reads: the screen
    runs in the morning, before a forecast is worth having, and a park's runs
    factor is not a home-run factor.
    """
    if park is None:
        return {}
    return {"1B": park.singles_factor, "2B": park.xbh_factor, "3B": park.xbh_factor}


def _league_nine() -> list[dict[str, float]]:
    """A league-average lineup, for the side of the game that is not screened."""
    return [dict(LEAGUE_RATES) for _ in range(LEAGUE_LINEUP_SLOTS)]


def _slot_vectors(
    lineup: list[tuple[int, int]],
    frame: pd.DataFrame,
    as_of: Date,
    *,
    form_days: int,
    starter_hand: str,
    starter: OutcomeRates,
    pen: OutcomeRates,
    pen_leverage: OutcomeRates,
    pen_bridge: OutcomeRates,
    is_home: bool,
    park_mult: dict[str, float],
    pen_mult: dict[str, float],
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, float]],
           list[dict[str, float]]]:
    """Per-slot matchup vectors against the starter and each phase of the pen.

    A slot with no readable hitter bats league-average rather than being dropped:
    the base states in front of the screened hitter have to come from somewhere,
    and a hole in the order is a worse error than an average major leaguer.

    The bullpen phases carry one adjustment the starter does not: the pen's own
    workload and zone tripwires. Which arms a manager has left tonight is a
    function of who threw the last three days, so the pen vectors are the
    aggregate rates moved by the measured workload penalty rather than the
    aggregate alone -- a pen missing its rested leverage arms allows more damage
    than its season line says.
    """
    vs_sp: list[dict[str, float]] = []
    vs_pen: list[dict[str, float]] = []
    vs_close: list[dict[str, float]] = []
    vs_bridge: list[dict[str, float]] = []
    pen_park = {k: park_mult.get(k, 1.0) * v for k, v in pen_mult.items()} | {
        k: v for k, v in park_mult.items() if k not in pen_mult
    }
    by_slot = {slot: mlbam_id for slot, mlbam_id in lineup}
    for slot in range(1, LEAGUE_LINEUP_SLOTS + 1):
        mlbam_id = by_slot.get(slot)
        if mlbam_id is None:
            vs_sp.append(dict(LEAGUE_RATES))
            vs_pen.append(dict(LEAGUE_RATES))
            vs_close.append(dict(LEAGUE_RATES))
            vs_bridge.append(dict(LEAGUE_RATES))
            continue
        profile = build_batter_profile(
            frame,
            mlbam_id,
            as_of,
            home_away_days=form_days,
            vs_rhp_days=form_days,
            vs_lhp_days=form_days,
        )
        bat = profile.for_context(is_home=is_home, opp_hand=starter_hand)
        vs_sp.append(apply_multipliers(combine(bat, starter), park_mult))
        vs_pen.append(apply_multipliers(combine(bat, pen), pen_park))
        vs_close.append(apply_multipliers(combine(bat, pen_leverage), pen_park))
        vs_bridge.append(apply_multipliers(combine(bat, pen_bridge), pen_park))
    return vs_sp, vs_pen, vs_close, vs_bridge


def _sim_config(
    section: MatchupSection,
    vs_sp: list[dict[str, float]],
    vs_pen: list[dict[str, float]],
    vs_close: list[dict[str, float]],
    vs_bridge: list[dict[str, float]],
) -> TeamSimConfig:
    """The screened offense, with the starter's exit taken from the screen itself.

    The screen has already measured this starter's hook two ways -- a
    batters-faced cap from the manager's record and a pitch budget burned at the
    rate the lineup he faces makes him throw -- and prints both. Reusing them
    means the simulated bullpen exposure equals the exposure the note's own table
    claims, instead of the simulator and the table disagreeing about when the
    starter leaves.
    """
    return TeamSimConfig(
        bat_vs_starter=vs_sp,
        bat_vs_pen=vs_pen,
        bat_vs_pen_close=vs_close,
        bat_vs_pen_bridge=vs_bridge,
        starter_bf_cap=section.starter_bf_cap,
        starter_pitch_cap=section.pitch_cap,
        pitch_eff=min(1.35, section.pitches_per_pa / 3.9 / max(section.discipline, 0.5)),
    )


def simulate_section(
    section: MatchupSection,
    *,
    lineup: list[tuple[int, int]],
    frame: pd.DataFrame,
    as_of: Date,
    form_days: int,
    pen_days: int,
    park: Park | None,
    is_home: bool,
    n_sims: int = N_SIMS,
    seed: int = SEED,
) -> dict[int, SimLine]:
    """Simulate one matchup and return a distribution per surviving hitter.

    ``lineup`` is ``(slot, mlbam_id)`` for the whole screened order, not only the
    survivors: a four-hole hitter's RBI depends on who bats in front of him.
    """
    starter = build_pitcher_profile(frame, section.starter.mlbam_id, as_of, form_days).allowed
    pen = build_bullpen_profile(frame, section.starter.team, as_of, pen_days)
    park_mult = _park_multipliers(park)
    vs_sp, vs_pen, vs_close, vs_bridge = _slot_vectors(
        lineup,
        frame,
        as_of,
        form_days=form_days,
        starter_hand=section.starter.throws,
        starter=starter,
        pen=pen.allowed,
        pen_leverage=pen.allowed_leverage,
        pen_bridge=pen.bridge,
        is_home=is_home,
        park_mult=park_mult,
        pen_mult=pen.npv_multipliers(),
    )
    screened = _sim_config(section, vs_sp, vs_pen, vs_close, vs_bridge)
    # The other half of the game: a league-average nine against a league-average
    # staff, present so the score margin and the innings played are not degenerate.
    opponent = TeamSimConfig(bat_vs_starter=_league_nine(), bat_vs_pen=_league_nine())
    home, away = (screened, opponent) if is_home else (opponent, screened)
    res = MonteCarlo(n_sims, seed=seed).simulate(home, away)
    team_key = "home" if is_home else "away"

    out: dict[int, SimLine] = {}
    for view in section.hitters:
        slot = view.line.slot
        if not slot or not 1 <= slot <= LEAGUE_LINEUP_SLOTS:
            continue
        idx = slot - 1
        bat = res.bat[team_key]
        stats: dict[str, Distribution] = {}
        for stat, lines in MARKET_LINES.items():
            arr = _stat_array(bat, stat, idx)
            if arr is None:
                continue
            stats[stat] = _distribution(arr, lines)
        out[view.line.mlbam_id] = SimLine(
            mlbam_id=view.line.mlbam_id,
            slot=slot,
            n_sims=n_sims,
            pa_mean=view.exposure.pa_total if view.exposure else math.nan,
            stats=stats,
        )
    return out


def _stat_array(bat: dict[str, np.ndarray], stat: str, idx: int) -> np.ndarray | None:
    """One slot's per-simulation totals for a market, including the derived ones."""
    if stat == "TB":
        needed = ("1B", "2B", "3B", "HR")
        if any(k not in bat for k in needed):
            return None
        return (
            bat["1B"][:, idx] + 2 * bat["2B"][:, idx]
            + 3 * bat["3B"][:, idx] + 4 * bat["HR"][:, idx]
        ).astype(float)
    if stat not in bat:
        return None
    return bat[stat][:, idx].astype(float)


def fair_price(prob: float) -> str:
    """American odds a probability is worth, for the note's table.

    A fair price, not a bet: it says what the model thinks, and comparing it to
    the board is the board section's job.
    """
    if prob <= 0 or prob >= 1 or math.isnan(prob):
        return "&mdash;"
    if prob >= 0.5:
        return f"{-round(100 * prob / (1 - prob))}"
    return f"+{round(100 * (1 - prob) / prob)}"


__all__ = [
    "MARKET_LINES",
    "N_SIMS",
    "SEED",
    "fair_price",
    "simulate_section",
]
