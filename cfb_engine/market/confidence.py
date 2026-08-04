"""Metric-driven marking layer: PPV confidence bumps + NPV veto gates.

The Monte Carlo (or Markov) sim already produces the raw probabilities. This
module never fabricates a probability -- it only decides, using the CFBD
advanced efficiency metrics that carry documented predictive value, whether an
already-priced selection should move a tier (confidence) or be dropped outright
(veto).

Confidence signals are restricted to the metrics that *repeat* season-to-season
(the reliability lesson from the MLB engine): opponent-agnostic EPA/play (net
PPA), success rate, and defensive havoc. Weights come from the empirical PPV of
each metric's "battle winner" (puntandrally, 2016-2025, ~7k games): net PPA
covers ~66.6%, havoc ~64.1%, success rate ~60.0%. Finishing-drives / red-zone
rate are deliberately *excluded* -- they barely autocorrelate, so they are noise
until proven. Totals lean on the combined-PPA scoring environment (~69% Over
above average vs ~26% below).

Caveat: CFBD per-play stats are not opponent-adjusted, so these diffs are noisy
cross-conference; they are used only as a small tie-breaker on top of the
already-opponent-adjusted SP+/ensemble rating that sets the number.

Veto gates encode negative-predictive-value findings: an extreme turnover margin
is a *regression/fade* signal precisely because it does not repeat (the recovery
half is a coin flip), a low/high-scoring environment sinks the wrong side of a
total, and an efficiency blowout makes a points-laying cover structurally
unlikely.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfb_engine.config import MarkingParams
from cfb_engine.data.advanced import AdvancedBook

# ATS cover rates of each metric's battle winner -> (rate - 0.5) weight.
_W_PPA = 0.166
_W_HAVOC = 0.141
_W_SUCCESS = 0.100
# Totals environment weights (Over rate above average - 0.5).
_W_TOT_PPA = 0.193
_W_TOT_EXPL = 0.113


@dataclass(frozen=True)
class MatchupSignal:
    """Home-perspective efficiency edges + totals scoring environment.

    ``None`` fields mark a stat that is unavailable for this matchup, so every
    consumer skips it rather than treating missing data as a neutral zero.
    """

    ppa_edge: float | None = None  # home net PPA/play minus away net PPA/play
    success_edge: float | None = None
    havoc_edge: float | None = None
    finishing_edge: float | None = None
    home_to_margin: float | None = None  # turnover margin per game
    away_to_margin: float | None = None
    # Totals environment, expressed as combined value minus twice the league
    # per-team mean (positive = higher-scoring than average).
    combined_ppa_env: float | None = None
    combined_explosive_env: float | None = None
    pace_env: float | None = None

    @property
    def has_efficiency(self) -> bool:
        return self.ppa_edge is not None


@dataclass(frozen=True)
class Veto:
    dropped: bool
    gate: str | None = None


def build_signal(book: AdvancedBook, home_name: str, away_name: str) -> MatchupSignal:
    """Assemble the marking signal for a matchup from the advanced-stats book."""
    home = book.get(home_name)
    away = book.get(away_name)
    if home is None or away is None:
        return MatchupSignal()
    return MatchupSignal(
        ppa_edge=home.net_ppa - away.net_ppa,
        success_edge=home.net_success - away.net_success,
        # Defensive disruption is a team-quality edge; higher havoc = better.
        havoc_edge=home.havoc - away.havoc,
        finishing_edge=home.off_finishing - away.off_finishing,
        home_to_margin=home.turnover_margin_pg,
        away_to_margin=away.turnover_margin_pg,
        combined_ppa_env=(home.off_ppa + away.off_ppa) - 2 * book.mean_off_ppa,
        combined_explosive_env=_env(
            home.off_explosive, away.off_explosive, book.mean_off_explosive
        ),
        pace_env=_env(home.plays_per_game, away.plays_per_game, book.mean_plays_per_game),
    )


def _env(home_val: float, away_val: float, mean: float) -> float | None:
    """Combined-vs-average environment, or None if either side's stat is unknown."""
    if home_val <= 0.0 or away_val <= 0.0:
        return None
    return (home_val + away_val) - 2 * mean


def _sign_for_side(team_side: str | None, side: str | None) -> int:
    """+1 if a home-perspective edge supports this bet, -1 if it opposes it."""
    if team_side == "home":
        return 1
    if team_side == "away":
        return -1
    if side == "over":
        return 1
    if side == "under":
        return -1
    return 0


def _support_score(
    market: str, team_side: str | None, side: str | None, sig: MatchupSignal, dead: float
) -> float:
    """PPV-weighted net support for the bet (positive favors it)."""
    orient = _sign_for_side(team_side, side)
    if orient == 0:
        return 0.0
    score = 0.0
    if market == "game_total":
        for env, weight in (
            (sig.combined_ppa_env, _W_TOT_PPA),
            (sig.combined_explosive_env, _W_TOT_EXPL),
        ):
            if env is not None:
                score += weight * orient * _step(env, 0.0)
        return score
    for edge, weight, band in (
        (sig.ppa_edge, _W_PPA, dead),
        (sig.success_edge, _W_SUCCESS, 0.02),
        (sig.havoc_edge, _W_HAVOC, 0.01),
    ):
        if edge is not None:
            score += weight * orient * _step(edge, band)
    return score


def _step(value: float, deadband: float) -> int:
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


def confidence_adjustment(
    market: str,
    team_side: str | None,
    side: str | None,
    sig: MatchupSignal,
    params: MarkingParams,
) -> tuple[int, list[str]]:
    """Tier steps (+1 / 0 / -1) from metric agreement, plus audit reasons."""
    score = _support_score(market, team_side, side, sig, params.ppa_deadband)
    if score >= params.bump_up:
        return 1, [f"metrics back it (score {score:+.2f})"]
    if score <= -params.bump_down:
        return -1, [f"metrics fade it (score {score:+.2f})"]
    return 0, []


def market_veto(
    market: str,
    team_side: str | None,
    side: str | None,
    line: float | None,
    sig: MatchupSignal,
    params: MarkingParams,
) -> Veto:
    """NPV gates: drop a structurally hostile bet (never create one)."""
    if params.veto_turnover and market in ("game_ml", "game_ats"):
        to_margin = sig.home_to_margin if team_side == "home" else sig.away_to_margin
        favored = line is not None and line < 0 if market == "game_ats" else True
        if to_margin is not None and to_margin >= params.turnover_extreme and favored:
            return Veto(True, f"turnover-luck regression (+{to_margin:.2f}/gm)")

    if params.veto_ats_blowout and market == "game_ats" and sig.ppa_edge is not None:
        orient = _sign_for_side(team_side, side)
        laying = line is not None and line < 0
        # Bet side is the one badly losing the efficiency battle yet laying points.
        against = -orient * sig.ppa_edge
        if laying and against >= params.ppa_blowout:
            return Veto(True, f"efficiency blowout vs the number ({against:.2f})")

    if params.veto_totals_env and market == "game_total":
        env = sig.combined_ppa_env
        pace = sig.pace_env
        expl = sig.combined_explosive_env
        if side == "over" and env is not None and pace is not None:
            if env < 0 and pace < 0:
                return Veto(True, "low-scoring environment (PPA & pace below avg)")
        if side == "under" and env is not None and expl is not None:
            if env > 0 and expl > 0:
                return Veto(True, "shootout environment (PPA & explosiveness above avg)")

    return Veto(False)
