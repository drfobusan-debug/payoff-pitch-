"""Stolen bases: a rate per time on first, a price at the 0.5 line, and a grade.

The market is new, and the three ways a new market has silently broken here
before are all covered: the selection string must give the player back (a market
missing from the suffix list joins to nothing), the box score must be read for
the stat (a market with no result grades every over as a loss), and the price
must be quoted without being bought until the ledger has an opinion.
"""

from __future__ import annotations

import numpy as np

from mlb_engine.audit.grade import LOSS, WIN, grade
from mlb_engine.data.oddsapi import (
    _BATTER_MARKETS,
    DEFAULT_PROP_MARKETS,
    PRICE_ONLY_MARKETS,
)
from mlb_engine.data.results import GameResult, PlayerLine
from mlb_engine.features.steals import (
    LEAGUE_SB_PER_OPP,
    MAX_RATE,
    SHRINK_K,
    SPRINT_DEFAULT,
    sprint_prior,
    steal_rate,
)
from mlb_engine.market import keys
from mlb_engine.models.props import p_steal_over_half
from mlb_engine.recommendations import Recommendation


def test_the_speed_curve_reads_the_typical_runner_not_the_league_aggregate() -> None:
    """At league-average speed the prior is 0.043, not the league's 0.074.

    That gap is the shape of stealing rather than a level error: the league rate
    is opportunity-weighted and carried by the few who run, while a fit of log
    rate lands near the man who does not. Scaling it up to 0.074 was measured and
    is worse held out (0.28917 v 0.28884) and over-predicts (0.1063 v 0.0974), so
    the test pins the fitted level rather than the intuitive one.
    """
    assert 0.035 < sprint_prior(SPRINT_DEFAULT) < 0.05
    assert sprint_prior(SPRINT_DEFAULT) < LEAGUE_SB_PER_OPP
    # No Statcast time on a hitter is not a claim that he is of average speed.
    assert sprint_prior(None) == LEAGUE_SB_PER_OPP
    assert sprint_prior(float("nan")) == LEAGUE_SB_PER_OPP


def test_speed_orders_the_runners_and_the_clamp_holds_the_tails() -> None:
    fast, average, slow = sprint_prior(30.5), sprint_prior(SPRINT_DEFAULT), sprint_prior(24.5)
    assert fast > average > slow
    # 30.5 ft/s is roughly the fastest man in baseball: the prior may be several
    # times the typical runner's, but it may not price a coin flip.
    assert average * 3 < fast <= MAX_RATE


def test_a_runner_with_no_record_is_priced_off_his_legs() -> None:
    assert steal_rate(0.0, 0.0, 30.0) == sprint_prior(30.0)
    assert steal_rate(0.0, 0.0, None) == LEAGUE_SB_PER_OPP


def test_an_empty_record_is_not_a_runner_who_never_steals() -> None:
    """The unshrunk rate scored *worse* than the league (0.34622 v 0.31531).

    A fast man 0-for-12 is the case that made it worse, so the shipped rate must
    stay near his prior there, and the shrinkage must be worth roughly its
    measured weight rather than a token.
    """
    prior = sprint_prior(29.5)
    thin = steal_rate(0.0, 12.0, 29.5)
    assert thin < prior
    assert thin > prior * 0.6
    # The mix is the measured one: 30 times-on-first of prior.
    assert abs(thin - (SHRINK_K * prior) / (12.0 + SHRINK_K)) < 1e-12


def test_a_full_season_of_stealing_outweighs_the_prior() -> None:
    """A burner with a season behind him is priced on the season, not the fit."""
    prior = sprint_prior(29.0)
    measured = 40.0 / 160.0
    rate = steal_rate(40.0, 160.0, 29.0)
    assert prior < rate < measured
    assert abs(rate - measured) < abs(rate - prior)


def test_a_slow_runner_with_a_long_clean_record_is_still_allowed_to_be_slow() -> None:
    assert steal_rate(0.0, 300.0, 25.0) < 0.02


def test_the_price_is_the_chance_of_a_steal_over_the_simulated_nights() -> None:
    """P(SB>=1) averaged over the sim's times on first, not over their mean.

    Averaging the probability rather than plugging in the mean keeps the
    convexity: the nights he reaches twice are the nights he steals.
    """
    reaches = np.array([0.0, 1.0, 2.0, 3.0])
    rate = 0.2
    want = float(np.mean([1 - (1 - rate) ** r for r in reaches]))
    assert abs(p_steal_over_half(reaches, rate) - want) < 1e-12
    # A night never on first is a night he cannot steal.
    assert p_steal_over_half(np.zeros(5), 0.4) == 0.0
    # Monotone in both arguments.
    assert p_steal_over_half(reaches, 0.3) > p_steal_over_half(reaches, 0.1)
    assert p_steal_over_half(reaches + 1, rate) > p_steal_over_half(reaches, rate)


def test_the_price_never_exceeds_the_chance_of_reaching_first() -> None:
    reaches = np.array([0.0, 1.0, 1.0, 2.0])
    p_reach = float((reaches > 0).mean())
    assert p_steal_over_half(reaches, MAX_RATE) < p_reach


def test_the_selection_gives_the_player_back() -> None:
    """The trap #180 was written about: a stat token absent from the suffix list
    leaves the name as "Chandler Simpson SB", and every outside board misses."""
    sel = keys.batter_prop("Chandler Simpson", "SB", 0.5, "over")
    assert sel == "Chandler Simpson SB o0.5"
    assert keys.player_from_selection(sel) == "Chandler Simpson"


def test_the_market_is_fetched_and_quoted_but_never_bought() -> None:
    assert _BATTER_MARKETS["batter_stolen_bases"] == ("batter_sb", "SB")
    assert "batter_stolen_bases" in DEFAULT_PROP_MARKETS
    assert "batter_sb" in PRICE_ONLY_MARKETS


def _result(steals_taken: int) -> GameResult:
    line = PlayerLine()
    line.batting = {"PA": 4, "H": 1, "1B": 1, "2B": 0, "3B": 0, "HR": 0,
                    "RBI": 0, "R": 1, "BB": 1, "K": 1, "SB": steals_taken}
    return GameResult(
        game_pk=1, final=True, home_runs=3, away_runs=2,
        f5_home=1, f5_away=1, players={99: line},
    )


def _rec() -> Recommendation:
    return Recommendation(
        game_date="2026-08-16", game_pk=1, matchup="TB @ DET", category="batter",
        market="batter_sb", selection="Chandler Simpson SB o0.5", model_prob=0.4,
        raw_prob=0.4, line=0.5, player_id=99, stat="SB", side="over",
    )


def test_a_steal_settles_the_over_and_a_quiet_night_settles_the_under() -> None:
    """Without ``SB`` in the box-score line this returns LOSS on both, which is
    the silent losing record every ungraded market here has started with."""
    assert grade(_rec(), _result(1)) == WIN
    assert grade(_rec(), _result(0)) == LOSS
