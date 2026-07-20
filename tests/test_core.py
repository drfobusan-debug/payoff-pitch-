"""Unit tests for the model, market, and audit layers (no network)."""

from __future__ import annotations

from datetime import date

import numpy as np

from mlb_engine.audit.grade import LOSS, PUSH, WIN, grade
from mlb_engine.audit.scorecard import build_scorecard
from mlb_engine.config import EVThresholds
from mlb_engine.data.results import GameResult, PlayerLine, _ip_to_outs
from mlb_engine.features.rolling import LEAGUE_RATES, OutcomeRates, rates_from_events
from mlb_engine.market.ev import MarketQuote, ev_per_dollar, evaluate
from mlb_engine.market.odds import (
    american_to_decimal,
    american_to_prob,
    no_vig_two_way,
    prob_to_american,
)
from mlb_engine.market.tiers import Tier, classify
from mlb_engine.models.markov_f5 import (
    f5_from_lineups,
    f5_from_rates,
    team_f5_distribution,
)
from mlb_engine.models.matchup import apply_multipliers, combine
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig
from mlb_engine.models.rbi_rule import evaluate_lineup, rbi_multiplier
from mlb_engine.recommendations import Recommendation


def _league_rates() -> OutcomeRates:
    r = LEAGUE_RATES
    return OutcomeRates(500, r["1B"], r["2B"], r["3B"], r["HR"], r["BB"], r["K"], r["OUT"])


# ---- odds math ----
def test_odds_roundtrip():
    for a in (-200, -110, 100, 150, 250):
        p = american_to_prob(a)
        assert 0 < p < 1
    assert abs(american_to_decimal(100) - 2.0) < 1e-9
    assert abs(american_to_decimal(-200) - 1.5) < 1e-9


def test_prob_to_american_inverse():
    for p in (0.4, 0.55, 0.7):
        a = prob_to_american(p)
        assert abs(american_to_prob(a) - p) < 1e-6


def test_no_vig_sums_to_one():
    a, b = no_vig_two_way(-110, -110)
    assert abs(a + b - 1.0) < 1e-9
    assert abs(a - 0.5) < 1e-9


# ---- matchup ----
def test_combine_normalizes():
    lg = _league_rates()
    out = combine(lg, lg)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    # combining league vs league should return ~league rates
    assert abs(out["HR"] - LEAGUE_RATES["HR"]) < 1e-3


def test_apply_multipliers_renormalizes():
    lg = _league_rates()
    base = combine(lg, lg)
    out = apply_multipliers(base, {"HR": 2.0})
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out["HR"] > base["HR"]


# ---- markov F5 ----
def test_f5_reasonable():
    lg = LEAGUE_RATES
    r = f5_from_rates(dict(lg), dict(lg))
    mean = sum(i * p for i, p in enumerate(r.home_dist))
    assert 1.5 < mean < 3.5  # ~2.4 runs/team through 5
    assert abs(r.p_home_ml + r.p_away_ml + r.p_tie - 1.0) < 1e-6
    assert abs(sum(r.total_dist) - 1.0) < 1e-6


def test_f5_lineup_matches_convolution_without_tto():
    lg = dict(LEAGUE_RATES)
    conv = f5_from_rates(lg, lg)
    dp = team_f5_distribution([lg] * 9, tto_factors=(1.0, 1.0, 1.0, 1.0))
    m_conv = sum(i * p for i, p in enumerate(conv.home_dist))
    m_dp = sum(i * p for i, p in enumerate(dp))
    assert abs(m_conv - m_dp) < 1e-4


def test_f5_tto_raises_scoring():
    lg = dict(LEAGUE_RATES)
    base = sum(i * p for i, p in enumerate(team_f5_distribution([lg] * 9, (1.0, 1.0, 1.0, 1.0))))
    tto = sum(i * p for i, p in enumerate(team_f5_distribution([lg] * 9)))
    assert tto > base
    r = f5_from_lineups([lg] * 9, [lg] * 9)
    assert abs(r.p_home_ml + r.p_away_ml + r.p_tie - 1.0) < 1e-6


# ---- weather (WAM park-config filter) ----
def test_weather_park_config_gates_wind():
    from mlb_engine.data.parks import get_park
    from mlb_engine.filters.weather import WeatherConditions, _effect

    out = WeatherConditions(85, 50, 15, 0, 15)  # 15 mph straight out to CF
    wrigley, _ = _effect(out, get_park(17))  # open bowl, wind-receptive
    oracle, _ = _effect(out, get_park(2395))  # shielded
    assert wrigley > 1.15  # wind reaches the field -> big HR boost
    assert oracle < wrigley  # architecture suppresses the same wind

    blow_in = WeatherConditions(85, 50, 15, 180, -15)
    wrigley_in, _ = _effect(blow_in, get_park(17))
    assert wrigley_in < 1.0  # in-from-CF suppresses power


# ---- monte carlo ----
def test_montecarlo_runs():
    lg = LEAGUE_RATES
    bat = [dict(lg) for _ in range(9)]
    cfg = TeamSimConfig(bat_vs_starter=bat, bat_vs_pen=bat)
    res = MonteCarlo(400, seed=1).simulate(cfg, cfg)
    assert res.home_runs_full.shape == (400,)
    assert 3.0 < res.home_runs_full.mean() < 6.5
    # home never trails-and-bats: full >= f5
    assert (res.home_runs_full >= res.home_runs_f5).all()


# ---- rbi rule ----
def _profile(rates: OutcomeRates):
    from mlb_engine.features.rolling import BatterProfile

    return BatterProfile(
        mlbam_id=1, home=rates, away=rates, vs_rhp=rates, vs_lhp=rates, overall=rates
    )


def test_rbi_rule_flags_high_obp():
    hi = OutcomeRates(200, 0.18, 0.06, 0.005, 0.05, 0.12, 0.18, 0.40)  # obp high
    lo = OutcomeRates(200, 0.10, 0.02, 0.0, 0.01, 0.04, 0.30, 0.53)  # obp low
    flags = evaluate_lineup([_profile(hi)] * 9)
    assert all(f.flagged for f in flags)
    assert rbi_multiplier(flags[0]) > 1.0
    flags2 = evaluate_lineup([_profile(lo)] * 9)
    assert not any(f.flagged for f in flags2)
    assert rbi_multiplier(flags2[0]) == 1.0


def _reg(xslg: float, zone_contact: float):
    from mlb_engine.features.regression import BatterRegression

    return BatterRegression(
        bbe=40, barrel_rate=0.08, hard_hit=0.40, sweet_spot=0.33, bat_speed=72.0,
        max_ev=108.0, whiff=0.24, zone_contact=zone_contact, xba=0.25, xslg=xslg,
        babip=0.29, woba=0.32, xwoba=0.32,
    )


def test_rbi_ppv_npv_tiers():
    hi = OutcomeRates(200, 0.18, 0.06, 0.005, 0.05, 0.12, 0.18, 0.40)
    profs = [_profile(hi)] * 9
    # PPV: elite xSLG with runners on boosts RBI above the volume-only boost.
    elite = evaluate_lineup(profs, regressions=[_reg(0.520, 0.82)] * 9)
    vol_only = evaluate_lineup(profs)
    assert rbi_multiplier(elite[0]) > rbi_multiplier(vol_only[0])
    # NPV: in-zone contact collapse caps RBI despite big opportunity.
    collapse = evaluate_lineup(profs, regressions=[_reg(0.400, 0.60)] * 9)
    assert rbi_multiplier(collapse[0]) < rbi_multiplier(vol_only[0])


# ---- ev + tiers ----
def test_ev_positive_when_underpriced():
    # model 60%, priced at +100 (fair 50%) -> positive EV
    ev = ev_per_dollar(0.60, 100)
    assert ev > 0.15


def test_classify_tiers():
    thr = EVThresholds(strong_buy=0.08, moderate_buy=0.03)
    q = [MarketQuote("draftkings", 120, handle_pct=70, bets_pct=45)]
    res = evaluate(0.60, q)
    tier, reasons = classify(res, thr)
    assert tier in (Tier.STRONG, Tier.MODERATE)
    # negative edge -> pass
    res2 = evaluate(0.30, q)
    tier2, _ = classify(res2, thr)
    assert tier2 == Tier.PASS


# ---- grading ----
def _rec(**kw) -> Recommendation:
    base = dict(
        game_date=date(2024, 7, 19),
        game_pk=1,
        matchup="AAA @ BBB",
        category="game",
        market="game_ml",
        selection="x",
        model_prob=0.5,
    )
    base.update(kw)
    return Recommendation(**base)


def test_grade_ml_and_total():
    res = GameResult(1, True, 5, 3, 3, 1)
    assert grade(_rec(market="game_ml", team_side="home", side="win"), res) == WIN
    assert grade(_rec(market="game_ml", team_side="away", side="win"), res) == LOSS
    over = _rec(category="game", market="game_total", line=7.5, side="over")
    assert grade(over, res) == WIN  # total 8 > 7.5
    under = _rec(category="game", market="game_total", line=8.5, side="under")
    assert grade(under, res) == WIN  # total 8 < 8.5


def test_grade_batter_prop():
    res = GameResult(
        1, True, 5, 3, 3, 1, players={99: PlayerLine(batting={"H": 2, "HR": 1, "R": 1, "RBI": 2})}
    )
    r = _rec(category="batter", market="batter_h", player_id=99, stat="H", line=1.5, side="over")
    assert grade(r, res) == WIN
    r2 = _rec(category="batter", market="batter_hr", player_id=99, stat="HR", line=1.5, side="over")
    assert grade(r2, res) == LOSS


def test_grade_push():
    res = GameResult(1, True, 4, 4, 2, 2)
    r = _rec(market="game_ml", team_side="home", side="win")
    assert grade(r, res) == PUSH


def test_ip_to_outs():
    assert _ip_to_outs("5.2") == 17
    assert _ip_to_outs("6.0") == 18


# ---- scorecard ----
def test_scorecard_metrics():
    graded = [
        (_rec(tier=Tier.STRONG), WIN),
        (_rec(tier=Tier.STRONG), WIN),
        (_rec(tier=Tier.STRONG), LOSS),
        (_rec(tier=Tier.PASS), LOSS),
        (_rec(tier=Tier.PASS), LOSS),
    ]
    rows = build_scorecard(graded, date(2024, 7, 19))
    strong = next(r for r in rows if r.tier == Tier.STRONG.value)
    assert strong.n == 3
    assert strong.wins == 2
    assert abs(strong.ppv - 2 / 3) < 1e-3


def test_rates_from_events_sums_to_one():
    import pandas as pd

    ev = pd.Series(["single", "home_run", "strikeout", "walk", "field_out"] * 20)
    r = rates_from_events(ev)
    total = sum(r.as_dict().values())
    assert abs(total - 1.0) < 1e-9
    assert 0 < r.obp < 1


def test_np_import_available():
    assert np.array([1, 2, 3]).sum() == 6
