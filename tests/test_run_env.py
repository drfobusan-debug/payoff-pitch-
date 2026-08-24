"""The two run-environment corrections: the batter tilt, and the league scale."""

from __future__ import annotations

from datetime import date as Date
from types import SimpleNamespace

from mlb_engine.config import Config, EVThresholds
from mlb_engine.market.ev import MarketQuote
from mlb_engine.models.run_env import (
    BASELINE_TOTAL,
    LEAGUE_TOTAL_BASELINE,
    MAX_ELEVATION,
    MIN_OUT_SHARE,
    NON_OUT,
    RUNS_PER_SCALE,
    SCALE_CLAMP,
    TOTALS,
    RunEnvTilt,
    apply_shift,
    logit_shift,
    scale_for_total,
    scale_rates,
)
from mlb_engine.pipeline import Pipeline

TILT = RunEnvTilt(over_tilt=0.08, env_slope=0.03)
MATCHUP = "MIA @ ATL"


def _never(before: Date, days: int) -> float:
    raise AssertionError("the league must not be read when the correction is off")


def test_elevation_is_runs_above_the_league_and_is_clamped() -> None:
    assert TILT.elevation(LEAGUE_TOTAL_BASELINE) == 0.0
    assert TILT.elevation(LEAGUE_TOTAL_BASELINE + 1.5) == 1.5
    assert TILT.elevation(LEAGUE_TOTAL_BASELINE - 1.0) == -1.0
    assert TILT.elevation(LEAGUE_TOTAL_BASELINE + 99) == MAX_ELEVATION
    assert TILT.elevation(LEAGUE_TOTAL_BASELINE - 99) == -MAX_ELEVATION
    # An unpriced or missing simulator total leaves the layer neutral.
    assert TILT.elevation(None) is None
    assert TILT.apply(0.62, None) == 0.62


def test_a_hot_game_marks_its_overs_down_and_a_cold_one_less() -> None:
    hot = TILT.apply(0.55, TILT.elevation(10.5))
    cold = TILT.apply(0.55, TILT.elevation(8.0))
    assert hot < cold < 0.55
    # The under is the complement, so the same call raises it by the same amount.
    assert 1 - hot > 1 - cold > 0.45


def test_the_constant_tilt_applies_at_the_league_run_level() -> None:
    assert TILT.apply(0.60, 0.0) < 0.60


def test_disabled_when_both_terms_are_zero() -> None:
    off = RunEnvTilt(over_tilt=0.0, env_slope=0.0)
    assert not off.enabled
    assert off.apply(0.62, 2.0) == 0.62


def test_probabilities_stay_in_range() -> None:
    for p in (1e-9, 0.001, 0.5, 0.999, 1 - 1e-9):
        for e in (-MAX_ELEVATION, 0.0, MAX_ELEVATION):
            out = TILT.apply(p, e)
            assert 0.0 < out < 1.0


def test_config_ships_the_fitted_values() -> None:
    cfg = Config()
    assert cfg.prop_over_tilt == 0.08
    assert cfg.prop_env_slope == 0.03
    assert cfg.run_env_tilt == RunEnvTilt(0.08, 0.03)


def _priced(
    market: str, side: str, env_elev: float | None, scale: float = 1.0, line: float = 0.5
):
    """One prop priced through the real chain, at a given run environment."""
    p = Pipeline.__new__(Pipeline)
    p.cfg = Config(ev=EVThresholds(min_prob=0.0, max_ev=1.0))
    p._calibrator = SimpleNamespace(apply=lambda market, prob: prob)
    p._shrink = None
    p._splits = {}
    p._quote_aliases = {}
    p._run_env_scale = scale
    game = SimpleNamespace(game_date="2026-08-01", game_pk=1)
    sel = f"Some Hitter H {'o' if side == 'over' else 'u'}{line}"
    quotes = {
        (MATCHUP, market, sel): [
            MarketQuote(book="dk", american=-110.0, opposite_american=-110.0)
        ]
    }
    return p._mk(
        game, MATCHUP, "batter", market, sel, 0.55, line=line, player_id=1,
        stat="H", side=side, quotes=quotes, env_elev=env_elev,
    )


def test_the_pipeline_marks_a_hot_game_down_on_the_batter_over_only() -> None:
    hot = _priced("batter_h", "over", RunEnvTilt.elevation(10.5))
    league = _priced("batter_h", "over", 0.0)
    unknown = _priced("batter_h", "over", None)
    assert hot.model_prob < league.model_prob < unknown.model_prob == 0.55
    # The fade of the same line is raised by exactly what the over lost.
    assert _priced("batter_h", "under", RunEnvTilt.elevation(10.5)).model_prob == 1 - (
        hot.model_prob
    )
    # A pitcher prop is not touched: its own fit did not support the correction.
    assert _priced("pitcher_k", "over", RunEnvTilt.elevation(10.5)).model_prob == 0.55


# ---- the league-level correction -----------------------------------------
def test_the_scale_solves_for_the_league_total_and_is_clamped() -> None:
    assert scale_for_total(BASELINE_TOTAL) == 1.0
    # A league playing a run below the simulator asks it to score a run less.
    cold = scale_for_total(BASELINE_TOTAL - 1.0)
    assert cold < 1.0
    assert abs(cold - (1.0 - 1.0 / RUNS_PER_SCALE)) < 1e-9
    assert scale_for_total(BASELINE_TOTAL + 1.0) > 1.0
    lo, hi = SCALE_CLAMP
    assert scale_for_total(3.0) == lo
    assert scale_for_total(20.0) == hi


def test_scaling_the_rates_moves_runners_on_and_gives_the_rest_to_the_out() -> None:
    rates = {"1B": 0.15, "2B": 0.05, "3B": 0.004, "HR": 0.035, "BB": 0.085,
             "K": 0.22, "OUT": 0.456}
    out = scale_rates(rates, 0.96)
    assert sum(out.values()) == 1.0
    for key in NON_OUT:
        assert out[key] < rates[key]
    # Strikeouts are a matchup read, not a run environment, so they are untouched
    # and the in-play out absorbs the whole residual.
    assert out["K"] == rates["K"]
    assert out["OUT"] > rates["OUT"]
    assert scale_rates(rates, 1.0) == rates


def test_a_degenerate_plate_appearance_is_handed_back_unchanged() -> None:
    # No room left for the residual out share: correcting this would be inventing
    # a probability rather than moving one.
    crowded = {"1B": 0.5, "HR": 0.45, "OUT": 0.05}
    assert scale_rates(crowded, 1.04) == crowded
    assert 1.0 - (0.5 + 0.45) * 1.04 < MIN_OUT_SHARE
    assert scale_rates({"1B": 0.2, "K": 0.8}, 0.96) == {"1B": 0.2, "K": 0.8}


def test_the_shift_is_measured_per_market_and_zero_where_it_is_not() -> None:
    cold = scale_for_total(8.58)
    assert logit_shift("game_total", 8.5, cold) < 0.0
    assert logit_shift("game_total", 8.5, cold) == TOTALS["game_total"][8.5] * (cold - 1.0)
    # Bigger lines move more per unit of scale, in the order the study measured.
    assert abs(logit_shift("game_total", 10.5, cold)) > abs(
        logit_shift("game_total", 7.5, cold)
    )
    # Markets and lines with no coefficient, and a league at the simulator's own
    # run level, are all left alone.
    for market, line in (("moneyline", None), ("batter_h", 0.5), ("pitcher_k", 5.5),
                         ("game_total", 14.5), ("game_total", None)):
        assert logit_shift(market, line, cold) == 0.0
    assert logit_shift("game_total", 8.5, 1.0) == 0.0


def test_a_cold_league_marks_the_total_over_down_and_stays_a_probability() -> None:
    cold = scale_for_total(8.58)
    assert apply_shift(0.52, "game_total", 8.5, cold) < 0.52
    assert apply_shift(0.52, "game_total", 8.5, scale_for_total(10.0)) > 0.52
    assert apply_shift(0.52, "game_total", 8.5, 1.0) == 0.52
    for p in (1e-9, 0.001, 0.5, 0.999, 1 - 1e-9):
        for scale in SCALE_CLAMP:
            assert 0.0 < apply_shift(p, "game_total", 10.5, scale) < 1.0
            assert 0.0 < apply_shift(p, "f5_total", 4.5, scale) < 1.0


def test_the_two_corrections_are_disjoint_by_market() -> None:
    # The batter tilt was fitted on rows priced without the league shift, so a
    # batter prop must not pick up both.
    assert set(TOTALS) == {"game_total", "f5_total"}
    cold = scale_for_total(8.58)
    for market in ("batter_h", "batter_hr", "batter_tb", "batter_hrr"):
        assert apply_shift(0.55, market, 0.5, cold) == 0.55


def test_config_ships_the_totals_correction_on() -> None:
    cfg = Config()
    assert cfg.run_env_totals
    assert cfg.run_env_target_days == 30


def test_the_pipeline_marks_a_cold_league_down_on_the_totals_only() -> None:
    cold = scale_for_total(8.58)
    for market, line in (("game_total", 8.5), ("f5_total", 4.5)):
        assert _priced(market, "over", None, cold, line).model_prob < 0.55
        # The two sides of the total still sum to one after the shift.
        assert _priced(market, "under", None, cold, line).model_prob == 1 - _priced(
            market, "over", None, cold, line
        ).model_prob
    # A league at the simulator's own run level, and the markets with no measured
    # coefficient, price as before.
    assert _priced("game_total", "over", None, 1.0, 8.5).model_prob == 0.55
    for market in ("moneyline", "run_line", "pitcher_k"):
        assert _priced(market, "over", None, cold, 5.5).model_prob == 0.55
    # And a batter prop takes the tilt, not the shift.
    assert _priced("batter_h", "over", None, cold).model_prob == _priced(
        "batter_h", "over", None, 1.0
    ).model_prob


def test_an_unreadable_league_leaves_the_totals_uncorrected() -> None:
    p = Pipeline.__new__(Pipeline)
    p.cfg = Config()
    p.deps = SimpleNamespace(
        stats=SimpleNamespace(league_runs_per_game=lambda before, days: None)
    )
    assert p._league_scale(Date(2026, 8, 24)) == 1.0
    # Off by configuration, the league is never read at all.
    p.cfg = Config(run_env_totals=False)
    p.deps = SimpleNamespace(stats=SimpleNamespace(league_runs_per_game=_never))
    assert p._league_scale(Date(2026, 8, 24)) == 1.0


def test_the_slate_reads_the_league_over_the_configured_window() -> None:
    seen: dict[str, object] = {}

    def read(before: Date, days: int) -> float:
        seen["before"], seen["days"] = before, days
        return 8.58

    p = Pipeline.__new__(Pipeline)
    p.cfg = Config()
    p.deps = SimpleNamespace(stats=SimpleNamespace(league_runs_per_game=read))
    assert p._league_scale(Date(2026, 8, 24)) == scale_for_total(8.58)
    assert seen == {"before": Date(2026, 8, 24), "days": 30}
