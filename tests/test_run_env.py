"""The batter-prop over correction, and the elevation it is charged on."""

from __future__ import annotations

from types import SimpleNamespace

from mlb_engine.config import Config, EVThresholds
from mlb_engine.market.ev import MarketQuote
from mlb_engine.models.run_env import LEAGUE_TOTAL_BASELINE, MAX_ELEVATION, RunEnvTilt
from mlb_engine.pipeline import Pipeline

TILT = RunEnvTilt(over_tilt=0.08, env_slope=0.03)
MATCHUP = "MIA @ ATL"


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


def _priced(market: str, side: str, env_elev: float | None):
    """One prop priced through the real chain, at a given run environment."""
    p = Pipeline.__new__(Pipeline)
    p.cfg = Config(ev=EVThresholds(min_prob=0.0, max_ev=1.0))
    p._calibrator = SimpleNamespace(apply=lambda market, prob: prob)
    p._shrink = None
    p._splits = {}
    p._quote_aliases = {}
    game = SimpleNamespace(game_date="2026-08-01", game_pk=1)
    sel = f"Some Hitter H {'o' if side == 'over' else 'u'}0.5"
    quotes = {
        (MATCHUP, market, sel): [
            MarketQuote(book="dk", american=-110.0, opposite_american=-110.0)
        ]
    }
    return p._mk(
        game, MATCHUP, "batter", market, sel, 0.55, line=0.5, player_id=1,
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
