"""Unit tests for the pitcher_outs false-negative bias correction."""

from __future__ import annotations

from mlb_engine.config import Config
from mlb_engine.pipeline import apply_outs_bias


def test_bias_lifts_mid_band() -> None:
    # A passed-but-profitable outs over in the meaty band is lifted.
    assert apply_outs_bias(0.52, 0.04, 0.62) == 0.56


def test_bias_skips_high_confidence_tail() -> None:
    # Above the cap the probability is left untouched (no tail inflation).
    assert apply_outs_bias(0.80, 0.04, 0.62) == 0.80


def test_bias_at_cap_is_applied() -> None:
    assert apply_outs_bias(0.62, 0.04, 0.62) == 0.66


def test_zero_bias_is_noop() -> None:
    assert apply_outs_bias(0.50, 0.0, 0.62) == 0.50


def test_bias_clamps_below_one() -> None:
    assert apply_outs_bias(0.61, 0.5, 0.62) == 1 - 1e-6


def test_config_defaults() -> None:
    cfg = Config()
    assert cfg.pitcher_outs_prob_bias == 0.04
    assert cfg.pitcher_outs_bias_max_prob == 0.62


def test_config_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_PITCHER_OUTS_PROB_BIAS", "0")
    assert Config().pitcher_outs_prob_bias == 0.0
