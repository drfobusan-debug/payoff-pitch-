"""Confidence shrink and the ledger-refit calibration command."""

from __future__ import annotations

import json

import pytest

from mlb_engine.calibration import Calibrator, ConfidenceShrink

SHRINK = ConfidenceShrink()


def test_shrink_pulls_the_confident_tail_in():
    # The .70+ bucket predicted 75.7% and won 59.3%; the shrink closes part of
    # that without pretending to know the exact residual.
    assert SHRINK.apply(0.757) == pytest.approx(0.686, abs=1e-3)
    assert SHRINK.apply(0.90) < 0.80


def test_shrink_leaves_the_calibrated_middle_and_the_low_tail_alone():
    # Below the pivot the engine is calibrated to ~2 points, and a .10 home-run
    # over is a rare event rather than an over-confident favorite's complement.
    for p in (0.02, 0.10, 0.35, 0.50, 0.55, 0.599):
        assert SHRINK.apply(p) == pytest.approx(p)


def test_shrink_is_monotone_and_continuous():
    probs = [i / 1000 for i in range(1, 1000)]
    out = [SHRINK.apply(p) for p in probs]
    assert all(b >= a for a, b in zip(out, out[1:], strict=False))
    assert max(abs(b - a) for a, b in zip(out, out[1:], strict=False)) < 0.01


def test_two_way_markets_shrink_sub_additively():
    """One-sided means the two sides can sum below 1 -- never above it."""
    for p in (0.62, 0.75, 0.93):
        assert SHRINK.apply(p) + SHRINK.apply(1 - p) < 1.0


def test_shrink_never_crosses_the_half_boundary():
    """It may never flip which side the model prefers."""
    assert SHRINK.apply(0.501) > 0.5
    assert SHRINK.apply(0.999) > 0.5


def test_refit_map_learns_a_market_the_packaged_fit_never_saw():
    # batter_tb is absent from calibration_2024.json, so it priced off the
    # pooled curve; a refit from graded rows gives it its own map.
    rows = [("batter_tb", 0.54, 0) for _ in range(900)]
    rows += [("batter_tb", 0.54, 1) for _ in range(100)]
    cal = Calibrator.fit(rows)
    assert "batter_tb" in cal.maps
    assert cal.apply("batter_tb", 0.54) < 0.5  # no longer a favored pick


def test_odds_api_errors_never_leak_the_key():
    from mlb_engine.data.oddsapi import _redact

    msg = (
        "401 Client Error: Unauthorized for url: "
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?apiKey=SEKRIT&regions=us"
    )
    out = _redact(msg, "SEKRIT")
    assert "SEKRIT" not in out
    assert "apiKey" not in out
    assert "401 Client Error" in out


def test_a_map_fit_on_other_features_is_ignored_rather_than_applied(tmp_path):
    """A map learns what *this* engine's 0.62 means; a foreign one would undo a fix."""
    from mlb_engine.calibration import FEATURE_BASIS

    rows = [("batter_tb", 0.54, 0) for _ in range(900)]
    rows += [("batter_tb", 0.54, 1) for _ in range(100)]
    path = tmp_path / "calibration_live.json"
    Calibrator.fit(rows).to_json(path)

    # Round-trips while the basis matches.
    assert json.loads(path.read_text())["basis"] == FEATURE_BASIS
    assert Calibrator.from_json(path).apply("batter_tb", 0.54) < 0.5

    # Stamped with anything else -- or unstamped, like every map fit before the
    # stamp existed -- it falls back to the identity map.
    for payload in ({**json.loads(path.read_text()), "basis": "older-features"},
                    {k: v for k, v in json.loads(path.read_text()).items() if k != "basis"}):
        path.write_text(json.dumps(payload))
        assert Calibrator.from_json(path).apply("batter_tb", 0.54) == 0.54
