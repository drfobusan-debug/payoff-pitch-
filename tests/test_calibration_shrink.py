"""Confidence shrink and the ledger-refit calibration command."""

from __future__ import annotations

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


def test_the_packaged_2026_map_wins_over_the_2024_one():
    from mlb_engine.pipeline import _CALIBRATION_DIR, load_calibrator

    cal = load_calibrator()
    packaged_2026 = Calibrator.from_json(_CALIBRATION_DIR / "calibration_2026.json")
    for mk in ("pitcher_k", "pitcher_bb", "pitcher_outs", "batter_tb"):
        assert cal.apply(mk, 0.4) == packaged_2026.apply(mk, 0.4)


def test_a_local_refit_still_outranks_every_packaged_map(tmp_path):
    from mlb_engine.pipeline import load_calibrator

    live = tmp_path / "live.json"
    rows = [("pitcher_k", 0.4, 1) for _ in range(600)]
    Calibrator.fit(rows).to_json(live)
    assert load_calibrator(live).apply("pitcher_k", 0.4) > 0.9


def test_the_2026_map_lifts_the_pitcher_k_and_bb_underconfidence():
    """Regression guard for the gap the 92-slate backfill measured.

    Over 7,242 graded strikeout props the 2024 map predicted .336 against a
    realized .366, and .357 vs .396 on 4,828 walk props -- it was talking the
    engine out of K/BB props it should have been buying. The refit has to push
    those probabilities *up* on average (not at every single point: the maps
    cross in a couple of places), and pull the over-confident total-bases map
    *down* from its +2.5-point gap.
    """
    from mlb_engine.pipeline import _CALIBRATION_DIR

    old = Calibrator.from_json(_CALIBRATION_DIR / "calibration_2024.json")
    new = Calibrator.from_json(_CALIBRATION_DIR / "calibration_2026.json")
    grid = [i / 20 for i in range(1, 20)]

    def mean(cal: Calibrator, mk: str) -> float:
        return sum(cal.apply(mk, p) for p in grid) / len(grid)

    for mk in ("pitcher_k", "pitcher_bb", "pitcher_outs"):
        assert mean(new, mk) > mean(old, mk)
    assert mean(new, "batter_tb") < mean(old, "batter_tb")


def test_markets_the_refit_lost_out_of_sample_keep_the_2024_map():
    """batter_h/1b/2b were better on the packaged fit, so they must be untouched."""
    from mlb_engine.pipeline import _CALIBRATION_DIR

    old = Calibrator.from_json(_CALIBRATION_DIR / "calibration_2024.json")
    new = Calibrator.from_json(_CALIBRATION_DIR / "calibration_2026.json")
    for mk in ("batter_h", "batter_1b", "batter_2b", "game_ml", "game_total", "f5_total"):
        for raw in (0.2, 0.45, 0.7):
            assert new.apply(mk, raw) == pytest.approx(old.apply(mk, raw))


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
