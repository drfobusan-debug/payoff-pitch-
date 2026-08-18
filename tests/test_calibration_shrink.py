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
    fresh = json.loads(path.read_text())
    stale = {**fresh, "markets": {"batter_tb": {**fresh["markets"]["batter_tb"], "basis": "older"}}}
    unstamped = {
        "basis": "older",
        "markets": {mk: {"x": v["x"], "y": v["y"]} for mk, v in fresh["markets"].items()},
        "default": fresh["default"],
    }
    for payload in (stale, unstamped):
        path.write_text(json.dumps(payload))
        assert Calibrator.from_json(path).apply("batter_tb", 0.54) == 0.54


def test_revalidate_keeps_the_stale_markets_that_still_beat_the_raw_probability(tmp_path):
    """The map is re-stamped market by market, on measured out-of-time evidence.

    ``pitcher_h`` is over-confident in the graded rows (priced .70, won 30%), so
    its old curve still helps and is carried onto the current basis; ``batter_tb``
    is priced right, so its curve now hurts and stays retired.
    """
    from mlb_engine.audit.grade import LOSS, WIN
    from mlb_engine.audit.ledger import LedgerEntry
    from mlb_engine.calibration import FEATURE_BASIS, IsotonicMap
    from mlb_engine.cli import _revalidate_map

    path = tmp_path / "calibration_live.json"
    Calibrator(
        maps={
            "pitcher_h": IsotonicMap([0.3, 0.7], [0.3, 0.35]),
            "batter_tb": IsotonicMap([0.3, 0.7], [0.1, 0.2]),
        },
        default=IsotonicMap([], []),
    ).to_json(path, bases={"pitcher_h": "older", "batter_tb": "older"})

    def rows(market, prob, wins, n, *, day="2026-08-10"):
        out = []
        for i in range(n):
            out.append(
                LedgerEntry(
                    date=day,
                    matchup="AAA @ BBB",
                    category="prop",
                    market=market,
                    selection="Over 5.5",
                    line=5.5,
                    book="dk",
                    odds=-110,
                    tier="Buy",
                    model_prob=prob,
                    raw_prob=prob,
                    ev=0.0,
                    result=WIN if i < wins else LOSS,
                    pnl=0.0,
                )
            )
        return out

    graded = rows("pitcher_h", 0.7, 90, 300) + rows("batter_tb", 0.7, 210, 300)
    assert _revalidate_map(path, graded, "2026-08-01", 200) == 0

    written = json.loads(path.read_text())["markets"]
    assert set(written) == {"pitcher_h"}
    assert written["pitcher_h"]["basis"] == FEATURE_BASIS
    assert Calibrator.from_json(path).apply("pitcher_h", 0.7) < 0.4

    # Nothing to keep -> the file is left alone and the caller sees a failure.
    assert _revalidate_map(path, rows("pitcher_h", 0.35, 105, 300), "2026-08-01", 200) == 1
    assert set(json.loads(path.read_text())["markets"]) == {"pitcher_h"}


def test_a_basis_bump_retires_only_the_markets_it_moved(tmp_path):
    """Retirement is per market: the bump that moves one path leaves the others priced.

    Measured out of time on the graded rows priced after the 2026-08 bumps, the
    retired map still beat the uncalibrated probability on fourteen of nineteen
    markets, so retiring the file wholesale threw away working corrections to
    kill the few that had genuinely gone stale.
    """
    from mlb_engine.calibration import FEATURE_BASIS

    rows = [("batter_tb", 0.54, 0) for _ in range(900)]
    rows += [("batter_tb", 0.54, 1) for _ in range(100)]
    rows += [("pitcher_h", 0.54, 0) for _ in range(900)]
    rows += [("pitcher_h", 0.54, 1) for _ in range(100)]
    path = tmp_path / "calibration_live.json"
    Calibrator.fit(rows).to_json(path, bases={"batter_tb": "total-bases-weighting-2026.07"})

    cal = Calibrator.from_json(path)
    assert cal.apply("batter_tb", 0.54) == 0.54  # retired: its own path changed
    assert cal.apply("pitcher_h", 0.54) < 0.5  # still priced by its own map
    assert json.loads(path.read_text())["markets"]["pitcher_h"]["basis"] == FEATURE_BASIS
