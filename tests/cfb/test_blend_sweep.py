"""The blend sweep rebuilds the card's rest/bye/travel points from its reasons."""

from __future__ import annotations

import pytest

from cfb_engine.config import Config
from scripts.cfb.blend_sweep import Game, score, situational


def test_situational_points_are_rebuilt_from_the_reasons():
    cfg = Config()
    f = cfg.features
    reasons = [
        "EV=+0.035 edge=+0.056",
        "Rest edge Utah (15d vs 8d)",
        "Utah off a bye",
        "Kansas travels 888 mi",
        "Wind 20 mph [observed, not scored]",
    ]
    want = (
        min(7 * f.rest_pts_per_day, f.rest_max_pts)
        + f.bye_bonus_pts
        + min(0.888 * f.travel_pts_per_1000mi, f.travel_max_pts)
    )
    assert situational(reasons, "Utah", cfg) == pytest.approx(want)
    assert situational(["Kansas off a bye"], "Utah", cfg) == pytest.approx(-f.bye_bonus_pts)


def test_weight_one_without_the_terms_prices_the_market_margin():
    cfg = Config()
    g = Game("2026-10-10", "A @ B", 10.0, 3.0, -3.0, 1.0, 0.6, 0.5, 7)
    at_market = score([g], 1.0, False, cfg)
    on_ratings = score([g], 0.0, False, cfg)
    assert at_market.se == pytest.approx(16.0)
    assert on_ratings.se == pytest.approx(9.0)
    assert at_market.n_ats == 1 and at_market.n_ml == 1
