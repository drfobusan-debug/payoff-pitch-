"""The shared paper -> Lean (probation) bar."""

import pytest

from engine_common import promotion


def _series(n: int, roi: float) -> list[float]:
    # alternating so both halves carry the same mean
    return [roi + 0.9, roi - 0.9] * (n // 2)


def test_close_roi_is_the_price_taken_under_the_closing_probability():
    assert promotion.close_roi(0.5, +100) == pytest.approx(0.0)
    assert promotion.close_roi(0.55, -110) == pytest.approx(0.55 * (1 + 100 / 110) - 1)
    # Whelan: a +1.5 pt move at -110 is still a losing price at the close
    assert promotion.close_roi(0.524 + 0.015, -110) > 0 > promotion.close_roi(0.524 - 0.015, -110)
    assert promotion.close_roi(0.20, +300) < 0 < promotion.close_roi(0.26, +300)


def test_below_the_volume_bar_is_paper_whatever_the_numbers_say():
    p = promotion.read("game_total", _series(98, 0.2), [0.01] * 50 + [0.02] * 48, 0.20, 0.25)
    assert p.status == promotion.PAPER and p.stake == 0 and "2 more buys" in p.why_not


def test_clears_only_with_roi_close_roi_and_brier_together():
    pnls = _series(100, 0.05)
    rois = [0.03, 0.01] * 50  # mean +2% at the close, tiny spread -> lower bound > 0
    good = promotion.read("game_total", pnls, rois, 0.240, 0.245)
    assert good.cleared and good.status == promotion.LEAN and good.stake == 0.5
    assert good.close_n == 100 and good.close_roi == pytest.approx(0.02)
    assert good.halves == (good.roi, good.roi) or abs(good.halves[0] - good.halves[1]) < 1e-9
    assert not promotion.read("game_total", pnls, rois, 0.250, 0.245).cleared  # Brier
    assert not promotion.read("game_total", pnls, rois, None, None).cleared  # no model
    bad_half = [0.5] * 50 + [-0.4] * 50
    assert "ROI" in promotion.read("game_total", bad_half, rois, 0.24, 0.245).why_not


def test_a_small_positive_move_that_does_not_clear_the_juice_stays_paper():
    pnls = _series(100, 0.05)
    # every close moved toward us (+0.5 pt, 52.5% -> 53.0%) but -115 needs 53.5%: still -1% at the close
    rois = [promotion.close_roi(0.530, -115)] * 100
    p = promotion.read("game_total", pnls, rois, 0.24, 0.245)
    assert p.close_roi < 0 and not p.cleared and "close ROI" in p.why_not
    noisy = promotion.read("game_total", pnls, [0.05, -0.05] * 50, 0.24, 0.245)
    assert not noisy.cleared and "close ROI" in noisy.why_not
    assert "close ROI" in promotion.read("game_total", pnls, [], 0.24, 0.245).why_not
