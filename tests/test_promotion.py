"""The shared paper -> Lean (probation) bar."""

from engine_common import promotion


def _series(n: int, roi: float) -> list[float]:
    # alternating so both halves carry the same mean
    return [roi + 0.9, roi - 0.9] * (n // 2)


def test_below_the_volume_bar_is_paper_whatever_the_numbers_say():
    p = promotion.read("game_total", _series(98, 0.2), [0.01] * 50 + [0.02] * 48, 0.20, 0.25)
    assert p.status == promotion.PAPER and p.stake == 0 and "2 more buys" in p.why_not


def test_clears_only_with_roi_clv_and_brier_together():
    pnls = _series(100, 0.05)
    clvs = [0.006, 0.004] * 50  # mean 0.5 pts, tiny spread -> lower bound > 0
    good = promotion.read("game_total", pnls, clvs, 0.240, 0.245)
    assert good.cleared and good.status == promotion.LEAN and good.stake == 0.5
    assert good.halves == (good.roi, good.roi) or abs(good.halves[0] - good.halves[1]) < 1e-9
    assert not promotion.read("game_total", pnls, clvs, 0.250, 0.245).cleared  # Brier
    assert not promotion.read("game_total", pnls, [0.02, -0.02] * 50, 0.24, 0.245).cleared  # CLV
    assert not promotion.read("game_total", pnls, clvs, None, None).cleared  # no model
    bad_half = [0.5] * 50 + [-0.4] * 50
    assert "ROI" in promotion.read("game_total", bad_half, clvs, 0.24, 0.245).why_not
