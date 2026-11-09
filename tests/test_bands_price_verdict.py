"""Shared band read: a band that beats the market's fair but not the price is no rule candidate."""

from engine_common.bands import BandRead


def _winning() -> list[float]:
    return [1.0, -1.0, 1.0] * 60  # 180 bets, +60u, clears 2 SE with both halves positive


def test_without_closes_the_verdict_is_unchanged():
    b = BandRead("x", _winning())
    assert b.beats_price is None
    assert b.price_read == "no close read"
    assert b.verdict == "wins: clears the bar"


def test_negative_close_implied_roi_reads_beats_the_fair_not_the_price():
    b = BandRead("x", _winning(), close_rois=[-0.03 + 0.001 * (i % 3) for i in range(180)])
    assert b.beats_price is False
    assert b.close_roi_lo < 0
    assert b.verdict == "beats the fair, not the price"
    assert "close-EV -2.9%" in b.price_read


def test_negative_realised_roi_also_fails_the_price_bar():
    pnls = [-0.95, 1.1, -0.95] * 60  # loses money at the price even if the close-EV is positive
    assert sum(pnls) < 0
    b = BandRead(
        "x", pnls, expected=[-0.6] * 180, close_rois=[0.02 + 0.001 * (i % 3) for i in range(180)]
    )
    assert b.excess > 2 * b.se and min(b.halves) > 0
    assert b.verdict == "beats the fair, not the price"


def test_positive_price_read_keeps_the_win():
    b = BandRead("x", _winning(), close_rois=[0.02 + 0.001 * (i % 3) for i in range(180)])
    assert b.beats_price is True
    assert b.verdict == "wins: clears the bar"
