"""The CLV-gap line: claimed edge beside the share of it the close confirmed."""

from __future__ import annotations

from cfb_engine.audit.clv import clv_gap


def test_clv_gap_averages_the_buys_with_a_close_only() -> None:
    gap = clv_gap(
        [
            (True, 0.60, 0.55, 0.01, -0.02),
            (True, 0.58, 0.52, 0.00, -0.03),
            (True, 0.70, 0.60, None, None),  # no close: left out
            (False, 0.80, 0.50, 0.05, 0.10),  # a pass: left out
        ]
    )
    assert gap is not None
    assert gap.n == 2
    assert abs(gap.claimed - 0.055) < 1e-9
    assert abs(gap.moved - 0.005) < 1e-9
    assert abs(gap.close_ev - -0.025) < 1e-9
    assert abs(gap.confirmed_share - 0.005 / 0.055) < 1e-9


def test_clv_gap_is_none_without_a_priced_buy() -> None:
    assert clv_gap([(True, 0.6, None, 0.01, 0.0), (False, 0.6, 0.5, 0.01, 0.0)]) is None
