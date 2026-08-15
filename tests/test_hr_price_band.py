"""A home-run buy has to sit inside a payable price band.

Home-run overs are chosen on EV, which is a small probability multiplied by a
large payout -- so the longer the price, the more of the "edge" is the model's
own error. The graded card is unambiguous: 62 buys at +500 or longer produced
three winners, while the +400-500 bucket was the only profitable one. This band
is deliberately blind to EV and probability; those are exactly the inputs that
were wrong.
"""

from __future__ import annotations

from mlb_engine.config import Config
from mlb_engine.features.market_gates import price_band_allows

BAND = (400.0, 700.0)


def test_the_profitable_bucket_is_kept() -> None:
    keep, reason = price_band_allows(450, *BAND)
    assert keep
    assert reason == ""


def test_the_band_is_inclusive_at_both_edges() -> None:
    assert price_band_allows(400, *BAND)[0]
    assert price_band_allows(700, *BAND)[0]


def test_a_longshot_is_passed_however_good_the_price_looks() -> None:
    keep, reason = price_band_allows(900, *BAND)
    assert not keep
    assert "+900" in reason


def test_a_short_price_is_passed_too() -> None:
    """Under the band the payout no longer covers a home run's base rate."""
    assert not price_band_allows(250, *BAND)[0]


def test_an_unpriced_selection_is_left_to_the_other_screens() -> None:
    keep, reason = price_band_allows(None, *BAND)
    assert keep
    assert reason == ""


def test_the_shipped_defaults_are_the_measured_band() -> None:
    cfg = Config()
    assert (cfg.hr_min_buy_odds, cfg.hr_max_buy_odds) == BAND
