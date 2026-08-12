"""The road moneyline underdog is refused whatever the model thinks of it.

Split the graded sides card by venue and role and only one of the four cells is
badly wrong: the away moneyline dog, 28.6% of 77 bets and -33.9%, negative in
both halves of the window and in the full game and first five separately.
Neither "underdogs" nor "road teams" is the losing category on its own -- home
dogs are break-even and road favourites lose no more than anything else -- so
the screen is the intersection, and it is blind to EV because EV is what put
those tickets on the card.
"""

from __future__ import annotations

from mlb_engine.config import Config
from mlb_engine.features.market_gates import price_ceiling_allows

REFUSE_AT = 100.0


def test_a_road_dog_is_refused() -> None:
    keep, reason = price_ceiling_allows(165, REFUSE_AT, "away-dog")
    assert not keep
    assert "+165" in reason


def test_the_ceiling_is_exclusive_so_pick_em_is_already_a_dog() -> None:
    """'+100 or longer' is the rule, so +100 itself is refused."""
    assert not price_ceiling_allows(100, REFUSE_AT)[0]
    assert price_ceiling_allows(99, REFUSE_AT)[0]


def test_a_road_favourite_is_left_alone() -> None:
    """Road favourites lose 7.7%, in line with the rest of the card."""
    assert price_ceiling_allows(-140, REFUSE_AT)[0]


def test_an_unpriced_side_is_left_to_the_other_screens() -> None:
    keep, reason = price_ceiling_allows(None, REFUSE_AT)
    assert keep
    assert reason == ""


def test_the_shipped_default_is_the_measured_cutoff() -> None:
    assert Config().away_ml_refuse_odds == REFUSE_AT
