"""One recommendation per prop, and run lines that are complements.

Two failures the ledger showed together: the engine bought both sides of the
same run line on 45 games (net -6.35u, which is the vig paid on purpose), and
the reason it thought both were good is that the two underdog run lines carried
each other's probability.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import (
    ONE_BUY_GATE,
    Recommendation,
    enforce_one_buy_per_group,
)


def _rec(
    market: str,
    selection: str,
    *,
    tier: Tier = Tier.STRONG,
    edge: float = 0.05,
    ev: float = 0.10,
    line: float | None = None,
    player_id: int | None = None,
    price: float | None = -110.0,
    game_pk: int = 1,
) -> Recommendation:
    return Recommendation(
        game_date=date(2026, 8, 15),
        game_pk=game_pk,
        matchup="ATH @ AZ",
        category="game",
        market=market,
        selection=selection,
        model_prob=0.55,
        line=line,
        player_id=player_id,
        market_american=price,
        edge=edge,
        ev=ev,
        tier=tier,
    )


def test_both_sides_of_a_run_line_cannot_both_be_bought() -> None:
    recs = [
        _rec("game_rl", "AZ -1.5", edge=0.04, ev=0.33),
        _rec("game_rl", "ATH +1.5", edge=0.09, ev=0.22),
    ]
    enforce_one_buy_per_group(recs)
    buys = [r for r in recs if r.tier is not Tier.PASS]
    assert len(buys) == 1
    # ranked on edge, as market.tiers ranks, not on the fatter EV
    assert buys[0].selection == "ATH +1.5"
    dropped = next(r for r in recs if r.tier is Tier.PASS)
    assert dropped.pass_gate == ONE_BUY_GATE
    assert "ATH +1.5" in dropped.reasons[-1]


def test_one_player_and_market_is_one_bet_across_lines() -> None:
    recs = [
        _rec("batter_h", "Vargas H o0.5", line=0.5, player_id=7, edge=0.08),
        _rec("batter_h", "Vargas H o1.5", line=1.5, player_id=7, edge=0.03,
             tier=Tier.MODERATE),
    ]
    enforce_one_buy_per_group(recs)
    assert [r.selection for r in recs if r.tier is not Tier.PASS] == ["Vargas H o0.5"]


def test_different_players_and_different_markets_are_left_alone() -> None:
    recs = [
        _rec("batter_h", "Vargas H o0.5", player_id=7),
        _rec("batter_h", "Marte H o0.5", player_id=8),
        _rec("batter_hr", "Vargas HR o0.5", player_id=7),
        _rec("game_rl", "AZ -1.5", game_pk=2),
    ]
    enforce_one_buy_per_group(recs)
    assert all(r.tier is not Tier.PASS for r in recs)


def test_a_strong_buy_outranks_a_moderate_with_a_bigger_edge() -> None:
    recs = [
        _rec("game_total", "Over 7.5", line=7.5, tier=Tier.STRONG, edge=0.04),
        _rec("game_total", "Over 8.5", line=8.5, tier=Tier.MODERATE, edge=0.09),
    ]
    enforce_one_buy_per_group(recs)
    assert [r.selection for r in recs if r.tier is not Tier.PASS] == ["Over 7.5"]


def test_unpriced_flags_are_not_deduplicated() -> None:
    """Both teams can honestly be resilient; comeback rows are not bets."""
    recs = [
        _rec("comeback", "AZ comeback", price=None, edge=None, ev=None),
        _rec("comeback", "ATH comeback", price=None, edge=None, ev=None),
    ]
    enforce_one_buy_per_group(recs)
    assert all(r.tier is Tier.STRONG for r in recs)
    assert all(r.pass_gate is None for r in recs)


def test_passes_are_never_promoted_or_relabelled() -> None:
    recs = [
        _rec("game_rl", "AZ -1.5", tier=Tier.PASS),
        _rec("game_rl", "ATH +1.5", tier=Tier.PASS),
    ]
    enforce_one_buy_per_group(recs)
    assert all(r.tier is Tier.PASS for r in recs)
    assert all(r.pass_gate is None for r in recs)


@pytest.mark.parametrize("margins", [
    np.array([3.0, -2.0, 1.0, 0.0, -1.0, 5.0, -4.0, 2.0]),
    np.array([1.0, 1.0, -1.0, -3.0]),
])
def test_the_two_sides_of_a_run_line_are_complements(margins: np.ndarray) -> None:
    """The dog at +1.5 wins exactly when the favourite at -1.5 does not.

    This is the arithmetic the pipeline got wrong: ``away +1.5`` was priced as
    ``P(margin > -1.5)``, which is the *home* dog's condition, so the two
    underdog lines were swapped and each game's complements summed to ~1.29
    instead of 1.
    """
    home_fav = float((margins > 1.5).mean())
    away_dog = float((margins < 1.5).mean())
    away_fav = float((-margins > 1.5).mean())
    home_dog = float((margins > -1.5).mean())

    assert home_fav + away_dog == pytest.approx(1.0)
    assert away_fav + home_dog == pytest.approx(1.0)
    # and the bug's signature: same-team pairs must NOT be complements
    assert home_fav + home_dog != pytest.approx(1.0)
