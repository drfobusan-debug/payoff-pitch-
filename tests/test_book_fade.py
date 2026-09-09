"""When the engine and the book back different sides, bet the book's side.

The engine's own buy becomes a Pass under ``book_fade``; the other side of the
same market, at the price it was quoted, takes the tier and records what it is
the fade of, so the ledger grades both halves of the decision.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import (
    BOOK_FADE_GATE,
    Recommendation,
    enforce_one_buy_per_group,
    fade_disagreements,
    load_json,
    save_json,
)


def _rec(
    market: str,
    selection: str,
    *,
    tier: Tier = Tier.PASS,
    fair: float | None = 0.5,
    line: float | None = None,
    player_id: int | None = None,
    team_side: str | None = None,
    side: str | None = None,
    price: float | None = -110.0,
    game_pk: int = 1,
) -> Recommendation:
    return Recommendation(
        game_date=date(2026, 9, 8),
        game_pk=game_pk,
        matchup="WSH @ SD",
        category="game",
        market=market,
        selection=selection,
        model_prob=0.55,
        line=line,
        player_id=player_id,
        team_side=team_side,
        side=side,
        market_american=price,
        fair_prob=fair,
        ev=0.08,
        edge=0.05,
        tier=tier,
        reasons=["EV=+0.080 edge=+0.050"],
        pass_gate=None if tier is not Tier.PASS else "thin_edge",
    )


def test_prop_buy_on_the_books_dog_moves_to_the_books_side() -> None:
    over = _rec(
        "batter_h",
        "Wood H o1.5",
        tier=Tier.STRONG,
        fair=0.38,
        line=1.5,
        player_id=7,
        team_side="away",
        side="over",
        price=150.0,
    )
    under = _rec(
        "batter_h",
        "Wood H u1.5",
        fair=0.62,
        line=1.5,
        player_id=7,
        team_side="away",
        side="under",
        price=-190.0,
    )
    fade_disagreements([over, under], 0.5)

    assert over.tier is Tier.PASS
    assert over.pass_gate == BOOK_FADE_GATE
    assert "Wood H u1.5" in over.reasons[-1]
    assert under.tier is Tier.STRONG
    assert under.pass_gate is None
    assert under.fade_of == "Wood H o1.5"
    assert under.market_american == -190.0
    assert under.reasons[0].startswith("book fade: engine backed Wood H o1.5")


def test_buy_on_the_books_favourite_is_left_alone() -> None:
    over = _rec(
        "batter_h",
        "Wood H o0.5",
        tier=Tier.MODERATE,
        fair=0.71,
        line=0.5,
        player_id=7,
        side="over",
        price=-240.0,
    )
    under = _rec(
        "batter_h", "Wood H u0.5", fair=0.29, line=0.5, player_id=7, side="under", price=190.0
    )
    fade_disagreements([over, under], 0.5)
    assert over.tier is Tier.MODERATE
    assert under.tier is Tier.PASS
    assert under.fade_of is None


def test_run_line_fade_takes_the_mirrored_line_not_the_other_dog() -> None:
    home_fav = _rec(
        "game_rl",
        "SD -1.5",
        tier=Tier.MODERATE,
        fair=0.41,
        line=-1.5,
        team_side="home",
        side="cover",
        price=140.0,
    )
    away_dog = _rec(
        "game_rl", "WSH +1.5", fair=0.59, line=1.5, team_side="away", side="cover", price=-165.0
    )
    away_fav = _rec(
        "game_rl", "WSH -1.5", fair=0.25, line=-1.5, team_side="away", side="cover", price=290.0
    )
    home_dog = _rec(
        "game_rl", "SD +1.5", fair=0.75, line=1.5, team_side="home", side="cover", price=-380.0
    )
    fade_disagreements([home_fav, away_dog, away_fav, home_dog], 0.5)

    assert home_fav.tier is Tier.PASS and home_fav.pass_gate == BOOK_FADE_GATE
    assert away_dog.tier is Tier.MODERATE and away_dog.fade_of == "SD -1.5"
    assert away_fav.tier is Tier.PASS and away_fav.fade_of is None
    assert home_dog.tier is Tier.PASS and home_dog.fade_of is None


def test_moneyline_fade_is_the_other_team() -> None:
    away = _rec(
        "game_ml", "WSH", tier=Tier.STRONG, fair=0.44, team_side="away", side="win", price=125.0
    )
    home = _rec("game_ml", "SD", fair=0.56, team_side="home", side="win", price=-145.0)
    fade_disagreements([away, home], 0.5)
    assert away.tier is Tier.PASS
    assert home.tier is Tier.STRONG and home.fade_of == "WSH"


def test_no_priced_other_side_keeps_the_engine_bet() -> None:
    over = _rec(
        "batter_hrr",
        "Wood HRR o1.5",
        tier=Tier.STRONG,
        fair=0.40,
        line=1.5,
        player_id=7,
        side="over",
        price=145.0,
    )
    under = _rec(
        "batter_hrr", "Wood HRR u1.5", fair=None, line=1.5, player_id=7, side="under", price=None
    )
    fade_disagreements([over, under], 0.5)
    assert over.tier is Tier.STRONG
    assert over.pass_gate is None
    assert under.tier is Tier.PASS


def test_zero_threshold_disables_and_market_list_confines() -> None:
    over = _rec(
        "batter_h",
        "Wood H o1.5",
        tier=Tier.STRONG,
        fair=0.38,
        line=1.5,
        player_id=7,
        side="over",
        price=150.0,
    )
    under = _rec(
        "batter_h", "Wood H u1.5", fair=0.62, line=1.5, player_id=7, side="under", price=-190.0
    )
    fade_disagreements([over, under], 0.0)
    assert over.tier is Tier.STRONG and under.tier is Tier.PASS
    fade_disagreements([over, under], 0.5, frozenset({"batter_2b"}))
    assert over.tier is Tier.STRONG and under.tier is Tier.PASS
    fade_disagreements([over, under], 0.5, frozenset({"batter_h"}))
    assert over.tier is Tier.PASS and under.tier is Tier.STRONG


def test_fade_after_one_buy_gate_still_leaves_one_buy_per_market() -> None:
    o05 = _rec(
        "batter_h",
        "Wood H o0.5",
        tier=Tier.MODERATE,
        fair=0.70,
        line=0.5,
        player_id=7,
        side="over",
        price=-230.0,
    )
    u05 = _rec(
        "batter_h", "Wood H u0.5", fair=0.30, line=0.5, player_id=7, side="under", price=185.0
    )
    o15 = _rec(
        "batter_h",
        "Wood H o1.5",
        tier=Tier.STRONG,
        fair=0.38,
        line=1.5,
        player_id=7,
        side="over",
        price=150.0,
    )
    u15 = _rec(
        "batter_h", "Wood H u1.5", fair=0.62, line=1.5, player_id=7, side="under", price=-190.0
    )
    recs = fade_disagreements(enforce_one_buy_per_group([o05, u05, o15, u15]), 0.5)
    buys = [r for r in recs if r.tier is not Tier.PASS]
    assert [b.selection for b in buys] == ["Wood H u1.5"]
    assert buys[0].fade_of == "Wood H o1.5"


def test_fade_of_round_trips_through_json_and_old_files_load(tmp_path: Path) -> None:
    under = _rec(
        "batter_h",
        "Wood H u1.5",
        tier=Tier.STRONG,
        fair=0.62,
        line=1.5,
        player_id=7,
        side="under",
        price=-190.0,
    )
    under.fade_of = "Wood H o1.5"
    p = tmp_path / "preds.json"
    save_json([under], p)
    assert load_json(p)[0].fade_of == "Wood H o1.5"

    old = json.loads(p.read_text())
    del old[0]["fade_of"]
    p.write_text(json.dumps(old))
    assert load_json(p)[0].fade_of is None
