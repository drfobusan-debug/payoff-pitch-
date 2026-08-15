"""Both sides of a prop are priced, and the two sides are one opinion.

The engine used to build every prop selection as ``o{line}`` and drop the under
outcome at parse time, so a prop's only expressible states were *buy the over*
and *pass*. That is what makes an over-shaded market unbeatable: on the 6,869
graded prop rows that carried both prices, backing every over returned -14.3%
while backing every under returned -1.4%, and the engine could only ever pick
the first of those.
"""

from types import SimpleNamespace

import numpy as np

from mlb_engine.config import Config
from mlb_engine.data.oddsapi import _opposite_prices
from mlb_engine.market import keys
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline


class _ShrinkToHalf:
    """A stand-in calibration map, so the complement is testably non-trivial."""

    def apply(self, prob: float) -> float:
        return 0.5 + 0.5 * (prob - 0.5)


def _pipeline(cfg: Config, shrink: object | None = None) -> Pipeline:
    p = Pipeline.__new__(Pipeline)
    p.cfg = cfg
    p._calibrator = SimpleNamespace(apply=lambda market, prob: prob)
    p._shrink = shrink
    p._splits = {}
    return p


def _pitcher_recs(
    cfg: Config, line: float, shrink: object | None = None, frac_low: float = 0.5
) -> list:
    p = _pipeline(cfg, shrink)
    game = SimpleNamespace(game_date="2026-08-01", game_pk=1)
    pitcher = SimpleNamespace(name="Some Pitcher", mlbam_id=42)
    n = 200
    res = SimpleNamespace(
        pit={"home": {k: np.full(n, 8.0) for k in ("K", "outs", "H", "BB", "ER")}}
    )
    res.pit["home"]["K"][: int(n * frac_low)] = 4.0
    quotes = {
        ("MATCH", "pitcher_k", keys.pitcher_prop(pitcher.name, "Ks", line, side)): [
            MarketQuote(book="dk", american=120.0, opposite_american=-140.0)
        ]
        for side in ("over", "under")
    }
    return p._pitcher_props(game, "MATCH", res, "home", pitcher, quotes)


def test_both_sides_are_emitted_and_sum_to_one() -> None:
    recs = _pitcher_recs(Config(), 5.5)
    k = {r.side: r for r in recs if r.market == "pitcher_k" and r.line == 5.5}
    assert set(k) == {"over", "under"}
    # One opinion, two expressions. The model can never hold contradictory
    # probabilities for the two sides of the same line.
    assert abs(k["over"].model_prob + k["under"].model_prob - 1.0) < 1e-9
    assert abs(k["over"].raw_prob + k["under"].raw_prob - 1.0) < 1e-9


def test_the_under_complements_the_calibrated_over_not_the_raw_one() -> None:
    """Calibration is fit on the over, so the complement must come after it.

    Calibrating each side independently would give ``cal(p) + cal(1-p)``, which
    is not 1 for any non-identity map -- the engine would price both sides of one
    line off two different beliefs.
    """
    # Away from 0.5, which is the shrink map's fixed point.
    recs = _pitcher_recs(Config(), 5.5, shrink=_ShrinkToHalf(), frac_low=0.25)
    k = {r.side: r for r in recs if r.market == "pitcher_k" and r.line == 5.5}
    assert abs(k["over"].model_prob + k["under"].model_prob - 1.0) < 1e-9
    # The map really did move the over, so this is not a trivial pass.
    assert abs(k["over"].model_prob - k["over"].raw_prob) > 1e-9


def test_the_over_buy_cap_does_not_gate_the_under() -> None:
    """The K cap is a screen on buying the over, not a view on the line."""
    recs = _pitcher_recs(Config(pitcher_k_max_buy_line=5.5), 6.5)
    k = {r.side: r for r in recs if r.market == "pitcher_k" and r.line == 6.5}
    assert k["over"].tier is Tier.PASS
    assert any("buy cap" in r for r in k["over"].reasons)
    assert not any("buy cap" in r for r in k["under"].reasons)


def test_prop_keys_carry_a_side_and_the_over_spelling_is_unchanged() -> None:
    # The over must keep its exact historic spelling, or every selection already
    # in the ledger and the closing captures stops matching by string.
    assert keys.batter_prop("Aaron Judge", "H", 1.5) == "Aaron Judge H o1.5"
    assert keys.batter_prop("Aaron Judge", "H", 1.5, "over") == "Aaron Judge H o1.5"
    assert keys.batter_prop("Aaron Judge", "H", 1.5, "under") == "Aaron Judge H u1.5"
    assert keys.pitcher_prop("Pablo Lopez", "Ks", 5.5) == "Pablo Lopez Ks o5.5"
    assert keys.pitcher_prop("Pablo Lopez", "Ks", 5.5, "under") == "Pablo Lopez Ks u5.5"


def test_two_overs_for_different_players_are_not_opposite_sides() -> None:
    """A book listing only the over for two players is a two-outcome market.

    Pairing those devigs one longshot against an unrelated one. The signature is
    a fair probability *above* the raw implied, which removing vig cannot do:
    +390 against +575 returns .579 where the honest number is near .196. This
    corrupted 83% of ``batter_hr`` rows, by a mean of +.14.
    """
    outcomes = [
        {"name": "Over", "description": "Gunnar Henderson", "point": 0.5, "price": 390.0},
        {"name": "Over", "description": "Jackson Holliday", "point": 0.5, "price": 575.0},
    ]
    assert _opposite_prices(outcomes) == {}
    # Unpaired falls back to the raw implied, which overstates the market by
    # about half the hold -- wrong, but honestly wrong, and never above implied.
    q = MarketQuote(book="dk", american=390.0, opposite_american=None)
    assert abs(q.no_vig_prob - 100 / 490) < 1e-9


def test_a_real_pair_still_devigs() -> None:
    outcomes = [
        {"name": "Over", "description": "Gunnar Henderson", "point": 0.5, "price": 390.0},
        {"name": "Under", "description": "Gunnar Henderson", "point": 0.5, "price": -520.0},
    ]
    opp = _opposite_prices(outcomes)
    assert opp[id(outcomes[0])] == -520.0
    assert opp[id(outcomes[1])] == 390.0
    q = MarketQuote(book="dk", american=390.0, opposite_american=-520.0)
    assert q.no_vig_prob < 100 / 490  # the devig can only move a side down


def test_a_team_two_way_market_still_pairs() -> None:
    """Moneylines and run lines carry no player, and their two points differ."""
    ml = [{"name": "Minnesota Twins", "price": -130.0},
          {"name": "Cleveland Guardians", "price": 110.0}]
    assert _opposite_prices(ml)[id(ml[0])] == 110.0
    rl = [{"name": "Minnesota Twins", "price": 105.0, "point": -1.5},
          {"name": "Cleveland Guardians", "price": -125.0, "point": 1.5}]
    assert _opposite_prices(rl)[id(rl[0])] == -125.0


def test_the_same_side_twice_is_never_a_pair() -> None:
    """Two outcomes for one player and line that carry the same name."""
    dup = [{"name": "Over", "description": "Aaron Judge", "point": 0.5, "price": -150.0},
           {"name": "Over", "description": "Aaron Judge", "point": 0.5, "price": -145.0}]
    assert _opposite_prices(dup) == {}
