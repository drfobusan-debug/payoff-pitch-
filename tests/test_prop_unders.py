"""Tests for the prop under side.

Every player prop used to be emitted over-only, so the engine's fade of a bad
number could only ever be a Pass. Both sides are now priced and exactly one
recommendation survives per line: the over, the under, or a pass.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlb_engine.config import Config
from mlb_engine.market import keys
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline


class _IdentityCalibrator:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _pipeline(cfg: Config) -> Pipeline:
    p = Pipeline.__new__(Pipeline)
    p.cfg = cfg
    p._calibrator = _IdentityCalibrator()
    p._shrink = None
    p._splits = {}
    p._quote_aliases = {}
    return p


def _hits_sides(
    cfg: Config,
    p_over: float,
    over_price: float,
    under_price: float,
    *,
    gate_reason: str | None = None,
    quote_under: bool = True,
):
    """One batter-hits line priced on both sides, returned as (over, under)."""
    p = _pipeline(cfg)
    game = SimpleNamespace(game_date="2026-08-01", game_pk=1)
    over_sel = keys.batter_prop("Some Hitter", "H", 0.5)
    under_sel = keys.batter_prop("Some Hitter", "H", 0.5, False)
    quotes = {
        ("MATCH", "batter_h", over_sel): [
            MarketQuote(book="dk", american=over_price, opposite_american=under_price)
        ]
    }
    if quote_under:
        quotes[("MATCH", "batter_h", under_sel)] = [
            MarketQuote(book="dk", american=under_price, opposite_american=over_price)
        ]
    recs = p._mk_sides(
        game, "MATCH", "batter", "batter_h", "Some Hitter", "H", p_over,
        line=0.5, player_id=7, stat="H", quotes=quotes, gate_reason=gate_reason,
    )
    assert len(recs) == 1
    return recs[0]


def test_one_recommendation_per_line() -> None:
    """Two sides are priced; one row comes back."""
    rec = _hits_sides(Config(), 0.62, -110, -110)
    assert rec.selection in (
        "Some Hitter H o0.5",
        "Some Hitter H u0.5",
    )


def test_the_under_is_recommended_when_it_is_the_priced_edge() -> None:
    # The model has the hitter at 45%, so the fade is a 5-point edge on a
    # no-vig 50% line -- inside the implausible-edge cap.
    rec = _hits_sides(Config(), 0.45, -110, -110)
    assert rec.side == "under"
    assert rec.selection == "Some Hitter H u0.5"
    assert rec.tier in (Tier.STRONG, Tier.MODERATE)
    # The number it is bet at is the complement of the same calibrated over.
    assert rec.model_prob == 0.55


def test_the_over_survives_when_it_is_the_edge() -> None:
    rec = _hits_sides(Config(), 0.55, -110, -110)
    assert rec.side == "over"
    assert rec.tier in (Tier.STRONG, Tier.MODERATE)


def test_neither_side_bettable_keeps_the_over_row() -> None:
    """The Pass row the ledger has always carried, so NPV history is comparable."""
    rec = _hits_sides(Config(), 0.50, -110, -110)
    assert rec.side == "over"
    assert rec.tier is Tier.PASS
    assert rec.pass_gate is not None


def test_an_over_only_gate_does_not_become_an_under_buy() -> None:
    """A contact-floor veto is a claim about the over, and does not travel.

    It must neither pass the under (it says nothing about it) nor promote it
    (it is not evidence for it): the under stands or falls on its own price.
    """
    gated = _hits_sides(Config(), 0.45, -110, -110, gate_reason="contact floor: weak bat")
    assert gated.side == "under"
    assert "contact floor: weak bat" not in gated.reasons
    # And the veto still removes the over it was aimed at.
    flat = _hits_sides(Config(), 0.55, -110, -110, gate_reason="contact floor: weak bat")
    assert flat.side == "over"
    assert flat.pass_gate == "contact_floor"


def test_no_under_quote_leaves_the_over() -> None:
    rec = _hits_sides(Config(), 0.45, -110, -110, quote_under=False)
    assert rec.side == "over"


def test_excluded_markets_are_over_only() -> None:
    """Home runs, doubles and triples never get a fade."""
    cfg = Config()
    for market in ("batter_hr", "batter_2b", "batter_3b"):
        assert market not in cfg.prop_under_markets


def test_home_run_line_emits_only_the_over() -> None:
    p = _pipeline(Config())
    game = SimpleNamespace(game_date="2026-08-01", game_pk=1)
    recs = p._mk_sides(
        game, "MATCH", "batter", "batter_hr", "Some Hitter", "HR", 0.12,
        line=0.5, player_id=7, stat="HR", quotes={},
    )
    assert [r.side for r in recs] == ["over"]


def _k_sides(cfg: Config, line: float, gate_reason: str | None):
    p = _pipeline(cfg)
    game = SimpleNamespace(game_date="2026-08-01", game_pk=1)
    pitcher = SimpleNamespace(name="Some Pitcher", mlbam_id=42)
    n = 200
    res = SimpleNamespace(
        pit={"home": {k: np.full(n, 8.0) for k in ("K", "outs", "H", "BB", "ER")}}
    )
    # Strikeouts land under any line at or below 8 half the time, so the model
    # sits at 50% and the under is the +EV side at these prices.
    res.pit["home"]["K"][: n // 2] = 4.0
    quotes = {
        ("MATCH", "pitcher_k", keys.pitcher_prop(pitcher.name, "Ks", line, over)): [
            MarketQuote(
                book="dk",
                american=-110.0 if over else 130.0,
                opposite_american=130.0 if over else -110.0,
            )
        ]
        for over in (True, False)
    }
    recs = p._pitcher_props(
        game, "MATCH", res, "home", pitcher, quotes, gate_reason=gate_reason
    )
    return [r for r in recs if r.line == line and r.market == "pitcher_k"]


def test_k_buy_cap_is_an_over_claim_and_the_under_still_prices() -> None:
    """The o6.5 cap says the model over-projects strikeouts, so the fade stands."""
    (rec,) = _k_sides(Config(pitcher_k_max_buy_line=5.5), 6.5, None)
    assert rec.side == "under"
    assert rec.tier in (Tier.STRONG, Tier.MODERATE)


def test_a_thin_starter_is_unbettable_in_both_directions() -> None:
    """The same prices that make the fade a buy above are refused here."""
    (rec,) = _k_sides(Config(), 5.5, "thin Statcast: Some Pitcher 0p < 150")
    assert rec.tier is Tier.PASS
