"""The screens that apply to a prop's under, and the markets that get one.

Both sides of a prop are priced (``test_prop_both_sides``) and one buy survives
per prop group (``test_one_buy_per_prop``). What is left is the fade's own
discipline: which markets are worth fading at all, the price beyond which a fade
cannot pay, and the one market-specific profile fitted on under rows.
"""

from __future__ import annotations

from types import SimpleNamespace

from mlb_engine.config import Config, EVThresholds
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline

# A 0.55 fade at -110 does not clear the shipped conviction floor once the
# probability is anchored, and the screens under test here are the fade's own.
LEVELS_OFF = EVThresholds(min_prob=0.0, max_ev=1.0)


def _pipeline(cfg: Config) -> Pipeline:
    p = Pipeline.__new__(Pipeline)
    p.cfg = cfg
    p._calibrator = SimpleNamespace(apply=lambda market, prob: prob)
    p._shrink = None
    p._splits = {}
    p._quote_aliases = {}
    return p


def _sides(
    cfg: Config,
    market: str,
    stat: str,
    p_over: float,
    over_price: float,
    under_price: float,
    **kw,
) -> dict[str, object]:
    """One prop line priced on both sides, keyed by side."""
    p = _pipeline(cfg)
    game = SimpleNamespace(game_date="2026-08-01", game_pk=1)
    price = {"over": over_price, "under": under_price}
    quotes = {
        ("MATCH", market, f"Some Hitter {stat} {'o' if s == 'over' else 'u'}0.5"): [
            MarketQuote(
                book="dk",
                american=price[s],
                opposite_american=price["under" if s == "over" else "over"],
            )
        ]
        for s in ("over", "under")
    }
    out = {}
    for side in p._prop_sides(market):
        out[side] = p._mk(
            game, "MATCH", "batter", market,
            f"Some Hitter {stat} {'o' if side == 'over' else 'u'}0.5", p_over,
            line=0.5, player_id=7, stat=stat, side=side, quotes=quotes, **kw,
        )
    return out


def test_rare_event_markets_are_priced_over_only() -> None:
    """Home runs, doubles and triples never get a fade.

    Their under is a heavy favourite -- a home-run under is around -600 -- so the
    vig eats the edge, and the engine's doubles number is now deliberately
    near-flat, which makes a fade there a bet on the prior.
    """
    p = _pipeline(Config())
    for market in ("batter_hr", "batter_2b", "batter_3b"):
        assert p._prop_sides(market) == ("over",)
    for market in ("batter_h", "batter_1b", "pitcher_k", "pitcher_er"):
        assert p._prop_sides(market) == ("over", "under")


def test_a_deep_favourite_under_is_refused() -> None:
    """Shorter than the price floor the fade has nothing to win.

    The shadow book's unders priced worse than -300 won 77% of the time and
    still returned -2.1%: the payout stops covering the hit rate.
    """
    sides = _sides(Config(), "batter_h", "H", 0.22, 260, -320)
    assert sides["under"].tier is Tier.PASS
    assert sides["under"].pass_gate == "under_price_floor"
    # The same edge at a payable price is a buy, so the floor is what refused it.
    priced = _sides(Config(prop_under_min_price=-1000.0), "batter_h", "H", 0.22, 260, -320)
    assert priced["under"].tier in (Tier.STRONG, Tier.MODERATE)


def test_a_singles_under_needs_the_profile() -> None:
    """The fade's one market-specific screen, on the market it was fitted on.

    Fitted out of time on 20,413 batter-games: a high-K, fly-ball-tilted bat is
    the only cell that predicted a no-single game in all 13 blocks.
    """

    def under(score: float | None):
        return _sides(
            # 0.60 on the fade, which anchors to 0.57 and keeps a 7-point edge:
            # singles carry a raised edge floor of their own.
            Config(ev=LEVELS_OFF), "batter_1b", "1B", 0.40, -110, -110,
            bat_singles_under=score,
        )["under"]

    for score in (3.0, 2.0):
        assert under(score).tier in (Tier.STRONG, Tier.MODERATE)
    # A contact bat is not a fade, and neither is a batter with no profile.
    for score in (1.0, 0.0, None):
        assert under(score).tier is Tier.PASS


def test_the_hits_under_carries_no_singles_screen() -> None:
    """The singles profile does not transfer: it went 38.9% for -16.9% on hits.

    A strikeout-prone bat still doubles and homers, and either clears a hits
    line, so the hits fade runs on price and EV alone.
    """
    sides = _sides(
        Config(ev=LEVELS_OFF), "batter_h", "H", 0.45, -110, -110, bat_singles_under=0.0
    )
    assert sides["under"].tier in (Tier.STRONG, Tier.MODERATE)
