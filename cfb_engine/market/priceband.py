"""Refuse prices outside a band -- measured now, acted on later.

The tier logic already ranks on edge rather than EV, which stops a long price
buying its way into the Strong tier arithmetically (``EV = decimal_odds x
edge``). It does not stop the price itself being the problem, and the two tails
fail for different reasons:

* **A long dog** turns a small probability error into a large EV error. At +400
  the model needs to be right about a 20% event, and a one-point probability
  miss is five points of EV; the simulator's tails are the part of a fitted
  distribution least worth trusting. The MLB engine's Strong tier filled with
  plus-money dogs and inverted against Moderate (39.9% against 46.9%).
* **A short favourite** has to be right about a near-certainty to earn anything,
  so most of the stake is exposed to the one outcome the price says will not
  happen -- and in college football that outcome is a 40-point underdog covering
  a moneyline nobody shops.

In practice this is a **moneyline screen**: ATS and totals prices cluster inside
-120/+100, so a band drawn outside that range can only ever bite on the
moneyline, and a band drawn inside it would refuse the spread board wholesale.
Hence a per-market override rather than one number for the engine.

Defaults refuse nothing. ``enabled`` is off, and the band itself is MLB's
(-250 to +200) because there is no graded CFB row to draw one from -- the same
reason ``cfb_engine.market.drift`` measures without vetoing, and the same
discipline that caught the VSiN home-field table and the marking bumps testing
null. With the band off, a row outside it is still annotated on the card, so
``screen_probation`` can grade what the veto would have cost before it is
switched on.

Sign convention: American odds, so ``-250`` is shorter than ``-150`` and ``+400``
is longer than ``+200``. The band is *inclusive* -- a price exactly on the number
is kept, because a threshold that refuses its own boundary makes every backtest
of it a point estimate at the boundary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Shortest favourite worth a stake. MLB's number, unverified here.
DEFAULT_MIN_AMERICAN = -250.0
# Longest dog worth a stake, and the same +200 that
# ``probation.CANDIDATE_SCREENS`` grades as a candidate.
DEFAULT_MAX_AMERICAN = 200.0

SHORT_GATE = "price_too_short"
LONG_GATE = "price_too_long"


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


def _num(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class PriceBand:
    """Prices a market is allowed to buy, and whether the band actually refuses.

    ``enabled`` governs *acting*, not measuring: off, an out-of-band price is
    reported as a reason string and bought anyway, which is what makes the
    refusal gradeable later.
    """

    enabled: bool = False
    min_american: float = DEFAULT_MIN_AMERICAN
    max_american: float = DEFAULT_MAX_AMERICAN

    @classmethod
    def from_env(cls) -> PriceBand:
        return cls(
            enabled=_flag("CFBE_PRICE_BAND", False),
            min_american=_num("CFBE_PRICE_MIN", DEFAULT_MIN_AMERICAN),
            max_american=_num("CFBE_PRICE_MAX", DEFAULT_MAX_AMERICAN),
        )

    def for_market(self, market: str) -> PriceBand:
        """Per-market band, overridable via ``CFBE_PRICE_MIN_GAME_ML`` etc.

        Per-market because one band cannot serve both boards: the moneyline runs
        from -3000 to +2500 while the spread barely leaves -110.
        """
        suffix = market.upper()
        return PriceBand(
            enabled=_flag(f"CFBE_PRICE_BAND_{suffix}", self.enabled),
            min_american=_num(f"CFBE_PRICE_MIN_{suffix}", self.min_american),
            max_american=_num(f"CFBE_PRICE_MAX_{suffix}", self.max_american),
        )

    def verdict(self, american: float | None) -> tuple[bool, str, str | None]:
        """``(keep, reason, gate)`` for a side priced at ``american``.

        Neutral on a missing price: an unpriced row is a data hole, and refusing
        it would file that hole under a betting screen in the probation table.
        """
        if american is None:
            return True, "", None
        if american < self.min_american:
            return self._refuse(
                f"price {american:+.0f} shorter than {self.min_american:+.0f}", SHORT_GATE
            )
        if american > self.max_american:
            return self._refuse(
                f"price {american:+.0f} longer than {self.max_american:+.0f}", LONG_GATE
            )
        return True, "", None

    def _refuse(self, reason: str, gate: str) -> tuple[bool, str, str | None]:
        if self.enabled:
            return False, f"{reason} -> PASS", gate
        return True, f"{reason} (band off, measuring)", None
