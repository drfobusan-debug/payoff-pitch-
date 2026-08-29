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

One tail refuses by default: **moneyline dogs longer than +200**. That is not a
CFB measurement -- there is no graded CFB row yet -- it is the MLB engine's, where
the Strong tier filled with plus-money dogs and inverted against Moderate, and
where model EV on moneylines scored AUC 0.33 (worse than a coin) while market
signal scored 0.80. A +260 dog priced off that input is the one bet built
entirely from the weakest thing the engine knows, so it is refused ahead of the
evidence rather than after it, and ``screen_probation`` grades the refusals
(``price_too_long``) counterfactually -- a ``LIFT`` verdict is how it comes back.

Everything else refuses nothing: the engine-wide band is off, the moneyline's
short tail is disarmed (``min_american=None``), and the numbers are MLB's, so an
out-of-band price elsewhere is only annotated on the card. Off-switch for the
live tail is per-market -- ``CFBE_PRICE_BAND_GAME_ML=0`` -- because a market
default is deliberately not something the engine-wide flag can silently undo.

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
# Longest dog worth a stake. Live for the moneyline (see ``MARKET_DEFAULTS``),
# and graded as a screen through ``probation.screen_probation``.
DEFAULT_MAX_AMERICAN = 200.0

SHORT_GATE = "price_too_short"
LONG_GATE = "price_too_long"


def _num_or_none(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if raw.strip().lower() in ("none", "off"):
        return None
    return float(raw)


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


@dataclass(frozen=True)
class PriceBand:
    """Prices a market is allowed to buy, and whether the band actually refuses.

    ``enabled`` governs *acting*, not measuring: off, an out-of-band price is
    reported as a reason string and bought anyway, which is what makes the
    refusal gradeable later.
    """

    enabled: bool = False
    min_american: float | None = DEFAULT_MIN_AMERICAN
    max_american: float | None = DEFAULT_MAX_AMERICAN

    @classmethod
    def from_env(cls) -> PriceBand:
        return cls(
            enabled=_flag("CFBE_PRICE_BAND", False),
            min_american=_num_or_none("CFBE_PRICE_MIN", DEFAULT_MIN_AMERICAN),
            max_american=_num_or_none("CFBE_PRICE_MAX", DEFAULT_MAX_AMERICAN),
        )

    def for_market(self, market: str) -> PriceBand:
        """Per-market band, overridable via ``CFBE_PRICE_MIN_GAME_ML`` etc.

        Per-market because one band cannot serve both boards: the moneyline runs
        from -3000 to +2500 while the spread barely leaves -110.

        A market listed in :data:`MARKET_DEFAULTS` starts from its own band
        rather than the engine-wide one, so switching the engine-wide flag on
        cannot widen a live tail and leaving it off cannot disarm one. Only the
        per-market variables move a market default.
        """
        suffix = market.upper()
        base = MARKET_DEFAULTS.get(market.lower(), self)
        return PriceBand(
            enabled=_flag(f"CFBE_PRICE_BAND_{suffix}", base.enabled or self.enabled),
            min_american=_num_or_none(f"CFBE_PRICE_MIN_{suffix}", base.min_american),
            max_american=_num_or_none(f"CFBE_PRICE_MAX_{suffix}", base.max_american),
        )

    def verdict(self, american: float | None) -> tuple[bool, str, str | None]:
        """``(keep, reason, gate)`` for a side priced at ``american``.

        Neutral on a missing price: an unpriced row is a data hole, and refusing
        it would file that hole under a betting screen in the probation table.
        """
        if american is None:
            return True, "", None
        if self.min_american is not None and american < self.min_american:
            return self._refuse(
                f"price {american:+.0f} shorter than {self.min_american:+.0f}", SHORT_GATE
            )
        if self.max_american is not None and american > self.max_american:
            return self._refuse(
                f"price {american:+.0f} longer than {self.max_american:+.0f}", LONG_GATE
            )
        return True, "", None

    def _refuse(self, reason: str, gate: str) -> tuple[bool, str, str | None]:
        if self.enabled:
            return False, f"{reason} -> PASS", gate
        return True, f"{reason} (band off, measuring)", None


# The one live tail. The short end is ``None`` rather than -250 because a short
# favourite loses money slowly and legibly, while a long dog is where this
# engine's worst-measured input meets its largest EV error -- only the second is
# worth refusing without a CFB number behind it.
MARKET_DEFAULTS: dict[str, PriceBand] = {
    "game_ml": PriceBand(
        enabled=True, min_american=None, max_american=DEFAULT_MAX_AMERICAN
    ),
}
