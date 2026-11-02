"""Markets the MLB card prices and grades but never buys.

The 2026 engine lost to the market in every family it bet: 3,619 buys returned
-9.2% (se 1.9) at a close-EV of -3.6%. On all 157,878 graded rows with a
devigged price, the model's gap to the market moved the result by +0.05
(+/- 0.02) per unit on batter props, 0.00 (+/- 0.05) on game lines and +0.01
(+/- 0.04) on pitcher props: the disagreement the engine bought was noise. So
every market starts the season graded-only, as CFB totals do. A buy the model
wanted is stamped ``graded_only_<market>`` and kept on the ledger at the price
it would have taken, and screen probation grades it like any other refusal: a
``LIFT`` verdict on held-out games is the case for buying that market again.

``MLBE_GRADED_ONLY_MARKETS`` is a comma list; ``*`` (the default) covers every
market and an empty value turns graded-only off.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ALL = "*"


def gate(market: str) -> str:
    """The ``pass_gate`` a graded-only refusal is stamped with."""
    return f"graded_only_{market}"


def is_gate(pass_gate: str | None) -> bool:
    return bool(pass_gate) and str(pass_gate).startswith("graded_only_")


@dataclass(frozen=True)
class GradedOnly:
    """Which markets are priced and graded without being bought."""

    markets: frozenset[str] = frozenset({ALL})

    @classmethod
    def from_env(cls) -> GradedOnly:
        raw = os.getenv("MLBE_GRADED_ONLY_MARKETS")
        if raw is None:
            return cls()
        return cls(frozenset(p.strip() for p in raw.split(",") if p.strip()))

    def covers(self, market: str) -> bool:
        return ALL in self.markets or market in self.markets

    def verdict(self, market: str, tier_label: str) -> tuple[bool, str, str | None]:
        """Return ``(keep_buy, reason, pass_gate)`` for a buy in ``market``."""
        if not self.covers(market):
            return True, "", None
        return (
            False,
            f"graded only, not a bet (would have been {tier_label}) -> PASS",
            gate(market),
        )
