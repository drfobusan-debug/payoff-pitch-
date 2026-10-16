"""Markets the card prices and grades but never buys.

CFB totals ship here. On the 2026 card the model's total-minus-line gap moved the
Under rate +0.020 +/- 0.069 per point against the 0.123 its own normal model
implies, the no-vig market out-scored it on Brier at every total SD tried
(0.2495 against 0.2551-0.2578), and no edge band paid. A buy the model wanted is
stamped ``graded_only_<market>`` and kept on the ledger at the price it would
have taken, so the audit grades it as a screen: a ``LIFT`` verdict is the case
for buying the market again.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MARKETS = ("game_total",)


def gate(market: str) -> str:
    """The ``pass_gate`` a graded-only refusal is stamped with."""
    return f"graded_only_{market}"


def is_gate(pass_gate: str | None) -> bool:
    return bool(pass_gate) and str(pass_gate).startswith("graded_only_")


@dataclass(frozen=True)
class GradedOnly:
    """Which markets are priced and graded without being bought."""

    markets: frozenset[str] = frozenset(DEFAULT_MARKETS)

    @classmethod
    def from_env(cls) -> GradedOnly:
        raw = os.getenv("CFBE_GRADED_ONLY_MARKETS")
        if raw is None:
            return cls()
        return cls(frozenset(p.strip() for p in raw.split(",") if p.strip()))

    def verdict(self, market: str, tier_label: str) -> tuple[bool, str, str | None]:
        """Return ``(keep_buy, reason, pass_gate)`` for a buy in ``market``."""
        if market not in self.markets:
            return True, "", None
        return False, f"graded only, not a bet (would have been {tier_label}) -> PASS", gate(market)
