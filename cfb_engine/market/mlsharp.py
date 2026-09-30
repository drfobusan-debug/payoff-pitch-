"""Sharp-money confirmation for moneyline buys.

The moneyline is the one market where this family of engines has evidence that
its *own* number is the problem. In the sibling MLB engine, graded ``game_ml``
buys were won *less* often the higher the model's EV said they should be (EV
AUC 0.33, p=0.004 over 102 rows) -- the engine was reliably taking the wrong
side of a two-way market -- while handle% minus tickets% on the side bet came
out at AUC 0.80 (p=0.027). Winning buys averaged +19.7 points of divergence,
losers -2.6.

So this screen asks the money to agree before a moneyline buy ships: the side
must be taking at least as large a share of the handle as of the tickets. It
never *creates* a bet -- MLB also measured an upgrade path (passes with
divergence >= +5 won 62% of 32) and it is deliberately not ported, because
promoting a row the EV screen rejected is a new bet justified by another sport's
sample.

Three deliberate limits:

* Moneyline only. The MLB inversion was specific to ``game_ml``; ATS and totals
  showed nothing like it, and a screen applied where nothing was measured is a
  guess with a config flag.
* Neutral on missing data. VSiN posts no split for much of the board (FCS
  visitors, weeknight mid-majors), and a gate that fired on absent data would
  turn a data hole into a betting decision.
* Attributed and reversible. A refusal is stamped ``ml_no_sharp_money`` so the
  audit grades the screen on the bets it *refused*, and ``CFBE_ML_SHARP_GATE=0``
  turns it into a measurement that annotates and refuses nothing.

Nothing here is measured on college football -- there are no graded CFB rows
yet. What ports is the direction, not the threshold: the default asks only that
the money keep pace with the tickets (divergence >= 0), which is the weakest
form of the finding, rather than MLB's +19.7 winner average.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cfb_engine.data.vsin_splits import Split

# Points of handle% - tickets% a moneyline buy must clear. Zero is "the money at
# least keeps pace with the ticket count", i.e. refuse only sides the public is
# on more heavily than the money is.
DEFAULT_MIN_DIVERGENCE = 0.0

GATE = "ml_no_sharp_money"


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _num(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class SharpGate:
    """Whether the public-money split confirms a moneyline buy."""

    enabled: bool = True
    min_divergence: float = DEFAULT_MIN_DIVERGENCE

    @classmethod
    def from_env(cls) -> SharpGate:
        return cls(
            enabled=_flag("CFBE_ML_SHARP_GATE", True),
            min_divergence=_num("CFBE_ML_MIN_DIVERGENCE", DEFAULT_MIN_DIVERGENCE),
        )

    def verdict(self, split: Split | None) -> tuple[bool, str, str | None]:
        """Return ``(keep_buy, reason, pass_gate)`` for a moneyline buy.

        ``pass_gate`` is set only on an actual refusal, so a row the EV screen or
        the price band already rejected is never re-attributed to this one.
        """
        divergence = None if split is None else split.divergence
        if divergence is None:
            return True, "no public-money split on this side", None
        book = f" at {split.book}" if split is not None and split.book else ""
        detail = f"handle-minus-tickets {divergence:+.0f}{book}"
        if divergence >= self.min_divergence:
            return True, f"money confirms ({detail})", None
        reason = f"{detail}, under {self.min_divergence:+.0f}"
        if self.enabled:
            return False, f"{reason} -> PASS", GATE
        return True, f"{reason} (gate off, measuring)", None
