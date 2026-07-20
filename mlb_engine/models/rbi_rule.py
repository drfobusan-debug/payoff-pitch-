"""RBI hard rule.

Flags a hitter when the on-base percentage of the three batters immediately
preceding him in the order (over the 3-week rolling window) exceeds a threshold
(default .345). A flagged hitter has more RBI opportunities than his own line
implies, so the rule applies a bounded multiplier to his simulated RBI total.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlb_engine.features.rolling import BatterProfile


@dataclass
class RBIFlag:
    slot: int  # 0-based lineup index
    preceding_obp: float
    flagged: bool


def evaluate_lineup(
    profiles_in_order: list[BatterProfile],
    threshold: float = 0.345,
) -> list[RBIFlag]:
    """Return an RBI flag per lineup slot (order = index 0..8)."""
    n = len(profiles_in_order)
    flags: list[RBIFlag] = []
    for i in range(n):
        preceding = [profiles_in_order[(i - k) % n] for k in (1, 2, 3)]
        obps = [p.overall.obp for p in preceding]
        avg = sum(obps) / len(obps) if obps else 0.0
        flags.append(RBIFlag(slot=i, preceding_obp=avg, flagged=avg > threshold))
    return flags


def rbi_multiplier(flag: RBIFlag, max_boost: float = 0.12) -> float:
    """Bounded multiplier for a flagged slot's RBI total.

    Scales with how far preceding OBP exceeds the threshold, capped at max_boost.
    """
    if not flag.flagged:
        return 1.0
    excess = flag.preceding_obp - 0.345
    boost = min(max_boost, max(0.0, excess) * 2.0)
    return 1.0 + boost
