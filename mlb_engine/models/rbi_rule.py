"""RBI prediction logic.

RBI is a dependent counting stat: it is the intersection of a hitter's own skill
and his team-dependent opportunity. The model follows a three-tier hierarchy:

  * Tier 1 (most sensitive / volume): lineup slot + on-base pct of the three
    batters immediately preceding him (3-week window). A preceding OBP > .345
    trips the user's hard rule and scales up his RBI opportunity.
  * Tier 2 (highest PPV / skill): xSLG (contact quality) as a proxy for
    xSLG-with-runners-on; elite quality of contact reliably cashes opportunities.
  * Tier 3 (highest NPV / system failure): an in-zone contact-rate collapse
    (< ~72%) caps the RBI total regardless of opportunity — a high-whiff hitter
    cannot drive runners in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mlb_engine.features.regression import BatterRegression
from mlb_engine.features.rolling import BatterProfile

OBP_BASELINE = 0.345
BL_XSLG = 0.400
ZONE_CONTACT_NPV = 0.72  # in-zone contact collapse threshold
MIN_BBE = 15


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class RBIFlag:
    slot: int  # 0-based lineup index
    preceding_obp: float
    flagged: bool
    xslg: float = float("nan")
    zone_contact: float = float("nan")
    bbe: int = 0


def evaluate_lineup(
    profiles_in_order: list[BatterProfile],
    threshold: float = OBP_BASELINE,
    regressions: list[BatterRegression] | None = None,
) -> list[RBIFlag]:
    """Return an RBI flag per lineup slot (order = index 0..8).

    ``regressions`` (aligned to the lineup order) supplies each hitter's own
    quality-of-contact metrics for the PPV/NPV tiers.
    """
    n = len(profiles_in_order)
    flags: list[RBIFlag] = []
    for i in range(n):
        preceding = [profiles_in_order[(i - k) % n] for k in (1, 2, 3)]
        obps = [p.overall.obp for p in preceding]
        avg = sum(obps) / len(obps) if obps else 0.0
        reg = regressions[i] if regressions and i < len(regressions) else None
        flags.append(
            RBIFlag(
                slot=i,
                preceding_obp=avg,
                flagged=avg > threshold,
                xslg=reg.xslg if reg else float("nan"),
                zone_contact=reg.zone_contact if reg else float("nan"),
                bbe=reg.bbe if reg else 0,
            )
        )
    return flags


def rbi_multiplier(flag: RBIFlag, max_boost: float = 0.15) -> float:
    """Bounded multiplier on a slot's simulated RBI total (three-tier model)."""
    m = 1.0

    # Tier 1 - volume: opportunity from preceding on-base production.
    if flag.flagged:
        excess = flag.preceding_obp - OBP_BASELINE
        m *= 1.0 + min(max_boost, max(0.0, excess) * 2.0)

    # Tier 2 - PPV: quality-of-contact (xSLG) cashes opportunities.
    if flag.bbe >= MIN_BBE and not math.isnan(flag.xslg):
        m *= 1.0 + _clip((flag.xslg - BL_XSLG) * 0.30, -0.08, 0.12)

    # Tier 3 - NPV: in-zone contact collapse caps RBI regardless of opportunity.
    if not math.isnan(flag.zone_contact) and flag.zone_contact < ZONE_CONTACT_NPV:
        deficit = ZONE_CONTACT_NPV - flag.zone_contact
        m *= 1.0 - _clip(deficit * 1.5, 0.0, 0.15)

    return _clip(m, 0.80, 1.30)
