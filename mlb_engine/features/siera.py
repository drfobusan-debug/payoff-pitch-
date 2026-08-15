"""SIERA (Skill-Interactive ERA) computed directly from Statcast.

SIERA estimates a pitcher's true run-prevention skill from strikeouts, walks and
batted-ball type -- independent of park, defense and sequencing luck.  It is the
right lens for a batter-vs-starter matchup gate: a low-SIERA arm suppresses hits
(more Ks, weaker contact), a high-SIERA arm allows more.

We reproduce the FanGraphs (Swartz) revised-SIERA formula from the per-plate-
appearance strikeout, walk and net-groundball rates, all derivable from a
pitcher's Statcast slice (``events`` for K/BB/PA, ``bb_type`` for GB/FB/PU).

The formula is then **re-centred onto the published scale**, which is a step the
first version of this module missed. Swartz's coefficients were fitted to the
run environment of their era; FanGraphs re-centres the output every season so
that league SIERA equals league ERA. Skipping that left every arm reading about
four tenths of a run better than its published number -- see
``SIERA_LEAGUE_ANCHOR``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# FanGraphs revised-SIERA coefficients (rates as per-PA decimals).
_A0 = 6.145
_A_SO = -16.986
_A_BB = 11.434
_A_GB = -1.858  # net groundball = (GB - FB - PU) / PA
_A_SO2 = 7.653
_A_GB2 = 6.664  # sign flips with the net-groundball direction
_A_SO_GB = 10.130
_A_BB_GB = -5.195

MIN_SIERA_PA = 80  # min plate appearances before SIERA is trusted

# Where league-average SIERA is pinned. FanGraphs defines SIERA so that the
# league mean equals league ERA, and re-centres it each season; the raw Swartz
# polynomial does not do this on its own.
#
# Fed league-average rates (K/PA .220, BB/PA .086, net-GB/PA .057, all measured
# on the Statcast cache) the bare formula returns 3.64, against a published
# league SIERA that sits near 4.05. Every input rate checks out individually, so
# the gap is the missing re-centring and not a broken term.
#
# This offset is the one number in the module that is *assumed* rather than
# measured, and it will drift with the run environment. It is worth carrying
# anyway, because ``opp_starter_siera`` is written into the ledger and printed
# in the preview: a reader comparing it against FanGraphs should see the same
# number, and the ace/scrub cut points are quoted on that scale too.
SIERA_LEAGUE_ANCHOR = 4.05
_SIERA_RAW_LEAGUE = 3.64  # the bare polynomial at league-average rates
SIERA_RECENTRE = SIERA_LEAGUE_ANCHOR - _SIERA_RAW_LEAGUE

_K_EVENTS = ("strikeout", "strikeout_double_play")


@dataclass(frozen=True)
class Siera:
    """A pitcher's SIERA and the rates behind it over a Statcast window."""

    pa: int
    so_rate: float
    bb_rate: float
    net_gb_rate: float
    siera: float

    @property
    def has_data(self) -> bool:
        return self.pa >= MIN_SIERA_PA


def pitcher_siera(pdf: pd.DataFrame) -> Siera:
    """Compute SIERA for a pitcher from their pitch-level Statcast slice.

    Returns a ``Siera`` with ``siera == nan`` when there are no plate
    appearances; callers should check :attr:`Siera.has_data` before gating.
    """
    ev = pdf["events"].dropna()
    pa = int(len(ev))
    if not pa:
        return Siera(pa=0, so_rate=float("nan"), bb_rate=float("nan"),
                     net_gb_rate=float("nan"), siera=float("nan"))

    so = float(ev.isin(_K_EVENTS).sum()) / pa
    bb = float(ev.eq("walk").sum()) / pa

    bt = pdf["bb_type"].dropna() if "bb_type" in pdf else pd.Series(dtype=object)
    gb = float(bt.eq("ground_ball").sum())
    fb = float(bt.eq("fly_ball").sum())
    pu = float(bt.eq("popup").sum())
    net_gb = (gb - fb - pu) / pa

    gb2_coeff = _A_GB2 if net_gb < 0 else -_A_GB2
    siera = (
        _A0
        + _A_SO * so
        + _A_BB * bb
        + _A_GB * net_gb
        + _A_SO2 * so * so
        + gb2_coeff * net_gb * net_gb
        + _A_SO_GB * so * net_gb
        + _A_BB_GB * bb * net_gb
    )
    return Siera(
        pa=pa,
        so_rate=so,
        bb_rate=bb,
        net_gb_rate=net_gb,
        siera=round(float(siera) + SIERA_RECENTRE, 3),
    )


def faces_ace(opp: Siera | None, ace_floor: float) -> bool:
    """True when the opposing starter is an ace (SIERA below the floor)."""
    return opp is not None and opp.has_data and opp.siera < ace_floor


def faces_scrub(opp: Siera | None, bad_ceiling: float) -> bool:
    """True when the opposing starter is a weak arm (SIERA above the ceiling)."""
    return opp is not None and opp.has_data and opp.siera > bad_ceiling
