"""Returning production: the one input measured to know something the closing
spread does not.

The study
---------
Nine candidate inputs were tested on 6,513 games (2014-2025) for residual signal
*after* the consensus closing spread. Every schedule-adjusted efficiency and
box-score measure came back at zero -- adjusted PPA -0.001, four-year recruiting
+0.008, line yards +0.017, explosiveness -0.012, none of whose season-clustered
intervals excluded zero. Returning production did not:

    partial r +0.0389, p = 0.001, season-clustered 95% CI [+0.020, +0.058]

with the effect present in every in-season bucket (weeks 4-7 +0.053, weeks 8+
+0.033) rather than only in September, which is the opposite of the obvious
story -- the market is not merely slow to update in week one.

Regressed on margin alongside the spread on held-out later seasons, the fitted
coefficient is **+2.5 points per unit of returning-production gap**. The gap's SD
is 0.34, so a typical game moves about 0.9 points and an extreme roster mismatch
about 3.

Why it is off by default
------------------------
Real is not the same as profitable. Betting this side of every game goes
**51.96% ATS (2663-2462-89, -0.8% ROI)** against a 52.38% break-even. That is by
far the closest anything in either engine has come to beating a price -- it
recovers four fifths of the vig -- and it still does not clear it. So the term
ships wired, documented and disabled; ``CFBE_RETURNING_PTS=2.5`` turns it on, and
CLV is the evidence that should decide whether it ever ships on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.data.teamnames import school_key

log = logging.getLogger(__name__)

# Fitted on 2014-2025, held-out later seasons: points of margin per unit of gap.
# This is the value to put in ``CFBE_RETURNING_PTS`` if the term is ever enabled.
FITTED_PTS_PER_UNIT = 2.5


@dataclass
class ReturningBook:
    """Share of last season's production each team brings back."""

    shares: dict[str, float]

    def get(self, team_name: str) -> float | None:
        return self.shares.get(school_key(team_name))

    def gap(self, home: str, away: str) -> float | None:
        """Home minus away returning share; ``None`` unless both are known.

        A missing side is deliberately *not* imputed to a league-average share.
        The +2.5 pts/unit coefficient was fit only on games where both teams
        appear in CFBD's returning-production table, so imputing one side would
        apply a measured coefficient to a gap the measurement never saw -- and a
        team absent from that table is usually an FCS opponent, whose returning
        share is not league-average to begin with.
        """
        h, a = self.get(home), self.get(away)
        if h is None or a is None:
            return None
        return h - a

    def margin_delta(self, home: str, away: str, pts_per_unit: float, cap: float) -> float:
        """Points to add to the home margin for the experience edge."""
        if pts_per_unit <= 0:
            return 0.0
        gap = self.gap(home, away)
        if gap is None:
            return 0.0
        return max(-cap, min(cap, gap * pts_per_unit))


def build_returning_book(cfbd: CFBDClient, season: int) -> ReturningBook | None:
    shares = cfbd.fetch_returning_production(season)
    if not shares:
        return None
    log.info("returning production: %d teams for %d", len(shares), season)
    return ReturningBook(shares=shares)
