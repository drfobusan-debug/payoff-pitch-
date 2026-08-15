"""Expected total bases per ball in play, from the league's own contact.

Statcast's expected slugging is a lookup: every batted ball is scored by what
balls hit at that exit velocity and launch angle actually produced, and a
hitter's xSLG is those expectations summed over his at-bats. The engine used to
approximate it by rescaling expected wOBA-on-contact -- ``xwOBAcon * 1.35`` --
which runs 87 points high against the slugging hitters actually post, because
xwOBAcon is a per-contact average with no strikeout in its denominator.

Fitting the lookup on the slate's own season keeps it self-calibrating: no table
to go stale, and the league mean lands on the league's real slugging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlb_engine.data.statcast import batted_balls

# 5 mph x 5 degrees. Finer cells fit the training balls better and predict held-
# out ones worse; coarser ones blur the line-drive window, whose value falls off
# within ten degrees on either side.
EV_EDGES = np.arange(40.0, 125.0, 5.0)
LA_EDGES = np.arange(-60.0, 65.0, 5.0)

# Batted balls a cell needs before it is trusted over the league ball. The
# corners of the grid -- 110 mph straight down, 45 mph at 50 degrees -- hold a
# handful of balls whose outcomes are otherwise taken at face value.
CELL_PRIOR = 25.0

TB_VALUE = {"single": 1.0, "double": 2.0, "triple": 3.0, "home_run": 4.0}
NON_AB_EVENTS = frozenset(
    {"walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"}
)

# Balls in play needed before a grid is worth fitting at all.
MIN_LEAGUE_BBE = 2_000


def _bins(ev: pd.Series, la: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.clip(np.digitize(ev.to_numpy(), EV_EDGES), 1, len(EV_EDGES) - 1),
        np.clip(np.digitize(la.to_numpy(), LA_EDGES), 1, len(LA_EDGES) - 1),
    )


@dataclass(frozen=True)
class LeagueXTB:
    """Total bases the league collects on a ball hit at each speed and angle."""

    cells: dict[tuple[int, int], float]
    league: float  # total bases per ball in play, all contact

    @classmethod
    def from_statcast(cls, df: pd.DataFrame) -> LeagueXTB | None:
        """Fit on every tracked ball in play in the frame, or ``None`` if too few."""
        if not {"launch_speed", "launch_angle", "events"}.issubset(df.columns):
            return None
        bip = batted_balls(df)
        bip = bip[bip["launch_angle"].notna() & bip["launch_speed"].notna()]
        if len(bip) < MIN_LEAGUE_BBE:
            return None
        tb = bip["events"].map(TB_VALUE).fillna(0.0)
        league = float(tb.mean())
        ev_bin, la_bin = _bins(bip["launch_speed"], bip["launch_angle"])
        grouped = pd.DataFrame({"ev": ev_bin, "la": la_bin, "tb": tb.to_numpy()}).groupby(
            ["ev", "la"]
        )["tb"]
        total, n = grouped.sum(), grouped.size()
        shrunk = (total + league * CELL_PRIOR) / (n + CELL_PRIOR)
        return cls(cells={(int(e), int(la)): float(v) for (e, la), v in shrunk.items()}, league=league)

    def expected(self, batted: pd.DataFrame) -> pd.Series:
        """Expected total bases for each tracked ball in ``batted``."""
        tracked = batted[batted["launch_angle"].notna() & batted["launch_speed"].notna()]
        if tracked.empty:
            return pd.Series(dtype=float)
        ev_bin, la_bin = _bins(tracked["launch_speed"], tracked["launch_angle"])
        return pd.Series(
            [
                self.cells.get((int(e), int(la)), self.league)
                for e, la in zip(ev_bin, la_bin, strict=True)
            ],
            index=tracked.index,
        )

    def xslg(self, batted: pd.DataFrame, at_bats: int) -> float:
        """Expected slugging: expected bases over at-bats, the way SLG is built.

        Untracked contact is credited at the league ball rather than at zero, so a
        missing reading costs a hitter nothing but sample.
        """
        if at_bats <= 0 or batted.empty:
            return float("nan")
        exp = self.expected(batted)
        missing = len(batted) - len(exp)
        return float((exp.sum() + missing * self.league) / at_bats)
