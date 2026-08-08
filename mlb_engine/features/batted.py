"""What counts as a batted-ball event.

Statcast tracks exit velocity on **fouls**, so ``launch_speed.notna()`` is not a
batted-ball filter: on the cached windows it returns ~1.9x the true batted-ball
count, roughly half of it foul balls. A tracked foul carries a ``launch_speed``
and a ``launch_angle`` but no ``launch_speed_angle`` and no ``bb_type``, which is
why barrel% (denominated on ``launch_speed_angle.dropna()``) was unaffected while
hard-hit% and sweet-spot% were diluted by nearly half -- and then compared
against true batted-ball league baselines (.400 hard-hit, .330 sweet-spot).

A batted-ball event is a ball put in play, i.e. ``description == "hit_into_play"``
with a tracked exit velocity.
"""

from __future__ import annotations

import pandas as pd

IN_PLAY = "hit_into_play"


def batted_balls(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``df`` that are tracked batted-ball events.

    Falls back to ``launch_speed.notna()`` when the frame has no ``description``
    column, which is the only way to identify a ball in play.
    """
    if "launch_speed" not in df:
        return df.iloc[:0]
    tracked = df["launch_speed"].notna()
    if "description" not in df:
        return df[tracked]
    return df[tracked & df["description"].eq(IN_PLAY)]
