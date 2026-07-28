"""Data-driven starter workload / early-hook model.

The Monte Carlo caps a starter's exposure with ``starter_bf_cap`` (batters faced
before the bullpen takes over). A flat manager cap misses the biggest driver of
strikeout *unders*: starters who simply do not pitch deep. This derives an
expected batters-faced ceiling from the pitcher's own recent starts, and flags
opener / bullpen-game usage, so the sim stops accruing the starter's strikeouts
at a realistic exit point.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta

import pandas as pd

from mlb_engine.data.managers import DEFAULT_BF_CAP

OPENER_BF = 12  # median BF/start at or below this reads as an opener / bullpen game
MIN_STARTS = 2  # need at least this many recent starts before trusting the data
BF_BUFFER = 3  # allow a few more BF than the recent average (upside outings)


def _bf_per_start(pit_rows: pd.DataFrame, as_of: Date, form_days: int) -> list[int]:
    """Batters faced in each of the pitcher's starts over the form window."""
    end = as_of - timedelta(days=1)
    start = end - timedelta(days=form_days - 1)
    if "game_date" not in pit_rows or "events" not in pit_rows:
        return []
    window = pit_rows[(pit_rows["game_date"] >= start) & (pit_rows["game_date"] <= end)]
    pa = window[window["events"].notna()]
    if pa.empty:
        return []
    return [int(n) for n in pa.groupby("game_date").size().tolist()]


def expected_bf_cap(
    pit_rows: pd.DataFrame,
    as_of: Date,
    form_days: int,
    manager_cap: int = DEFAULT_BF_CAP,
) -> int:
    """Effective batters-faced cap = min(manager hook, recent workload + buffer).

    Falls back to the manager cap when there aren't enough recent starts. An
    opener / bullpen game (low median BF) collapses the cap to ``OPENER_BF``.
    """
    bf = _bf_per_start(pit_rows, as_of, form_days)
    if len(bf) < MIN_STARTS:
        return manager_cap
    bf_sorted = sorted(bf)
    median = bf_sorted[len(bf_sorted) // 2]
    if median <= OPENER_BF:
        return min(manager_cap, OPENER_BF)
    avg = sum(bf) / len(bf)
    return int(min(manager_cap, round(avg) + BF_BUFFER))
