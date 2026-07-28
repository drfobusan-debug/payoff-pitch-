"""Public Baseball Savant expected-statistics feed (xSLG / xwOBA).

The Statcast expected-statistics leaderboard is a free, no-login CSV endpoint on
the same host the engine already uses. It exposes ``est_slg`` (xSLG) keyed by
``player_id`` (an MLBAM id), so the batter xSLG distribution-tail hook activates
without a FanGraphs subscription. xSLG is a season-to-date stabilizing skill
metric (the leaderboard's native granularity), folded into the tail z-composite
alongside the rolling-window Statcast metrics.

Returns ``{}`` on any failure so the tail layer stays neutral.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

log = logging.getLogger(__name__)

_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type=batter&year={year}&position=&team=&min=q&csv=true"
)
_HEADERS = {"User-Agent": "Mozilla/5.0 (mlb-prediction-engine)"}


def load_batter_xslg(year: int, timeout: int = 30) -> dict[int, float]:
    """Return ``{mlbam_id: est_slg}`` for qualified batters (``{}`` on failure)."""
    try:
        resp = requests.get(_URL.format(year=year), headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:  # optional enrichment
        log.warning("Savant expected-stats unavailable: %s", exc)
        return {}

    if not {"player_id", "est_slg"}.issubset(df.columns):
        log.warning("Savant expected-stats columns missing (have %s)", list(df.columns)[:8])
        return {}

    out: dict[int, float] = {}
    for _, row in df.iterrows():
        try:
            out[int(row["player_id"])] = float(row["est_slg"])
        except (TypeError, ValueError):
            continue
    return out
