"""Public Baseball Savant expected-statistics feed (xSLG / xwOBA / xERA).

The Statcast expected-statistics leaderboard is a free, no-login CSV endpoint on
the same host the engine already uses. It exposes ``est_slg`` (xSLG) keyed by
``player_id`` (an MLBAM id), so the batter xSLG distribution-tail hook activates
without a FanGraphs subscription. xSLG is a season-to-date stabilizing skill
metric (the leaderboard's native granularity), folded into the tail z-composite
alongside the rolling-window Statcast metrics.

The pitcher side of the same board carries ``xera``, which is Savant's own
number rather than anything the engine reconstructs, and the ``pa`` behind it so
a thin line can be refused rather than trusted. Only qualified pitchers appear,
which is the point: an arm the board has never listed has not thrown enough for
an expected-run figure to mean anything.

Returns ``{}`` on any failure so the layer that asked stays neutral.
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from mlb_engine.data import http

log = logging.getLogger(__name__)

_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
    "?type={side}&year={year}&position=&team=&min=q&csv=true"
)
_HEADERS = {"User-Agent": "Mozilla/5.0 (mlb-prediction-engine)"}


def load_batter_xslg(year: int, timeout: int = 30) -> dict[int, float]:
    """Return ``{mlbam_id: est_slg}`` for qualified batters (``{}`` on failure)."""
    df = _board("batter", year, timeout)
    if df is None:
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


def load_pitcher_xera(year: int, timeout: int = 30) -> dict[int, tuple[float, int]]:
    """Return ``{mlbam_id: (xera, pa)}`` for qualified pitchers (``{}`` on failure).

    The plate appearances come back with the number so the caller can hold the
    figure to its own sample floor instead of trusting a short line, and an arm
    the board omits reads as *no xERA* rather than as an average one.
    """
    df = _board("pitcher", year, timeout)
    if df is None:
        return {}

    if not {"player_id", "xera", "pa"}.issubset(df.columns):
        log.warning("Savant pitcher xERA columns missing (have %s)", list(df.columns)[:8])
        return {}

    out: dict[int, tuple[float, int]] = {}
    for _, row in df.iterrows():
        try:
            out[int(row["player_id"])] = (float(row["xera"]), int(row["pa"]))
        except (TypeError, ValueError):
            continue
    return out


def _board(side: str, year: int, timeout: int) -> pd.DataFrame | None:
    try:
        resp = http.get(_URL.format(side=side, year=year), headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:  # optional enrichment
        log.warning("Savant expected-stats (%s) unavailable: %s", side, exc)
        return None
