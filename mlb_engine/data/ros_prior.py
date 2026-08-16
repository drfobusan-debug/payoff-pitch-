"""Keep the hitter projection on disk fresh, wherever the engine is running.

The prior file is not part of the synced state -- it is derived, not recorded --
so a machine that has never built it has no file, and the batter prior silently
falls back to the league mean. That is the one failure mode worth engineering
against here: the engine would keep pricing, and nothing in the output would say
that every hitter had just been handed the league line.

So the slate run refreshes it, and the refresh is cheap enough to belong there:
three season lines and one age lookup off the official API, a few seconds. It is
also genuinely needed on a schedule rather than once -- the current season is the
heaviest of Marcel's three and grows every night -- and ``MAX_AGE_DAYS`` is set
to a week because that is how long it takes a full slate week to move a rate
enough to matter.

A failure here is not a failure of the slate: the engine falls back to the league
mean, which is what it did before the projection existed, and says so loudly.
"""

from __future__ import annotations

import logging
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.features.marcel import marcel_projection
from mlb_engine.features.rolling import ros_rates_from_projection

log = logging.getLogger(__name__)

MAX_AGE_DAYS = 7
SEASONS = 3


def is_stale(path: Path, today: Date, max_age_days: int = MAX_AGE_DAYS) -> bool:
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
    return mtime < today - timedelta(days=max_age_days)


def build(
    client: MLBStatsClient, season: int, out: Path, min_pa: float = 100.0
) -> pd.DataFrame:
    """Write the projection for ``season`` to ``out`` and return it."""
    rows: list[dict[str, int]] = []
    for back in range(SEASONS):
        got = client.season_hitting(season - back)
        log.info("%d season lines: %d hitters", season - back, len(got))
        rows.extend(got)
    if not rows:
        raise RuntimeError(f"no season lines returned for {season} and the two before it")
    lines = pd.DataFrame(rows)
    ages = client.player_ages({int(i) for i in lines["mlbam_id"]})
    ros = ros_rates_from_projection(
        marcel_projection(lines, ages, season, min_weighted_pa=min_pa)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    ros.to_csv(out, index=False)
    return ros


def refresh_if_stale(
    path: str | Path | None, today: Date, client: MLBStatsClient | None = None
) -> None:
    """Rebuild the projection when it is missing or a week old.

    Never raises: a slate that cannot reach the API still prices, on the league
    mean, and the warning says which hitters that affects (all of them).
    """
    if path is None:
        return
    out = Path(path)
    if not is_stale(out, today):
        return
    try:
        ros = build(client or MLBStatsClient(), today.year, out)
    except Exception as exc:  # noqa: BLE001 -- a stale prior must not stop a slate
        log.warning(
            "could not rebuild the hitter projection at %s (%s); every batter "
            "falls back to the league prior for this slate",
            out,
            exc,
        )
        return
    log.info("hitter projection rebuilt: %d hitters -> %s", len(ros), out)
