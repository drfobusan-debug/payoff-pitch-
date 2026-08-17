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

A subscriber's projection beats the Marcel, so anything dropped in the
projections folder is read first and the Marcel covers the hitters it omits --
usually the bench and the callups, who are exactly the players a paid projection
leaves out and the engine still has to price. The file is picked up the moment it
appears rather than on the weekly clock, since it is refreshed by hand.
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
MIN_EXPORT_PA = 25.0


def newest_export(folder: Path | None, source: str = "") -> Path | None:
    """The projection CSV in ``folder`` to price off, if any.

    ``source`` names the preferred system and is matched against the file name,
    so a folder holding every system's export resolves to one file rather than to
    whichever download finished last. The match is required rather than
    preferred, because this folder is often the browser's download folder: with
    no source named, "the newest CSV here" is a bank statement away from becoming
    the batter prior. Naming a system that is not present is a warning and the
    Marcel, not a guess.

    An empty ``source`` does mean the newest CSV, which is only safe in a folder
    kept for projections.
    """
    if folder is None or not folder.is_dir():
        return None
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]
    if not files:
        return None
    if not source:
        return max(files, key=lambda f: f.stat().st_mtime)
    wanted = [f for f in files if source.lower() in f.name.lower()]
    if not wanted:
        log.warning(
            "no projection export naming %r in %s (%d CSVs there); using the Marcel",
            source,
            folder,
            len(files),
        )
        return None
    return max(wanted, key=lambda f: f.stat().st_mtime)


def _from_export(path: Path) -> pd.DataFrame | None:
    """Rate vectors from a projection export, or None if it cannot be read.

    A malformed or ID-less export is a warning, not an error: the Marcel behind
    it is a complete projection on its own, so the slate prices either way.

    Rows under ``MIN_EXPORT_PA`` are dropped because some exports round their
    counting stats to integers, and a rate read off two projected plate
    appearances is rounding error rather than a projection -- ATC's 54 hitters
    at 1-3 PA all come out with no hits, no walks and a .000 wOBA, which would
    price a callup as an automatic out. Above the cut the same export is sane
    (spread across hitters .0282 of projected wOBA against .0973 uncut, and
    within .001 of a system that exports fractions), and the Marcel covers
    everyone dropped.
    """
    try:
        rows = pd.read_csv(path)
        if "PA" in rows.columns:
            rows = rows[pd.to_numeric(rows["PA"], errors="coerce") >= MIN_EXPORT_PA]
        return ros_rates_from_projection(rows)
    except (OSError, ValueError, KeyError) as exc:
        log.warning("could not read the projection export %s (%s); using the Marcel", path, exc)
        return None


def is_stale(
    path: Path,
    today: Date,
    max_age_days: int = MAX_AGE_DAYS,
    projections: Path | None = None,
    source: str = "",
) -> bool:
    if not path.exists():
        return True
    export = newest_export(projections, source)
    if export is not None and export.stat().st_mtime > path.stat().st_mtime:
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
    return mtime < today - timedelta(days=max_age_days)


def _marcel(client: MLBStatsClient, season: int, min_pa: float) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for back in range(SEASONS):
        got = client.season_hitting(season - back)
        log.info("%d season lines: %d hitters", season - back, len(got))
        rows.extend(got)
    if not rows:
        raise RuntimeError(f"no season lines returned for {season} and the two before it")
    lines = pd.DataFrame(rows)
    ages = client.player_ages({int(i) for i in lines["mlbam_id"]})
    return ros_rates_from_projection(
        marcel_projection(lines, ages, season, min_weighted_pa=min_pa)
    )


def build(
    client: MLBStatsClient,
    season: int,
    out: Path,
    min_pa: float = 100.0,
    projections: Path | None = None,
    source: str = "",
) -> pd.DataFrame:
    """Write the projection for ``season`` to ``out`` and return it.

    A dropped-in export takes precedence hitter by hitter, not wholesale: it is
    the better estimate for the players it lists, and the Marcel is the only
    estimate for the ones it does not.
    """
    export = newest_export(projections, source)
    dropped = _from_export(export) if export is not None else None
    try:
        ros = _marcel(client, season, min_pa)
    except Exception:
        if dropped is None or dropped.empty:
            raise
        log.warning("no season lines for the Marcel; pricing off %s alone", export)
        ros = dropped
    else:
        if dropped is not None and not dropped.empty:
            filled = ros[~ros["mlbam_id"].isin(dropped["mlbam_id"])]
            log.info(
                "projection export %s: %d hitters, %d more from the Marcel",
                export,
                len(dropped),
                len(filled),
            )
            ros = pd.concat([dropped, filled], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    ros.to_csv(out, index=False)
    return ros


def refresh_if_stale(
    path: str | Path | None,
    today: Date,
    client: MLBStatsClient | None = None,
    projections: Path | None = None,
    source: str = "",
) -> None:
    """Rebuild the projection when it is missing or a week old.

    Never raises: a slate that cannot reach the API still prices, on the league
    mean, and the warning says which hitters that affects (all of them).
    """
    if path is None:
        return
    out = Path(path)
    if not is_stale(out, today, projections=projections, source=source):
        return
    try:
        ros = build(
            client or MLBStatsClient(),
            today.year,
            out,
            projections=projections,
            source=source,
        )
    except Exception as exc:  # noqa: BLE001 -- a stale prior must not stop a slate
        log.warning(
            "could not rebuild the hitter projection at %s (%s); every batter "
            "falls back to the league prior for this slate",
            out,
            exc,
        )
        return
    log.info("hitter projection rebuilt: %d hitters -> %s", len(ros), out)
