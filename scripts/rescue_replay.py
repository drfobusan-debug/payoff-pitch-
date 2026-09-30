#!/usr/bin/env python3
"""Replay the swing rescue over past slates and grade every row against the box score.

#280 added a stage-two rescue: a hitter the luck-gap cut wants dropped is kept
when his bat speed and blast rate are above ``RESCUE_POWER_Z``. That constant
shipped at league average and had never been graded, so this walks real slates,
re-runs the screen's own cuts on the order that actually batted, and records what
each hitter did that night in four buckets:

* ``kept`` -- survived the cuts on his own, no rescue involved;
* ``rescued`` -- the luck gap flagged him and the swing kept him anyway;
* ``still_cut`` -- the luck gap flagged him and the swing did not save him;
* ``cut_other`` -- removed by one of the other cuts, kept here for the base rate.

A rescue only earns its constant if ``rescued`` hits like ``kept`` or better while
``still_cut`` lags -- otherwise the threshold is admitting the wrong half. The
``--power-z`` flag replays alternative thresholds, so the number can be read off
the record rather than assumed.

    python scripts/rescue_replay.py --start 2026-07-01 --end 2026-08-22 \
        --power-z 0.0 --power-z 0.375 --keep 6

Two limits, the same ones the #264 replay carries. It is a **posted-lineup**
replay -- the morning screen fills unposted orders from Rotowire projections,
which are not archived -- and it replays the selection stage only. A month of
slates is a few hundred plate appearances, which is a check on the panel's
threshold and not a substitute for it.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.results import fetch_result
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.features.swing import CONTRADICTED, stage_two
from mlb_engine.output.power_screen import (
    MAX_LUCK_GAP,
    MIN_BATTER_PA,
    MIN_WRC,
    RESCUE_POWER_Z,
    HitterLine,
    PoolBatter,
    apply_cuts,
    batter_window_line,
    hitter_pool,
    rank_starters,
    starter_damage,
)

log = logging.getLogger("rescue_replay")

MARKETS = ("H", "TB", "XBH", "HR", "R", "RBI")
BUCKETS = ("kept", "rescued", "still_cut", "cut_other")


@dataclass
class Row:
    """One hitter on one slate, the bucket he fell in, and the night he had."""

    day: Date
    name: str
    mlbam_id: int
    versus: str
    bucket: str
    luck_gap: float
    power_z: float
    pa_window: int
    result: dict[str, int]

    @property
    def played(self) -> bool:
        return self.result.get("PA", 0) > 0


@dataclass
class Record:
    rows: int = 0
    pa: int = 0
    totals: dict[str, int] = field(default_factory=lambda: dict.fromkeys(MARKETS, 0))
    tb_2plus: int = 0
    hit_1plus: int = 0

    def add(self, row: Row) -> None:
        if not row.played:
            return
        self.rows += 1
        self.pa += row.result["PA"]
        for market in MARKETS:
            self.totals[market] += row.result[market]
        self.tb_2plus += 1 if row.result["TB"] >= 2 else 0
        self.hit_1plus += 1 if row.result["H"] >= 1 else 0

    def per_pa(self, market: str) -> float:
        return self.totals[market] / self.pa if self.pa else math.nan

    def rate(self, count: int) -> float:
        return count / self.rows if self.rows else math.nan


def _result_line(batting: dict[str, int]) -> dict[str, int]:
    singles, doubles = batting.get("1B", 0), batting.get("2B", 0)
    triples, hr = batting.get("3B", 0), batting.get("HR", 0)
    return {
        "PA": batting.get("PA", 0),
        "H": batting.get("H", 0),
        "TB": singles + 2 * doubles + 3 * triples + 4 * hr,
        "XBH": doubles + triples + hr,
        "HR": hr,
        "R": batting.get("R", 0),
        "RBI": batting.get("RBI", 0),
    }


def _league(window: pd.DataFrame, key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for hand in ("R", "L"):
        rows = window[window["p_throws"] == hand] if "p_throws" in window else window
        line = batter_window_line(rows)
        if line:
            out[hand] = line[key]
    return out


def _bucket(h: HitterLine, min_power_z: float) -> str:
    """Which bucket a pooled hitter falls in at this rescue threshold."""
    if h.kept:
        return "kept"
    if h.cut_reason.startswith("wOBA outruns"):
        verdict = stage_two(h.luck_gap, h.swing, min_power_z=min_power_z)
        return "rescued" if verdict == CONTRADICTED else "still_cut"
    return "cut_other"


def _replay_day(
    day: Date,
    *,
    stats: MLBStatsClient,
    repo: StatcastRepository,
    cache_dir: Path,
    form: int,
    window_days: list[int],
    keep: int,
    min_pa: int,
    min_wrc: float,
    thresholds: list[float],
) -> dict[float, list[Row]]:
    out: dict[float, list[Row]] = {t: [] for t in thresholds}
    slate = stats.get_slate(day)
    if not slate.games:
        return out
    frame = repo.max_window(day, window_days)
    end = day - timedelta(days=1)
    window = frame[
        (frame["game_date"] >= end - timedelta(days=form - 1)) & (frame["game_date"] <= end)
    ]
    if window.empty:
        log.warning("%s: no window rows", day)
        return out
    lg_woba, lg_xwoba = _league(window, "woba"), _league(window, "xwoba_pa")

    cards, context = [], {}
    for game in slate.games:
        for team, opp in ((game.home, game.away), (game.away, game.home)):
            pitcher = team.probable_pitcher
            if pitcher is None or not pitcher.mlbam_id:
                continue
            throws = getattr(pitcher.throws, "value", pitcher.throws) or "R"
            cards.append(
                starter_damage(
                    window[window["pitcher"] == pitcher.mlbam_id],
                    name=pitcher.name,
                    mlbam_id=int(pitcher.mlbam_id),
                    team=team.abbrev,
                    opponent=opp.abbrev,
                    throws=str(throws),
                )
            )
            context[int(pitcher.mlbam_id)] = (opp, game)

    for card in rank_starters(cards, top_n=keep):
        lineup_team, game = context[card.mlbam_id]
        batters = [
            PoolBatter(
                mlbam_id=int(slot.player.mlbam_id),
                name=slot.player.name,
                slot=slot.order,
                bats=getattr(slot.player.bats, "value", slot.player.bats),
            )
            for slot in (lineup_team.lineup or [])
            if slot.player.mlbam_id
        ]
        if not batters:
            continue
        pool = hitter_pool(
            window,
            batters,
            hand=card.throws,
            team=lineup_team.abbrev,
            versus=card.name,
            league_woba=lg_woba.get(card.throws, lg_woba.get("R", 0.315)),
        )
        if not pool:
            continue
        try:
            result = fetch_result(game.game_pk, cache_dir=cache_dir)
        except RuntimeError as exc:
            log.warning("%s: no box score for %s (%s)", day, game.game_pk, exc)
            continue
        floor = lg_xwoba.get(card.throws, lg_xwoba.get("R", 0.305))
        apply_cuts(pool, floor, min_pa=min_pa, min_wrc=min_wrc)
        for h in pool:
            line = _result_line(result.batter(h.mlbam_id))
            for thr in thresholds:
                out[thr].append(
                    Row(
                        day=day,
                        name=h.name,
                        mlbam_id=h.mlbam_id,
                        versus=card.name,
                        bucket=_bucket(h, thr),
                        luck_gap=h.luck_gap,
                        power_z=h.swing.power_z if h.swing else math.nan,
                        pa_window=h.pa,
                        result=line,
                    )
                )
    return out


def _print(threshold: float, rows: list[Row]) -> None:
    records = {b: Record() for b in BUCKETS}
    for row in rows:
        records[row.bucket].add(row)
    print(f"\n=== rescue threshold power z >= {threshold:+.3f} ===")
    for bucket in BUCKETS:
        rec = records[bucket]
        if not rec.rows:
            print(f"  {bucket:<10} no graded rows")
            continue
        print(
            f"  {bucket:<10} {rec.rows:>4} rows {rec.pa:>5} PA | H/PA {rec.per_pa('H'):.3f}"
            f"  TB/PA {rec.per_pa('TB'):.3f}  XBH/PA {rec.per_pa('XBH'):.3f}"
            f"  HR/PA {rec.per_pa('HR'):.3f} | 1+H {rec.rate(rec.hit_1plus):.3f}"
            f"  2+TB {rec.rate(rec.tb_2plus):.3f}"
        )
    kept, rescued, still = records["kept"], records["rescued"], records["still_cut"]
    if rescued.rows and kept.rows:
        print(
            f"  rescued minus kept: TB/PA {rescued.per_pa('TB') - kept.per_pa('TB'):+.3f}"
            f"  HR/PA {rescued.per_pa('HR') - kept.per_pa('HR'):+.4f}"
        )
    if rescued.rows and still.rows:
        print(
            f"  rescued minus still cut: TB/PA {rescued.per_pa('TB') - still.per_pa('TB'):+.3f}"
            f"  HR/PA {rescued.per_pa('HR') - still.per_pa('HR'):+.4f}"
        )
    named = [r for r in rows if r.bucket == "rescued" and r.played]
    if named:
        print("  every hitter the rescue admitted:")
        for r in sorted(named, key=lambda q: (q.day, q.name)):
            print(
                f"    {r.day} {r.name:<22} vs {r.versus:<20} gap {r.luck_gap:+.3f}"
                f"  power z {r.power_z:+.2f}  -> {r.result['H']}H {r.result['TB']}TB"
                f" {r.result['HR']}HR"
            )


def _write_csv(path: Path, per_threshold: dict[float, list[Row]]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "threshold",
                "date",
                "batter",
                "player_id",
                "versus",
                "bucket",
                "luck_gap",
                "power_z",
                "window_pa",
                *MARKETS,
                "PA",
            ]
        )
        for thr, rows in per_threshold.items():
            for r in rows:
                w.writerow(
                    [
                        f"{thr:+.3f}",
                        r.day.isoformat(),
                        r.name,
                        r.mlbam_id,
                        r.versus,
                        r.bucket,
                        f"{r.luck_gap:+.4f}",
                        f"{r.power_z:+.3f}",
                        r.pa_window,
                        *[r.result[m] for m in MARKETS],
                        r.result["PA"],
                    ]
                )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=Date.fromisoformat, required=True)
    p.add_argument("--end", type=Date.fromisoformat, required=True)
    p.add_argument("--keep", type=int, default=6, help="starters screened per slate")
    p.add_argument("--min-pa", type=int, default=MIN_BATTER_PA)
    p.add_argument("--min-wrc", type=float, default=MIN_WRC)
    p.add_argument(
        "--power-z",
        type=float,
        action="append",
        default=None,
        help="rescue threshold to replay; repeatable (default: the shipped constant)",
    )
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    thresholds = args.power_z or [RESCUE_POWER_Z]
    cfg = load_config()
    form = cfg.windows.pitcher_form_days
    window_days = [
        form,
        cfg.windows.batter_vs_rhp_days,
        cfg.windows.batter_vs_lhp_days,
        cfg.windows.bullpen_skill_days,
    ]
    stats = MLBStatsClient()
    repo = StatcastRepository(cfg.cache_dir)

    per_threshold: dict[float, list[Row]] = {t: [] for t in thresholds}
    day = args.start
    while day <= args.end:
        try:
            got = _replay_day(
                day,
                stats=stats,
                repo=repo,
                cache_dir=cfg.cache_dir,
                form=form,
                window_days=window_days,
                keep=args.keep,
                min_pa=args.min_pa,
                min_wrc=args.min_wrc,
                thresholds=thresholds,
            )
        except Exception as exc:  # one bad date must not cost the range
            log.warning("%s: replay failed (%s)", day, exc)
            day += timedelta(days=1)
            continue
        for thr, rows in got.items():
            per_threshold[thr].extend(rows)
        day += timedelta(days=1)

    print(
        f"=== {args.start} .. {args.end}: posted-lineup rescue replay, {args.keep} arms a slate ==="
    )
    print(
        f"the luck-gap cut fires above {MAX_LUCK_GAP:+.3f}; the shipped rescue is {RESCUE_POWER_Z:+.3f}"
    )
    for thr in thresholds:
        _print(thr, per_threshold[thr])
    if args.csv:
        _write_csv(args.csv, per_threshold)
        print(f"\nwrote {args.csv}")
    print(
        "\nPosted-lineup replay of the selection stage only, on a month of slates:"
        " a check on the panel's threshold, not a validated edge."
    )


if __name__ == "__main__":
    main()
