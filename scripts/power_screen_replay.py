#!/usr/bin/env python3
"""Replay past power screens under both scoring rules and grade what each kept.

The screen used to give every metric one point and a second for a top-five
finish; #263 replaced that with a weight equal to the metric's measured
split-half reliability at the hitter's own sample, and stopped an unreadable top
finish from carrying a hitter through the ``no top-K finish`` cut. This script
re-runs the cuts on past slates under both rules and grades the survivors
against the box score, so the change can be judged on the record instead of on
the argument for it.

Two things it is not. It is a **posted-lineup replay**: the morning screen fills
unposted orders from Rotowire projections, which are not archived, so the pool
here is built from the order that actually batted. And it replays the *selection*
only -- the arsenal, exposure and simulation stages are untouched by #263's
scoring change, so they are skipped.

    python scripts/power_screen_replay.py --start 2026-08-01 --end 2026-08-21
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.results import fetch_result
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.output.power_screen import (
    METRIC,
    MIN_BATTER_PA,
    MIN_WRC,
    SCORED,
    TOP_K,
    HitterLine,
    PoolBatter,
    apply_cuts,
    batter_window_line,
    hitter_pool,
    rank_starters,
    score_pool,
    starter_damage,
)

log = logging.getLogger("power_screen_replay")

#: What a night is worth to a bat, read off the box score.
MARKETS = ("H", "TB", "XBH", "HR", "R", "RBI")


def flat_score(pool: list[HitterLine]) -> None:
    """The screen's retired scoring: a point a metric, two for a top-five finish.

    Kept here rather than in the engine because nothing should score this way
    again -- it exists so the replay has something to compare against.
    """
    for h in pool:
        h.points = 0
        h.score = 0.0
        h.top_in = ()
        h.withheld = ()
    for attr, label, higher_better in SCORED:
        read = METRIC[attr]
        ranked = sorted(
            (h for h in pool if not math.isnan(read(h))), key=read, reverse=higher_better
        )
        for i, h in enumerate(ranked):
            h.score += 1.0
            if i < TOP_K:
                h.score += 1.0
                h.top_in = (*h.top_in, label)
    for h in pool:
        h.points = int(round(h.score))


@dataclass
class Pick:
    """One hitter a rule kept on one date, with the night he then had."""

    day: Date
    name: str
    mlbam_id: int
    versus: str
    game_pk: int
    pa_window: int
    rank: int
    withheld: tuple[str, ...]
    result: dict[str, int]

    @property
    def played(self) -> bool:
        return self.result.get("PA", 0) > 0


@dataclass
class Record:
    """The pooled record of every pick a rule made."""

    picks: int = 0
    pa: int = 0
    hit_1plus: int = 0
    tb_2plus: int = 0
    totals: dict[str, int] = field(default_factory=lambda: dict.fromkeys(MARKETS, 0))

    def add(self, pick: Pick) -> None:
        if not pick.played:
            return
        self.picks += 1
        self.pa += pick.result["PA"]
        for market in MARKETS:
            self.totals[market] += pick.result[market]
        self.hit_1plus += 1 if pick.result["H"] >= 1 else 0
        self.tb_2plus += 1 if pick.result["TB"] >= 2 else 0

    def per_pa(self, market: str) -> float:
        return self.totals[market] / self.pa if self.pa else math.nan

    def rate(self, count: int) -> float:
        return count / self.picks if self.picks else math.nan


def _result_line(result_batting: dict[str, int]) -> dict[str, int]:
    """The box-score batting line reduced to the markets the screen sells."""
    singles = result_batting.get("1B", 0)
    doubles = result_batting.get("2B", 0)
    triples = result_batting.get("3B", 0)
    hr = result_batting.get("HR", 0)
    return {
        "PA": result_batting.get("PA", 0),
        "H": result_batting.get("H", 0),
        "TB": singles + 2 * doubles + 3 * triples + 4 * hr,
        "XBH": doubles + triples + hr,
        "HR": hr,
        "R": result_batting.get("R", 0),
        "RBI": result_batting.get("RBI", 0),
    }


def _league_woba(window: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for hand in ("R", "L"):
        rows = window[window["p_throws"] == hand] if "p_throws" in window else window
        line = batter_window_line(rows)
        if line:
            out[hand] = line["woba"]
    return out


def _league_xwoba(window: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for hand in ("R", "L"):
        rows = window[window["p_throws"] == hand] if "p_throws" in window else window
        line = batter_window_line(rows)
        if line:
            out[hand] = line["xwoba_pa"]
    return out


def _replay_day(
    day: Date,
    *,
    stats: MLBStatsClient,
    repo: StatcastRepository,
    cache_dir: Path,
    form: int,
    window_days: list[int],
    arms: int,
    keep: int,
    min_pa: int,
    min_wrc: float,
    refresh: bool,
) -> tuple[list[Pick], list[Pick]]:
    """(picks the old rule made, picks the new rule made) for one slate."""
    slate = stats.get_slate(day)
    if not slate.games:
        return [], []
    frame = repo.max_window(day, window_days, refresh=refresh)
    end = day - timedelta(days=1)
    start = end - timedelta(days=form - 1)
    window = frame[(frame["game_date"] >= start) & (frame["game_date"] <= end)]
    if window.empty:
        log.warning("%s: no window rows, skipping", day)
        return [], []
    lg_woba = _league_woba(window)
    lg_xwoba = _league_xwoba(window)

    cards = []
    context = {}
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
    targets = rank_starters(cards, top_n=max(arms, keep))[:keep]

    old_picks: list[Pick] = []
    new_picks: list[Pick] = []
    for card in targets:
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
            log.warning("%s: no posted order for %s", day, lineup_team.abbrev)
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
        floor = lg_xwoba.get(card.throws, lg_xwoba.get("R", 0.305))
        try:
            result = fetch_result(game.game_pk, cache_dir=cache_dir)
        except RuntimeError as exc:
            log.warning("%s: no box score for %s (%s)", day, game.game_pk, exc)
            continue
        for scorer, bucket in ((flat_score, old_picks), (score_pool, new_picks)):
            candidates = copy.deepcopy(pool)
            kept = apply_cuts(
                candidates, floor, min_pa=min_pa, min_wrc=min_wrc, scorer=scorer
            )
            for rank, h in enumerate(kept, 1):
                bucket.append(
                    Pick(
                        day=day,
                        name=h.name,
                        mlbam_id=h.mlbam_id,
                        versus=card.name,
                        game_pk=game.game_pk,
                        pa_window=h.pa,
                        rank=rank,
                        withheld=h.withheld,
                        result=_result_line(result.batter(h.mlbam_id)),
                    )
                )
    return old_picks, new_picks


def _print_day(day: Date, old: list[Pick], new: list[Pick]) -> None:
    old_ids = {p.mlbam_id for p in old}
    new_ids = {p.mlbam_id for p in new}
    dropped = [p for p in old if p.mlbam_id not in new_ids]
    added = [p for p in new if p.mlbam_id not in old_ids]
    print(f"\n{day}  old kept {len(old)}, new kept {len(new)}")
    for p in sorted(new, key=lambda q: q.rank):
        if p.mlbam_id in old_ids:
            print(
                f"  = {p.name:<22} {p.versus:<20} {p.pa_window:>3} PA  new #{p.rank}"
                f"  -> {p.result['H']}H {p.result['TB']}TB"
            )
    for p in dropped:
        withheld = ", ".join(p.withheld) or "-"
        print(
            f"  - {p.name:<22} {p.versus:<20} {p.pa_window:>3} PA  dropped"
            f"  -> {p.result['H']}H {p.result['TB']}TB   unreadable: {withheld}"
        )
    for p in added:
        print(
            f"  + {p.name:<22} {p.versus:<20} {p.pa_window:>3} PA  added #{p.rank}"
            f"  -> {p.result['H']}H {p.result['TB']}TB"
        )


def _print_record(label: str, record: Record) -> None:
    if not record.picks:
        print(f"{label:<28} no graded picks")
        return
    print(
        f"{label:<28} {record.picks:>4} picks {record.pa:>5} PA | "
        f"H/PA {record.per_pa('H'):.3f}  TB/PA {record.per_pa('TB'):.3f}  "
        f"XBH/PA {record.per_pa('XBH'):.3f}  HR/PA {record.per_pa('HR'):.3f} | "
        f"1+H {record.rate(record.hit_1plus):.3f}  2+TB {record.rate(record.tb_2plus):.3f}"
    )


def _write_csv(path: Path, rows: list[tuple[str, Pick]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["rule", "date", "batter", "player_id", "versus", "game_pk", "window_pa",
             "rank", "withheld", *MARKETS, "PA"]
        )
        for rule, p in rows:
            writer.writerow(
                [rule, p.day.isoformat(), p.name, p.mlbam_id, p.versus, p.game_pk,
                 p.pa_window, p.rank, "|".join(p.withheld),
                 *[p.result[m] for m in MARKETS], p.result["PA"]]
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=Date.fromisoformat, required=True)
    p.add_argument("--end", type=Date.fromisoformat, required=True)
    p.add_argument("--arms", type=int, default=6, help="starters ranked per slate")
    p.add_argument("--keep", type=int, default=3, help="starters screened per slate")
    p.add_argument("--min-pa", type=int, default=MIN_BATTER_PA)
    p.add_argument("--min-wrc", type=float, default=MIN_WRC)
    p.add_argument("--refresh", action="store_true", help="re-download each Statcast window")
    p.add_argument("--csv", type=Path, default=None, help="write every pick to this file")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
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

    old_all: list[Pick] = []
    new_all: list[Pick] = []
    day = args.start
    while day <= args.end:
        try:
            old, new = _replay_day(
                day,
                stats=stats,
                repo=repo,
                cache_dir=cfg.cache_dir,
                form=form,
                window_days=window_days,
                arms=args.arms,
                keep=args.keep,
                min_pa=args.min_pa,
                min_wrc=args.min_wrc,
                refresh=args.refresh,
            )
        except Exception as exc:  # a bad date must not cost the rest of the range
            log.warning("%s: replay failed (%s)", day, exc)
            day += timedelta(days=1)
            continue
        if old or new:
            _print_day(day, old, new)
        old_all.extend(old)
        new_all.extend(new)
        day += timedelta(days=1)

    old_ids = defaultdict(set)
    new_ids = defaultdict(set)
    for p in old_all:
        old_ids[p.day].add(p.mlbam_id)
    for p in new_all:
        new_ids[p.day].add(p.mlbam_id)

    both = Record()
    only_old = Record()
    only_new = Record()
    for p in old_all:
        (both if p.mlbam_id in new_ids[p.day] else only_old).add(p)
    for p in new_all:
        if p.mlbam_id not in old_ids[p.day]:
            only_new.add(p)

    print(f"\n=== {args.start} .. {args.end}: posted-lineup replay ===")
    _print_record("old rule, all picks", _pooled(old_all))
    _print_record("new rule, all picks", _pooled(new_all))
    _print_record("kept by both", both)
    _print_record("dropped by the new rule", only_old)
    _print_record("added by the new rule", only_new)
    withheld_counts: dict[str, int] = defaultdict(int)
    for p in new_all + [q for q in old_all if q.mlbam_id not in new_ids[q.day]]:
        for metric in p.withheld:
            withheld_counts[metric] += 1
    if withheld_counts:
        print("\nmetrics withheld as unreadable (top-five finishes that no longer vote):")
        for metric, count in sorted(withheld_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {metric:<12} {count}")
    print(
        "\nEvery number above is a posted-lineup replay of the selection stage only, "
        "on a sample this size; it is a diagnostic, not a validated edge."
    )
    if args.csv:
        rows = [("old", p) for p in old_all] + [("new", p) for p in new_all]
        _write_csv(args.csv, rows)
        print(f"wrote {len(rows)} picks to {args.csv}")


def _pooled(picks: list[Pick]) -> Record:
    record = Record()
    for p in picks:
        record.add(p)
    return record


if __name__ == "__main__":
    main()
