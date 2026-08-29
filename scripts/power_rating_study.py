#!/usr/bin/env python3
"""Grade the power screen's matchup rating: does its score sort the night?

The note prints a BUY/HOLD/AVOID (displayed as MATCHUP A/B/C since #261) built
from five hand-set indicators -- share of the game against the screened starter,
arsenal fit, the exposure-weighted opponent, contact quality, strikeout risk --
summed into a score, with the label falling out of two hand-set cuts at 3 and 1.
No part of that has ever been fitted, and the ledger it has (114 positions) is
far too small to grade it: over those rows BUY ran *behind* HOLD.

This builds the panel the ledger cannot. For every slate in a range it screens
every starter on the board rather than the four the note keeps, assembles each
survivor exactly as the note does -- same pool, same cuts, same arsenal fit,
same exposure model, same bullpen profile -- reads the label off the same
function the note calls, and grades the night off the box score.

Two limits it does not hide. It is a **posted-lineup panel**: the morning note
fills unposted orders from Rotowire projections, which are not archived, so a
slate contributes only the orders the feed carries. And it grades the *rating*
only: the price the note quotes beside it comes from the card's board, which is
archived for a handful of days, so this says whether the matchup grade sorts
production, not whether it sorts money.

    python scripts/power_rating_study.py --start 2026-05-01 --end 2026-08-22 \
        --csv /home/ubuntu/rating_panel.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pandas as pd

from mlb_engine.config import Config, load_config
from mlb_engine.data.managers import DEFAULT_BF_CAP
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.results import fetch_result
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.features.efficiency import (
    build_pitcher_efficiency,
    opponent_discipline_factor,
)
from mlb_engine.features.rolling import build_bullpen_profile
from mlb_engine.features.workload import _bf_per_start, expected_bf_cap
from mlb_engine.output.power_report import _rating
from mlb_engine.output.power_screen import (
    MIN_BATTER_PA,
    MIN_WRC,
    BullpenCard,
    HitterView,
    PoolBatter,
    apply_cuts,
    arsenal,
    arsenal_fit,
    batter_arsenal,
    batter_window_line,
    bf_pmf,
    contact_line,
    exposure,
    hitter_pool,
    pa_vs_starter,
    rank_starters,
    starter_damage,
)

log = logging.getLogger("power_rating_study")

FALLBACK_TEAM_PA = 38.6
TEAM_PA_SD = 4.0
MARKETS = ("H", "TB", "XBH", "HR", "R", "RBI", "HRR")
RATINGS = ("BUY", "HOLD", "AVOID")


@dataclass
class Row:
    """One rated survivor on one date, and the night he then had."""

    day: Date
    name: str
    mlbam_id: int
    versus: str
    game_pk: int
    slot: int
    rating: str
    score: int
    share: float
    fit_delta: float
    opp_xwoba: float
    xwoba_con: float
    k_pct: float
    wrc: float
    points: int
    result: dict[str, int]

    @property
    def pa(self) -> int:
        return self.result.get("PA", 0)


def _result_line(batting: dict[str, int]) -> dict[str, int]:
    singles = batting.get("1B", 0)
    doubles = batting.get("2B", 0)
    triples = batting.get("3B", 0)
    hr = batting.get("HR", 0)
    hits = batting.get("H", 0)
    return {
        "PA": batting.get("PA", 0),
        "H": hits,
        "TB": singles + 2 * doubles + 3 * triples + 4 * hr,
        "XBH": doubles + triples + hr,
        "HR": hr,
        "R": batting.get("R", 0),
        "RBI": batting.get("RBI", 0),
        "HRR": hits + batting.get("R", 0) + batting.get("RBI", 0),
    }


def _league(window: pd.DataFrame, key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for hand in ("R", "L"):
        rows = window[window["p_throws"] == hand] if "p_throws" in window else window
        line = batter_window_line(rows)
        if line:
            out[hand] = line[key]
    return out


def _team_pa_per_game(window: pd.DataFrame) -> float:
    keys = ["game_date", "home_team", "away_team"]
    if not all(k in window for k in keys) or "events" not in window:
        return FALLBACK_TEAM_PA
    pa_rows = window[window["events"].notna()]
    if pa_rows.empty:
        return FALLBACK_TEAM_PA
    per_game = pa_rows.groupby(keys).size()
    if per_game.empty:
        return FALLBACK_TEAM_PA
    return float(per_game.mean()) / 2.0


def _projected_pa(slot: int, team_pa: float) -> float:
    return pa_vs_starter(slot, bf_pmf(team_pa, TEAM_PA_SD, cap=60, limit=60))


def _bullpen_cards(
    frame: pd.DataFrame, teams: list[str], cfg: Config, as_of: Date
) -> dict[str, BullpenCard]:
    w = cfg.windows
    profiles = []
    for team in teams:
        profile = build_bullpen_profile(
            frame,
            team,
            as_of,
            w.bullpen_days,
            w.bullpen_min_inning,
            skill_days=w.bullpen_skill_days,
            xwoba_shrink=w.bullpen_xwoba_shrink,
        )
        if profile.xwoba_allowed is None:
            continue
        profiles.append((team, profile))
    profiles.sort(key=lambda t: -(t[1].xwoba_allowed or 0.0))
    out: dict[str, BullpenCard] = {}
    for rank, (team, profile) in enumerate(profiles, 1):
        allowed = profile.allowed
        out[team] = BullpenCard(
            team=team,
            rank=rank,
            of_n=len(profiles),
            relief_pa=int(allowed.pa),
            xwoba=float(profile.xwoba_allowed or math.nan),
            k_pct=allowed.p_k,
            bb_pct=allowed.p_bb,
            hr_pct=allowed.p_hr,
            zone_pct=profile.zone_pct,
            late_k_pct=profile.allowed_leverage.p_k,
        )
    return out


def _study_day(
    day: Date,
    *,
    stats: MLBStatsClient,
    repo: StatcastRepository,
    cfg: Config,
    keep: int,
    min_pa: int,
    min_wrc: float,
) -> list[Row]:
    """Every rated survivor on one slate, graded off the box score."""
    slate = stats.get_slate(day)
    if not slate.games:
        return []
    form = cfg.windows.pitcher_form_days
    frame = repo.max_window(
        day,
        [
            form,
            cfg.windows.batter_vs_rhp_days,
            cfg.windows.batter_vs_lhp_days,
            cfg.windows.bullpen_skill_days,
        ],
    )
    end = day - timedelta(days=1)
    start = end - timedelta(days=form - 1)
    window = frame[(frame["game_date"] >= start) & (frame["game_date"] <= end)]
    if window.empty:
        log.warning("%s: no window rows", day)
        return []
    lg_woba = _league(window, "woba")
    lg_xwoba = _league(window, "xwoba_pa")
    team_pa = _team_pa_per_game(window)

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
            context[int(pitcher.mlbam_id)] = (opp, team, game)
    targets = rank_starters(cards, top_n=keep)[:keep]
    if not targets:
        return []

    pens = _bullpen_cards(
        frame,
        sorted({t.abbrev for g in slate.games for t in (g.home, g.away)}),
        cfg,
        day,
    )

    rows: list[Row] = []
    for card in targets:
        lineup_team, pen_team, game = context[card.mlbam_id]
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
        hand = card.throws
        pool = hitter_pool(
            window,
            batters,
            hand=hand,
            team=lineup_team.abbrev,
            versus=card.name,
            league_woba=lg_woba.get(hand, lg_woba.get("R", 0.315)),
        )
        if not pool:
            continue
        kept = apply_cuts(
            pool,
            lg_xwoba.get(hand, lg_xwoba.get("R", 0.305)),
            min_pa=min_pa,
            min_wrc=min_wrc,
        )
        if not kept:
            continue
        try:
            result = fetch_result(game.game_pk, cache_dir=cfg.cache_dir)
        except RuntimeError as exc:
            log.warning("%s: no box score for %s (%s)", day, game.game_pk, exc)
            continue

        pitcher_rows = window[window["pitcher"] == card.mlbam_id]
        lines, usage = arsenal(pitcher_rows)
        card.arsenal = lines
        card.usage = usage
        families = sorted(usage, key=lambda k: -usage[k])

        all_rows = frame[frame["pitcher"] == card.mlbam_id]
        eff = build_pitcher_efficiency(all_rows, day, form)
        bf_cap = expected_bf_cap(all_rows, day, form, manager_cap=DEFAULT_BF_CAP)
        disc = opponent_discipline_factor(frame, [h.mlbam_id for h in pool], day, form)
        ppa = eff.blended_pitches_per_pa() / disc
        bf_mean = min(eff.pitch_cap / ppa, bf_cap) if ppa else float(bf_cap)
        log_bf = _bf_per_start(all_rows, day, form)
        spread = pd.Series(log_bf).std(ddof=1) if len(log_bf) > 1 else math.nan
        bf_sd = float(spread) if not pd.isna(spread) else 4.0
        pmf = bf_pmf(bf_mean, bf_sd, bf_cap)

        for h in kept:
            batter_rows = window[
                (window["batter"] == h.mlbam_id) & (window["p_throws"] == hand)
            ]
            per_pitch = batter_arsenal(batter_rows, families)
            overall = contact_line(batter_rows)
            fit_w, fit_b, fallback = arsenal_fit(per_pitch, overall, usage)
            view = HitterView(
                line=h,
                per_pitch=per_pitch,
                overall=overall,
                fit_xwoba=fit_w,
                fit_xba=fit_b,
                fallback_share=fallback,
            )
            if h.slot:
                view.exposure = exposure(
                    h.slot,
                    _projected_pa(h.slot, team_pa),
                    pmf,
                    card.xwobacon,
                    pens[pen_team.abbrev].xwoba if pen_team.abbrev in pens else None,
                )
            rating, _ = _rating(view)
            e = view.exposure
            rows.append(
                Row(
                    day=day,
                    name=h.name,
                    mlbam_id=h.mlbam_id,
                    versus=card.name,
                    game_pk=game.game_pk,
                    slot=h.slot or 0,
                    rating=rating,
                    score=_rating_score(view),
                    share=e.share_vs_starter if e else math.nan,
                    fit_delta=view.fit_delta,
                    opp_xwoba=e.opponent_xwoba if e else math.nan,
                    xwoba_con=h.xwoba_con,
                    k_pct=h.k,
                    wrc=h.wrc,
                    points=h.points,
                    result=_result_line(result.batter(h.mlbam_id)),
                )
            )
    return rows


def _rating_score(view: HitterView) -> int:
    """The integer score behind the label, rebuilt from the same five reads.

    ``_rating`` returns only the label and its prose, and the label collapses
    five indicators into three buckets; the score is what a monotonicity check
    needs, so it is recomputed here from the same cuts. Kept beside the panel
    rather than exported from the report, because nothing in production should
    start depending on the number until it grades out.
    """
    h = view.line
    e = view.exposure
    share = e.share_vs_starter if e else math.nan
    opp = e.opponent_xwoba if e else math.nan
    delta = view.fit_delta
    score = 0
    if not math.isnan(share):
        score += 1 if share >= 0.58 else (-1 if share < 0.45 else 0)
    if not math.isnan(delta):
        score += 1 if delta >= 0.030 else (-1 if delta <= -0.020 else 0)
    if not math.isnan(opp):
        score += 1 if opp >= 0.340 else (-1 if opp <= 0.300 else 0)
    if h.xwoba_con >= 0.440:
        score += 1
    if h.k >= 0.30:
        score -= 1
    return score


# --- reporting ------------------------------------------------------------


def _per_pa(rows: list[Row], market: str) -> float:
    pa = sum(r.pa for r in rows)
    return sum(r.result[market] for r in rows) / pa if pa else math.nan


def _boot_diff(
    left: list[Row], right: list[Row], market: str, *, draws: int = 2000
) -> tuple[float, float, float]:
    """left-minus-right per-PA, with a hitter-clustered bootstrap interval.

    Clustered on the hitter, not the row: the same bat appears on many slates and
    an unclustered interval would count him as that many independent readings.
    """
    point = _per_pa(left, market) - _per_pa(right, market)
    by_id: dict[int, list[Row]] = defaultdict(list)
    for r in left + right:
        by_id[r.mlbam_id].append(r)
    ids = list(by_id)
    left_ids = {id(r) for r in left}
    rng = random.Random(17)
    diffs = []
    for _ in range(draws):
        picks = [by_id[rng.choice(ids)] for _ in ids]
        flat = [r for group in picks for r in group]
        lo = [r for r in flat if id(r) in left_ids]
        hi = [r for r in flat if id(r) not in left_ids]
        if not lo or not hi:
            continue
        diffs.append(_per_pa(lo, market) - _per_pa(hi, market))
    diffs.sort()
    if len(diffs) < 100:
        return point, math.nan, math.nan
    return point, diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def _line(label: str, rows: list[Row]) -> str:
    pa = sum(r.pa for r in rows)
    return (
        f"  {label:<26} n {len(rows):>5}  PA {pa:>6}  "
        f"TB/PA {_per_pa(rows, 'TB'):.4f}  H/PA {_per_pa(rows, 'H'):.4f}  "
        f"HRR/PA {_per_pa(rows, 'HRR'):.4f}  HR/PA {_per_pa(rows, 'HR'):.4f}"
    )


def _report(rows: list[Row]) -> None:
    played = [r for r in rows if r.pa > 0]
    dates = sorted({r.day for r in played})
    print(
        f"\n=== rating panel: {len(played)} rated survivors who batted, "
        f"{len({r.mlbam_id for r in played})} hitters, {len(dates)} slates "
        f"({dates[0]}..{dates[-1]}) ==="
    )

    print("\n--- by label, the note's own three buckets ---")
    for label in RATINGS:
        bucket = [r for r in played if r.rating == label]
        if bucket:
            print(_line(label, bucket))

    print("\n--- by score, which the label collapses ---")
    for score in sorted({r.score for r in played}):
        bucket = [r for r in played if r.score == score]
        if len(bucket) >= 20:
            print(_line(f"score {score:+d}", bucket))

    buys = [r for r in played if r.rating == "BUY"]
    holds = [r for r in played if r.rating == "HOLD"]
    avoids = [r for r in played if r.rating == "AVOID"]
    print("\n--- BUY minus HOLD, hitter-clustered ---")
    for market in ("TB", "H", "HRR", "HR"):
        point, lo, hi = _boot_diff(buys, holds, market)
        print(f"  {market + '/PA':<8} {point:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print("\n--- HOLD minus AVOID, hitter-clustered ---")
    for market in ("TB", "H", "HRR", "HR"):
        point, lo, hi = _boot_diff(holds, avoids, market)
        print(f"  {market + '/PA':<8} {point:+.4f} [{lo:+.4f},{hi:+.4f}]")

    print("\n--- each indicator on its own, TB/PA ---")
    for label, read, cut, high in (
        ("share vs starter", lambda r: r.share, 0.58, True),
        ("arsenal fit", lambda r: r.fit_delta, 0.030, True),
        ("opponent xwOBA", lambda r: r.opp_xwoba, 0.340, True),
        ("xwOBA on contact", lambda r: r.xwoba_con, 0.440, True),
        ("strikeout rate", lambda r: r.k_pct, 0.30, False),
    ):
        on = [r for r in played if not math.isnan(read(r)) and read(r) >= cut]
        off = [r for r in played if not math.isnan(read(r)) and read(r) < cut]
        if len(on) < 20 or len(off) < 20:
            print(f"  {label:<20} too thin ({len(on)} / {len(off)})")
            continue
        point, lo, hi = _boot_diff(on, off, "TB")
        note = "" if high else "   (scored as a negative)"
        print(
            f"  {label:<20} over the cut n {len(on):>5} {_per_pa(on, 'TB'):.4f}  "
            f"under n {len(off):>5} {_per_pa(off, 'TB'):.4f}  "
            f"diff {point:+.4f} [{lo:+.4f},{hi:+.4f}]{note}"
        )

    print("\n--- what the screen's own sort does, for comparison ---")
    ranked = sorted(played, key=lambda r: -r.points)
    half = len(ranked) // 2
    point, lo, hi = _boot_diff(ranked[:half], ranked[half:], "TB")
    print(
        f"  top half by screen points  {_per_pa(ranked[:half], 'TB'):.4f} vs "
        f"{_per_pa(ranked[half:], 'TB'):.4f}   diff {point:+.4f} [{lo:+.4f},{hi:+.4f}]"
    )
    wrc = sorted((r for r in played if not math.isnan(r.wrc)), key=lambda r: -r.wrc)
    half = len(wrc) // 2
    point, lo, hi = _boot_diff(wrc[:half], wrc[half:], "TB")
    print(
        f"  top half by window wRC+    {_per_pa(wrc[:half], 'TB'):.4f} vs "
        f"{_per_pa(wrc[half:], 'TB'):.4f}   diff {point:+.4f} [{lo:+.4f},{hi:+.4f}]"
    )

    print("\n--- season halves, same label split ---")
    mid = dates[len(dates) // 2]
    for label, subset in (
        (f"{dates[0]}..{mid}", [r for r in played if r.day <= mid]),
        (f"{mid}..{dates[-1]}", [r for r in played if r.day > mid]),
    ):
        b = [r for r in subset if r.rating == "BUY"]
        h = [r for r in subset if r.rating == "HOLD"]
        if len(b) >= 20 and len(h) >= 20:
            print(
                f"  {label}  BUY {_per_pa(b, 'TB'):.4f} (n {len(b)})  "
                f"HOLD {_per_pa(h, 'TB'):.4f} (n {len(h)})  "
                f"diff {_per_pa(b, 'TB') - _per_pa(h, 'TB'):+.4f}"
            )
        else:
            print(f"  {label}  too thin ({len(b)} BUY / {len(h)} HOLD)")

    spread = [r.score for r in played]
    print(
        f"\nscore mean {statistics.mean(spread):+.2f}, "
        f"sd {statistics.pstdev(spread):.2f}, "
        f"labels BUY {len(buys)} / HOLD {len(holds)} / AVOID {len(avoids)}"
    )


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "date", "batter", "player_id", "versus", "game_pk", "slot", "rating",
                "score", "share", "fit_delta", "opp_xwoba", "xwoba_con", "k_pct",
                "wrc", "points", *MARKETS, "PA",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.day.isoformat(), r.name, r.mlbam_id, r.versus, r.game_pk, r.slot,
                    r.rating, r.score, f"{r.share:.4f}", f"{r.fit_delta:.4f}",
                    f"{r.opp_xwoba:.4f}", f"{r.xwoba_con:.4f}", f"{r.k_pct:.4f}",
                    f"{r.wrc:.1f}", r.points,
                    *[r.result[m] for m in MARKETS], r.result["PA"],
                ]
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=Date.fromisoformat, required=True)
    p.add_argument("--end", type=Date.fromisoformat, required=True)
    p.add_argument(
        "--keep", type=int, default=20, help="starters screened per slate (all, by default)"
    )
    p.add_argument("--min-pa", type=int, default=MIN_BATTER_PA)
    p.add_argument("--min-wrc", type=float, default=MIN_WRC)
    p.add_argument("--csv", type=Path, default=None, help="write the panel here")
    p.add_argument("--panel", type=Path, default=None, help="read a written panel instead")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _load_panel(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="") as fh:
        for rec in csv.DictReader(fh):
            rows.append(
                Row(
                    day=Date.fromisoformat(rec["date"]),
                    name=rec["batter"],
                    mlbam_id=int(rec["player_id"]),
                    versus=rec["versus"],
                    game_pk=int(rec["game_pk"]),
                    slot=int(rec["slot"]),
                    rating=rec["rating"],
                    score=int(rec["score"]),
                    share=float(rec["share"]),
                    fit_delta=float(rec["fit_delta"]),
                    opp_xwoba=float(rec["opp_xwoba"]),
                    xwoba_con=float(rec["xwoba_con"]),
                    k_pct=float(rec["k_pct"]),
                    wrc=float(rec["wrc"]),
                    points=int(rec["points"]),
                    result={m: int(rec[m]) for m in (*MARKETS, "PA")},
                )
            )
    return rows


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.panel:
        _report(_load_panel(args.panel))
        return

    cfg = load_config()
    stats = MLBStatsClient()
    repo = StatcastRepository(cfg.cache_dir)
    rows: list[Row] = []
    day = args.start
    while day <= args.end:
        try:
            got = _study_day(
                day,
                stats=stats,
                repo=repo,
                cfg=cfg,
                keep=args.keep,
                min_pa=args.min_pa,
                min_wrc=args.min_wrc,
            )
        except Exception as exc:  # a bad slate must not cost the range
            log.warning("%s: failed (%s)", day, exc)
            day += timedelta(days=1)
            continue
        rows.extend(got)
        print(f"{day}: {len(got)} rated survivors ({len(rows)} so far)", flush=True)
        day += timedelta(days=1)
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"wrote {len(rows)} rows -> {args.csv}")
    if rows:
        _report(rows)


if __name__ == "__main__":
    main()
