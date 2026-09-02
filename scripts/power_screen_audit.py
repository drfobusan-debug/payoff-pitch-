"""Grade the power screen's ledger: one day, or every day it has ever recorded.

The morning note already grades yesterday, but only yesterday, and only as part
of a full screen run -- so the screen's whole record has never been readable
without rerunning it. This script reads ``power_screen_ledger.csv`` and grades
whatever range is asked for off the box scores.

The three questions the ledger module keeps apart are kept apart here too:

* **Did the positions win?** Wins, losses and units at the recorded price, split
  by market, by tier and by the note's own BUY/HOLD/AVOID rating.
* **Was the number better than the price?** Brier of the model probability and of
  the printed (anchored) probability against the two-sided no-vig mark, on the
  rows where the vig could actually be stripped.
* **Did the buy decision discriminate?** PPV and NPV of "the card bet it" against
  the base rate, because a screen that passes on everything scores a free NPV.
* **Did the ranking discriminate?** The composite's own order, in quartiles of
  rank, which the ledger could not answer at all until it began recording it.

A day can hold more than one recorded run of the screen -- a morning capture and
the re-run once lineups post are two boards, both true -- so by default only the
day's last run is graded and ``--all-runs`` grades every capture.

Nothing here writes a price, a probability or a rating. It reads the receipt.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date as Date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlb_engine.audit.grade import LOSS, PUSH, WIN  # noqa: E402
from mlb_engine.audit.power_ledger import (  # noqa: E402
    GradedPosition,
    Position,
    grade_positions,
    load,
)
from mlb_engine.config import load_config  # noqa: E402
from mlb_engine.data.results import GameResult, fetch_result  # noqa: E402
from mlb_engine.output.power_board import DISPLAY_ONLY, MARKET_LABEL  # noqa: E402

log = logging.getLogger("power_screen_audit")

LEDGER_NAME = "power_screen_ledger.csv"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=Date.fromisoformat, default=None, help="first ledger day")
    p.add_argument("--end", type=Date.fromisoformat, default=None, help="last ledger day")
    p.add_argument("--date", type=Date.fromisoformat, default=None, help="a single ledger day")
    p.add_argument("--ledger", default=None, help="override the ledger path")
    p.add_argument("--no-daily", action="store_true", help="totals only")
    p.add_argument("--rows", action="store_true", help="print every graded position")
    p.add_argument(
        "--all-runs",
        action="store_true",
        help="grade every recorded run of each day, not only the day's last one",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _in_range(p: Position, start: Date | None, end: Date | None) -> bool:
    if not p.date:
        return False
    day = Date.fromisoformat(p.date)
    return (start is None or day >= start) and (end is None or day <= end)


def _results(positions: list[Position], cache_dir: Path) -> dict[int, GameResult]:
    """Every box score the graded rows need, fetched once each."""
    out: dict[int, GameResult] = {}
    for pk in sorted({p.game_pk for p in positions if p.game_pk is not None}):
        try:
            out[pk] = fetch_result(pk, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 - one missing box score voids one game
            log.warning("could not fetch the box score for %s: %s", pk, exc)
    return out


def _units(graded: list[GradedPosition]) -> float:
    return sum(g.units for g in graded)


def _wl(graded: list[GradedPosition]) -> tuple[int, int, int]:
    return (
        sum(1 for g in graded if g.result == WIN),
        sum(1 for g in graded if g.result == LOSS),
        sum(1 for g in graded if g.result == PUSH),
    )


def _line(label: str, graded: list[GradedPosition], width: int = 22) -> str:
    w, losses, pushes = _wl(graded)
    decided = w + losses
    n = len(graded)
    pct = f"{w / decided:6.1%}" if decided else "     -"
    units = _units(graded)
    roi = f"{units / n:+7.1%}" if n else "      -"
    record = f"{w}-{losses}" + (f"-{pushes}" if pushes else "")
    return f"  {label:<{width}} {record:>9}  {pct}  {units:+7.2f}u  {roi} ROI"


def _brier(pairs: list[tuple[float, int]]) -> float | None:
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs) if pairs else None


def _priced(graded: list[GradedPosition]) -> list[tuple[GradedPosition, int]]:
    """The decided rows whose price had two sides, with their realized outcome."""
    return [
        (g, 1 if g.result == WIN else 0)
        for g in graded
        if g.result != PUSH and g.position.devigged and g.position.fair_prob is not None
    ]


def _print_probabilities(graded: list[GradedPosition]) -> None:
    pairs = _priced(graded)
    if not pairs:
        print("  no two-sided rows: nothing to score against a no-vig mark")
        return
    model = _brier([(g.position.model_prob, o) for g, o in pairs])
    shown = _brier([(g.position.shown_prob, o) for g, o in pairs])
    market = _brier([(g.position.fair_prob or 0.0, o) for g, o in pairs])
    hit = sum(o for _, o in pairs) / len(pairs)
    assert model is not None and shown is not None and market is not None
    print(f"  scored on {len(pairs)} two-sided rows; they hit {hit:.1%}")
    print(
        f"  {'Brier, model':<22} {model:.4f}   mean prob "
        f"{sum(g.position.model_prob for g, _ in pairs) / len(pairs):.3f}"
    )
    print(
        f"  {'Brier, printed':<22} {shown:.4f}   mean prob "
        f"{sum(g.position.shown_prob for g, _ in pairs) / len(pairs):.3f}"
    )
    print(
        f"  {'Brier, no-vig market':<22} {market:.4f}   mean prob "
        f"{sum(g.position.fair_prob or 0.0 for g, _ in pairs) / len(pairs):.3f}"
    )
    closer = min(("model", model), ("printed", shown), ("market", market), key=lambda x: x[1])
    print(f"  closest to the outcome: {closer[0]}")


def _last_run(positions: list[Position]) -> list[Position]:
    """One capture per day: the last run recorded for it.

    Two runs of one day are two boards and not a duplicated one, so pooling them
    counts the hitters they share twice and weights a re-run day double.
    """
    keep: dict[str, str] = {}
    for p in positions:
        keep[p.date] = max(keep.get(p.date, ""), p.run_id)
    return [p for p in positions if p.run_id == keep[p.date]]


def _print_ranking(graded: list[GradedPosition]) -> None:
    """Did the composite's order pick the bat? Quartiles of rank, best first.

    The screen's central claim is the ordering, and the ledger recorded the tier
    and the rating and never the rank, so this cut is empty for every row written
    before it was recorded -- said rather than silently skipped.
    """
    decided = [g for g in graded if g.result != PUSH and g.position.rank is not None]
    if not decided:
        print("  no row carries a rank: the ordering was not recorded")
        return
    ranks = sorted({g.position.rank for g in decided if g.position.rank is not None})
    if len(ranks) < 4:
        for r in ranks:
            print(_line(f"rank {r}", [g for g in decided if g.position.rank == r]))
        return
    step = len(ranks) / 4
    for q in range(4):
        lo, hi = ranks[int(q * step)], ranks[min(int((q + 1) * step) - 1, len(ranks) - 1)]
        bucket = [g for g in decided if g.position.rank is not None and lo <= g.position.rank <= hi]
        print(_line(f"rank {lo}-{hi}", bucket))
    missing = len([g for g in graded if g.result != PUSH]) - len(decided)
    if missing:
        print(f"  ({missing} decided rows carry no rank and are left out)")


def _print_decision(graded: list[GradedPosition]) -> None:
    """PPV and NPV of the buy decision, against the base rate it has to beat."""
    decided = [g for g in graded if g.result != PUSH]
    if not decided:
        return
    tp = sum(1 for g in decided if g.position.is_buy and g.result == WIN)
    fp = sum(1 for g in decided if g.position.is_buy and g.result == LOSS)
    fn = sum(1 for g in decided if not g.position.is_buy and g.result == WIN)
    tn = sum(1 for g in decided if not g.position.is_buy and g.result == LOSS)
    base_win = (tp + fn) / len(decided)
    ppv = tp / (tp + fp) if tp + fp else None
    npv = tn / (tn + fn) if tn + fn else None
    print(f"  base rate: {base_win:.1%} of the screen's decided rows won")
    if ppv is None:
        print("  PPV: the card bet nothing gradeable")
    else:
        print(f"  PPV  {ppv:.1%} on {tp + fp} bets      lift {100 * (ppv - base_win):+.1f}pp")
    if npv is None:
        print("  NPV: the card passed on nothing gradeable")
    else:
        print(f"  NPV  {npv:.1%} on {tn + fn} passes    lift {100 * (npv - (1 - base_win)):+.1f}pp")


def _print_day(day: str, graded: list[GradedPosition], voided: int) -> None:
    if not graded:
        print(f"{day}  nothing gradeable ({voided} voided)")
        return
    w, losses, pushes = _wl(graded)
    record = f"{w}-{losses}" + (f"-{pushes}" if pushes else "")
    buys = [g for g in graded if g.position.is_buy]
    bought = (
        f", {sum(1 for g in buys if g.result == WIN)}-"
        f"{sum(1 for g in buys if g.result == LOSS)} on the {len(buys)} it bet"
        if buys
        else ""
    )
    print(
        f"{day}  {record:>9}  {_units(graded):+7.2f}u"
        + (f"  ({voided} voided)" if voided else "")
        + bought
    )


def _print_rows(graded: list[GradedPosition]) -> None:
    """Every position, as it was shown, beside what the hitter actually did."""
    for g in sorted(graded, key=lambda x: (x.position.batter, x.position.stat)):
        p = g.position
        price = "" if p.odds is None else f"{p.odds:+.0f}"
        print(
            f"    {p.batter:<22} {p.label:<10} {price:>6} {p.book:<14}"
            f" p={p.shown_prob:.3f} mkt="
            + ("    -" if p.fair_prob is None else f"{p.fair_prob:.3f}")
            + f"  {p.tier:<13} {p.rating:<6} actual {g.actual:>2}"
            f"  {g.result:<5} {g.units:+.2f}u"
        )


def _grouped(graded: list[GradedPosition], key: str) -> list[tuple[str, list[GradedPosition]]]:
    buckets: dict[str, list[GradedPosition]] = defaultdict(list)
    for g in graded:
        if key == "market":
            buckets[MARKET_LABEL.get(g.position.stat, g.position.stat)].append(g)
        elif key == "tier":
            buckets[g.position.tier or "(none)"].append(g)
        else:
            buckets[g.position.rating or "(none)"].append(g)
    return sorted(buckets.items(), key=lambda kv: -len(kv[1]))


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    cfg = load_config()
    path = Path(args.ledger) if args.ledger else cfg.audit_dir / LEDGER_NAME
    start = args.date or args.start
    end = args.date or args.end
    recorded = [p for p in load(path) if _in_range(p, start, end)]
    # The note shows these but holds none of them, and its own scorecard leaves
    # them out; grading them here would score the screen on rows it never took.
    positions = [p for p in recorded if p.stat not in DISPLAY_ONLY]
    if positions and not args.all_runs:
        positions = _last_run(positions)
    if not positions:
        print(f"no recorded positions in {path}")
        return
    days = sorted({p.date for p in positions})
    shown_only = len(recorded) - len(positions)
    print(f"=== power screen ledger: {len(positions)} positions, {days[0]} .. {days[-1]} ===")
    if shown_only:
        print(f"({shown_only} display-only rows left out, as the note's own scorecard does)")
    print()

    results = _results(positions, cfg.cache_dir)
    all_graded: list[GradedPosition] = []
    total_voided = 0
    for day in days:
        rows = [p for p in positions if p.date == day]
        graded, voided = grade_positions(rows, results)
        all_graded.extend(graded)
        total_voided += voided
        if not args.no_daily:
            _print_day(day, graded, voided)
            if args.rows:
                _print_rows(graded)

    if not all_graded:
        print(f"\nnothing gradeable across {len(days)} days ({total_voided} voided)")
        return

    print(f"\n--- the whole record ({total_voided} voided) ---")
    print(_line("all positions", all_graded))
    print(_line("the card bet", [g for g in all_graded if g.position.is_buy]))
    print(_line("the card passed", [g for g in all_graded if not g.position.is_buy]))
    for header, key in (("by market", "market"), ("by tier", "tier"), ("by rating", "rating")):
        print(f"\n--- {header} ---")
        for label, bucket in _grouped(all_graded, key):
            print(_line(label, bucket))
    print("\n--- was the number better than the price? ---")
    _print_probabilities(all_graded)
    print("\n--- did the buy decision discriminate? ---")
    _print_decision(all_graded)
    print("\n--- did the composite's order discriminate? ---")
    _print_ranking(all_graded)
    print(
        "\nUnits are at the recorded price, one unit a position, pushes at zero;"
        "\na hitter who never batted is voided rather than lost."
    )


if __name__ == "__main__":
    main()
