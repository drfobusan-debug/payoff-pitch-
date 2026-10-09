"""Season audit of the NFL game ledger: buys, one model lean per game-market, screens, CLV.

Read-only. Rows the ledger has not graded yet are settled in memory against nflverse
finals, so the audit is current even when ``nfl-engine grade`` lags the workbook;
nothing is written back.

The ledger holds both sides of every quote at every book, so raw Pass W-L is the
vig. The model is judged on one lean per game-market: the side it rates above the
de-vigged fair, at the quote with the most paired books (for spreads and totals, the
rung nearest a coin flip), scored by record and Brier against fair.

Usage::

    python -m scripts.nfl.ledger_audit [--season 2026]
"""

from __future__ import annotations

import argparse
import math
import statistics as st
from collections import defaultdict
from collections.abc import Callable

from nfl_engine.audit.ledger import ENGINE, LOSS, PUSH, WIN, LedgerEntry, grade, load_ledger
from nfl_engine.cli import _final_scores, ledger_path
from nfl_engine.market.screens import Tier

MARKETS = ("moneyline", "spread", "total")
GAP_BANDS = ((0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 1.0))


def _mean(values: list[float]) -> float:
    return st.mean(values) if values else float("nan")


def _wins(entries: list[LedgerEntry]) -> int:
    return sum(e.result == WIN for e in entries)


def record_line(label: str, entries: list[LedgerEntry]) -> str:
    wins = _wins(entries)
    losses = sum(e.result == LOSS for e in entries)
    pushes = sum(e.result == PUSH for e in entries)
    n = wins + losses + pushes
    units = sum(e.pnl for e in entries)
    roi = units / n if n else 0.0
    se = st.pstdev([e.pnl for e in entries]) / math.sqrt(n) if n > 1 else float("nan")
    clv = [e.clv for e in entries if e.clv is not None]
    drift = [e.drift for e in entries if e.drift is not None]
    return (
        f"  {label:<34} n {n:>5}  {wins}-{losses}-{pushes}  units {units:+7.2f}"
        f"  ROI {roi:+.3f} (se {se:.3f})  CLV {_mean(clv):+.4f} (n {len(clv)})"
        f"  drift {_mean(drift):+.4f} (n {len(drift)})"
    )


def settle_missing(entries: list[LedgerEntry], season: int | None) -> int:
    scores = _final_scores(season)
    filled = 0
    for entry in entries:
        if entry.result:
            continue
        final = scores.get((entry.matchup, entry.date))
        if final is None:
            continue
        grade(entry, final[0], final[1], home=entry.matchup.split(" @ ")[-1])
        filled += 1
    return filled


def _quote_rank(entry: LedgerEntry) -> tuple[int, float]:
    fair = entry.fair_prob if entry.fair_prob is not None else 0.0
    rung = 0.0 if entry.market == "moneyline" else -abs(fair - 0.5)
    return entry.paired_books, rung


def model_leans(graded: list[LedgerEntry]) -> list[LedgerEntry]:
    groups: dict[tuple[int, int, str, str], list[LedgerEntry]] = defaultdict(list)
    for e in graded:
        if e.fair_prob is not None and e.result != PUSH:
            groups[(e.season, e.week, e.matchup, e.market)].append(e)
    leans = []
    for rows in groups.values():
        positive = [e for e in rows if e.fair_prob is not None and e.model_prob > e.fair_prob]
        if positive:
            leans.append(max(positive, key=_quote_rank))
    return leans


def _fair(entry: LedgerEntry) -> float:
    return entry.fair_prob if entry.fair_prob is not None else float("nan")


def brier(entries: list[LedgerEntry], prob: Callable[[LedgerEntry], float]) -> float:
    return _mean([(prob(e) - (e.result == WIN)) ** 2 for e in entries])


def _model(entry: LedgerEntry) -> float:
    return entry.model_prob


def lean_lines(leans: list[LedgerEntry]) -> list[str]:
    out = []
    for market in MARKETS:
        rows = [e for e in leans if e.market == market]
        if not rows:
            continue
        wins = _wins(rows)
        out.append(
            f"  {market:<9} n {len(rows):>3}  {wins}-{len(rows) - wins}"
            f"  win {wins / len(rows):.3f}  avg fair {_mean([_fair(e) for e in rows]):.3f}"
            f"  avg model {_mean([e.model_prob for e in rows]):.3f}"
            f"  Brier model {brier(rows, _model):.4f} fair {brier(rows, _fair):.4f}"
            f"  units@taken price {sum(e.pnl for e in rows):+.2f}"
        )
        for lo, hi in GAP_BANDS:
            band = [e for e in rows if lo <= e.model_prob - _fair(e) < hi]
            if band:
                bw = _wins(band)
                out.append(
                    f"      gap {lo:.2f}-{hi:.2f}: n {len(band):>3}  {bw}-{len(band) - bw}"
                    f"  win {bw / len(band):.3f} vs fair {_mean([_fair(e) for e in band]):.3f}"
                    f"  units {sum(e.pnl for e in band):+.2f}"
                )
        if market == "total":
            overs = [e for e in rows if e.side == "over"]
            ow = _wins(overs)
            out.append(
                f"      model leans over {len(overs)}/{len(rows)}; overs {ow}-{len(overs) - ow}"
            )
    out.append("  by week:")
    for week in sorted({(e.season, e.week) for e in leans}):
        rows = [e for e in leans if (e.season, e.week) == week]
        wins = _wins(rows)
        out.append(
            f"    {week[0]} wk{week[1]}: {wins}-{len(rows) - wins}"
            f"  Brier model {brier(rows, _model):.4f} fair {brier(rows, _fair):.4f}"
        )
    return out


def screen_lines(leans: list[LedgerEntry]) -> list[str]:
    by: dict[str, list[LedgerEntry]] = defaultdict(list)
    for e in leans:
        for screen in filter(None, (s.strip() for s in (e.screens or "").split(";"))):
            by[screen].append(e)
    out = []
    for screen, rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        wins = _wins(rows)
        out.append(
            f"  {screen:<20} n {len(rows):>3}  {wins}-{len(rows) - wins}"
            f"  win {wins / len(rows):.3f} vs fair {_mean([_fair(e) for e in rows]):.3f}"
            f"  units if bet {sum(e.pnl for e in rows):+.2f}"
        )
    return out


def report(entries: list[LedgerEntry], filled: int) -> list[str]:
    graded = [e for e in entries if e.result in (WIN, LOSS, PUSH)]
    buys = [e for e in graded if e.tier != Tier.PASS.value]
    out = [
        f"rows {len(entries)}  graded {len(graded)} (graded here off nflverse: {filled})"
        f"  ungraded {len(entries) - len(graded)}",
        f"weeks: {sorted({(e.season, e.week) for e in entries})}",
        "",
        "1) Every buy the engine made",
    ]
    for e in sorted(buys, key=lambda e: (e.season, e.week)):
        rung = "" if e.line is None else f" {e.line:+g}"
        price = "" if e.odds is None else f" {e.odds:+.0f}"
        out.append(
            f"  wk{e.week} {e.matchup:<11} {e.market:<9} {e.side}{rung}{price} {e.book}"
            f"  {e.tier}  model {e.model_prob:.3f} fair {_fair(e):.3f}"
            f"  -> {e.result} {e.pnl:+.2f}  clv {e.clv}"
        )
    for tier in (Tier.STRONG, Tier.MODERATE):
        out.append(record_line(tier.value, [e for e in buys if e.tier == tier.value]))
    out.append(record_line("Buys", buys))
    out.append(
        record_line(
            "Pass rows (both sides, all books)", [e for e in graded if e.tier == Tier.PASS.value]
        )
    )
    leans = model_leans(graded)
    out += ["", "2) One model lean per game-market (side with model > fair)", *lean_lines(leans)]
    out += [
        "",
        "3) What each screen refused, graded on the model leans it blocked",
        *screen_lines(leans),
    ]
    out += ["", "4) CLV / drift on the leans", record_line("all leans", leans)]
    out += [record_line(f"leans {m}", [e for e in leans if e.market == m]) for m in MARKETS]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()
    path = ledger_path()
    entries = [
        e
        for e in load_ledger(path)
        if e.source == ENGINE and (args.season is None or e.season == args.season)
    ]
    filled = settle_missing(entries, args.season)
    print(f"ledger {path}")
    for line in report(entries, filled):
        print(line)


if __name__ == "__main__":
    main()
