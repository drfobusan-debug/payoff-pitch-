"""Report-only replay of the close quarantine over the graded ledger.

Applies the rules that need no tape -- ``implausible`` and ``move_gt_15pp`` --
to every ledger row that already carries a close, and prints the counts and the
mean CLV / ``clv_ev`` and ROI with and without the quarantined rows. Never
rewrites the ledger.

    python scripts/clv_quarantine_replay.py --ledger path/to/ledger.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from mlb_engine.audit import clv
from mlb_engine.audit.ledger import engine_rows, load_ledger, one_side_per_prop

BUY_TIERS = ("Strong buy", "Moderate buy")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--move-limit", type=float, default=clv.MOVE_LIMIT)
    args = ap.parse_args(argv)
    rows = [
        e for e in one_side_per_prop(engine_rows(load_ledger(args.ledger)))
        if e.clv is not None and e.clv_ev is not None and e.close_odds is not None
    ]
    reasons: Counter[str] = Counter()
    kept = []
    for e in rows:
        code = None
        if not clv.is_plausible_close(e.close_odds or 0.0, e.market):
            code = clv.IMPLAUSIBLE
        elif e.close_prob is not None and abs(e.close_prob - clv._no_vig_at_bet(e)) > args.move_limit:
            code = clv.MOVE_GT_15PP
        if code:
            reasons[code] += 1
            print(f"  {code}: {e.date} {e.matchup} {e.selection} {e.odds:+.0f} -> {e.close_odds:+.0f}")
        else:
            kept.append(e)
    print(f"CLV rows: {len(rows)}; quarantined {sum(reasons.values())} {dict(reasons)}")
    buys = [e for e in rows if e.tier in BUY_TIERS]
    kept_buys = [e for e in kept if e.tier in BUY_TIERS]
    print(f"Buy rows with a close: {len(buys)}; quarantined {len(buys) - len(kept_buys)}")
    for label, xs in (
        ("all", rows), ("quarantined out", kept),
        ("buys", buys), ("buys, quar. out", kept_buys),
    ):
        bought = [e for e in xs if e.tier in BUY_TIERS]
        roi = sum(e.pnl for e in bought) / len(bought) if bought else float("nan")
        print(
            f"  {label:16s} n={len(xs)} mean clv={_mean([e.clv or 0.0 for e in xs]):+.4f} "
            f"mean clv_ev={_mean([e.clv_ev or 0.0 for e in xs]) * 100:+.2f}c buy ROI={roi:+.2%}"
        )
    print("Report only: the ledger is not rewritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
