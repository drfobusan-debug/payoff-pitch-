"""Probe faded picks (model_prob < 0.5) that won for a reclaimable commonality.

A false negative here = the model faded a bet (raw prob < 50%) that actually hit.
We look for *systematic* pockets -- a (market, raw-probability band) where the
realized win rate is meaningfully above 50%. Those are picks the raw model
under-rates; the isotonic calibrator maps them back above 0.5, converting them
into playable positives. This script quantifies how many such picks exist and
where, and checks side/home-away commonalities.
"""

from __future__ import annotations

import pickle
from collections import defaultdict

from mlb_engine.audit.grade import PUSH, WIN
from mlb_engine.calibration import Calibrator
from mlb_engine.pipeline import _CALIBRATION_FILE


def main() -> int:
    graded = pickle.load(open("/tmp/backtest_graded.pkl", "rb"))
    cal = Calibrator.from_json(_CALIBRATION_FILE)

    # faded picks: raw prob < 0.5
    faded = [
        (r, res) for r, res in graded if res != PUSH and (r.raw_prob or r.model_prob) < 0.5
    ]
    won = sum(1 for _, res in faded if res == WIN)
    print(f"Faded picks (raw<0.5): {len(faded)}, of which won (false negatives): {won} "
          f"({won / len(faded) * 100:.1f}%)")

    # reclaimable pockets: (market, 0.05 raw band) with realized win% > 52.4% (breakeven)
    band: dict[tuple[str, int], list[int]] = defaultdict(list)
    for r, res in faded:
        raw = r.raw_prob if r.raw_prob is not None else r.model_prob
        b = int(raw * 20)  # 0.05-wide band index
        band[(r.market, b)].append(1 if res == WIN else 0)

    print("\nReclaimable pockets (faded band, realized win% > 52.4%, n>=100):")
    reclaimed = 0
    for (market, b), outs in sorted(band.items()):
        if len(outs) < 100:
            continue
        wr = sum(outs) / len(outs)
        if wr > 0.524:
            lo, hi = b * 5, (b + 1) * 5
            # does calibration actually lift this band above 0.5?
            mid = (lo + hi) / 200
            lifted = cal.apply(market, mid)
            reclaimed += sum(outs)
            flag = "-> calibrated ABOVE 0.5 (reclaimed)" if lifted >= 0.5 else "(still faded)"
            print(f"  {market:<13} raw {lo:>2}-{hi:<2}% n={len(outs):<5} "
                  f"realized={wr * 100:4.1f}%  cal({mid * 100:.0f}%)={lifted * 100:4.1f}% {flag}")
    print(f"\nWins sitting in >52.4% faded pockets: {reclaimed}")

    # commonality by side / home-away among false negatives
    print("\nFalse-negative rate by dimension (won | total | win%):")
    for label, keyfn in (
        ("side", lambda r: r.side or "-"),
        ("home/away", lambda r: r.team_side or "-"),
        ("category", lambda r: r.category),
    ):
        agg: dict[str, list[int]] = defaultdict(list)
        for r, res in faded:
            agg[keyfn(r)].append(1 if res == WIN else 0)
        for k in sorted(agg):
            o = agg[k]
            if len(o) >= 100:
                print(f"  {label:<10} {k:<8} {sum(o):>6} | {len(o):>6} | {sum(o) / len(o) * 100:4.1f}%")

    # ---- FALSE POSITIVES: favored (raw>=0.5) picks that lost ----
    fav = [
        (r, res) for r, res in graded if res != PUSH and (r.raw_prob or r.model_prob) >= 0.5
    ]
    fp = sum(1 for _, res in fav if res != WIN)
    print(f"\nFavored picks (raw>=0.5): {len(fav)}, of which lost (false positives): {fp} "
          f"({fp / len(fav) * 100:.1f}%)")

    print("\nWorst FP pockets (favored band, realized win% < 47.6% breakeven, n>=100):")
    fband: dict[tuple[str, int], list[int]] = defaultdict(list)
    for r, res in fav:
        raw = r.raw_prob if r.raw_prob is not None else r.model_prob
        fband[(r.market, int(raw * 20))].append(1 if res == WIN else 0)
    for (market, b), outs in sorted(fband.items()):
        if len(outs) < 100:
            continue
        wr = sum(outs) / len(outs)
        if wr < 0.476:
            lo, hi = b * 5, (b + 1) * 5
            mid = (lo + hi) / 200
            lifted = cal.apply(market, mid)
            flag = "-> calibrated BELOW 0.5 (demoted)" if lifted < 0.5 else "(still favored)"
            print(f"  {market:<13} raw {lo:>2}-{hi:<2}% n={len(outs):<5} "
                  f"realized={wr * 100:4.1f}%  cal({mid * 100:.0f}%)={lifted * 100:4.1f}% {flag}")

    print("\nFalse-positive rate by dimension (lost | total | loss%):")
    for label, keyfn in (
        ("side", lambda r: r.side or "-"),
        ("home/away", lambda r: r.team_side or "-"),
        ("category", lambda r: r.category),
        ("market", lambda r: r.market),
    ):
        agg2: dict[str, list[int]] = defaultdict(list)
        for r, res in fav:
            agg2[keyfn(r)].append(0 if res == WIN else 1)  # 1 = lost = FP
        for k in sorted(agg2):
            o = agg2[k]
            if len(o) >= 100:
                print(f"  {label:<10} {k:<10} {sum(o):>6} | {len(o):>6} | {sum(o) / len(o) * 100:4.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
