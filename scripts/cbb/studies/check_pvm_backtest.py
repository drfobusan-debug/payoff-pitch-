"""Power vs mid-major Under on the bought 2024-25 hist closes (median book, median line). Read-only, paper.

    PYTHONPATH=. python scripts/cbb/studies/check_pvm_backtest.py --root ~/.cbb_engine/runtime \
        --espn ~/.cbb_engine/data --results 2025_team_results.csv
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import band_backtest as bt  # noqa: E402

from cbb_engine.bands import POWER  # noqa: E402
from cbb_engine.data import torvik  # noqa: E402
from cbb_engine.output import teamstats  # noqa: E402
from engine_runtime import ledger  # noqa: E402


def dec(a: float) -> float:
    return 1 + (a / 100 if a > 0 else 100 / -a)


def excess(r: ledger.Row) -> float:
    if r.outcome == "push" or r.american is None or r.p_fair is None:
        return 0.0
    w = dec(r.american) - 1
    return (r.pnl or 0.0) - (r.p_fair * w - (1 - r.p_fair))


def ms(x: list[float]) -> tuple[float, float]:
    return (
        (st.mean(x), st.stdev(x) / math.sqrt(len(x)))
        if len(x) > 1
        else (float("nan"), float("nan"))
    )


def show(label: str, g: list[tuple[str, ledger.Row, float | None]]) -> None:
    if not g:
        print(f"  {label}: none")
        return
    rs = [r for _, r, _ in g]
    w, lo, pu = (sum(r.outcome == k for r in rs) for k in ("win", "loss", "push"))
    roi, ex = ms([r.pnl or 0.0 for r in rs]), ms([excess(r) for r in rs])
    h = len(g) // 2
    halves = sum(excess(r) for _, r, _ in g[:h]), sum(excess(r) for _, r, _ in g[h:])
    print(
        f"  {label}: n={len(g)} Under {w}-{lo}-{pu}, {sum(r.pnl or 0 for r in rs):+.1f}u, ROI {roi[0]:+.1%} ± {roi[1]:.1%}; "
        f"vs no-vig {ex[0]:+.1%} ± {ex[1]:.1%} = {ex[0] / ex[1] if ex[1] else 0:.2f} SE; halves {halves[0]:+.1f} / {halves[1]:+.1f}"
    )


def trim(fav: float | None) -> bool:
    return fav is not None and 10 <= fav < 30


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--espn", type=Path, required=True)
    ap.add_argument(
        "--results", type=Path, required=True, help="Torvik 2025_team_results.csv (conferences)"
    )
    ap.add_argument("--start", default="2024-11-04")
    ap.add_argument("--end", default="2025-04-08")
    a = ap.parse_args()
    confs = {r.get("team", ""): r.get("conf", "") for r in torvik.read_csv(a.results)}
    pvm: list[tuple[str, ledger.Row, float | None]] = []
    rest: list[tuple[str, ledger.Row, float | None]] = []
    unmatched = 0
    day, end = Date.fromisoformat(a.start), Date.fromisoformat(a.end)
    while day <= end:
        teams = bt.hist_teams(a.root, day.isoformat())
        if teams:
            rows = bt.close_rows(a.root, day.isoformat(), teams, bt.board(a.espn, day))
            for eid in sorted({r.event_id for r in rows}):
                under = next(
                    (
                        r
                        for r in rows
                        if r.event_id == eid and r.market == "game_total" and r.side == "under"
                    ),
                    None,
                )
                home = next(
                    (
                        r
                        for r in rows
                        if r.event_id == eid and r.market == "game_ats" and r.side == "home"
                    ),
                    None,
                )
                if under is None:
                    continue
                th, ta = (teamstats.match_team(n, confs) for n in teams[eid])
                if not th or not ta:
                    unmatched += 1
                    continue
                ph, pa = confs[th] in POWER, confs[ta] in POWER
                spread = home.line if home else None
                if ph != pa:
                    fav = None if spread is None else (-spread if ph else spread)
                    pvm.append((day.isoformat(), under, fav))
                else:
                    rest.append((day.isoformat(), under, None))
        day += timedelta(days=1)
    print(
        f"{a.start} to {a.end}: {len(pvm)} Power-vs-mid and {len(rest)} other games with a graded close Under; "
        f"{unmatched} unmatched to Torvik"
    )
    for label, cut in (
        ("all season", lambda d: True),
        ("November", lambda d: d[5:7] == "11"),
        ("Dec onward", lambda d: d[5:7] != "11"),
    ):
        print(f"== {label} ==")
        show("Power vs mid, Power fav 10-29.5", [g for g in pvm if cut(g[0]) and trim(g[2])])
        show("all Power vs mid", [g for g in pvm if cut(g[0])])
        show("every other game", [g for g in rest if cut(g[0])])
    print("== Power vs mid by Power closing spread (season) ==")
    for lo, hi in ((-99, 10), (10, 20), (20, 30), (30, 99)):
        show(f"fav {lo}-{hi}", [g for g in pvm if g[2] is not None and lo <= g[2] < hi])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
