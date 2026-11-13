"""Paper check: high-z (1<=z<2) home vs average-z Under, on as-of Torvik tables. Read-only."""
from __future__ import annotations

import math
import statistics as st
import sys
from datetime import date, timedelta

from cbb_engine import audit, bands, cli, teamgroups
from cbb_engine.config import EXECUTION_BOOKS, SPORT, runtime_root
from engine_runtime import ledger, quotes

DAY = date.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else (date.today() - timedelta(days=1)).isoformat())
root = runtime_root()
days = sorted(p.name for p in (root / "ledger" / SPORT).glob("*") if p.is_dir() and p.name <= DAY.isoformat())
season = [r for d in days for r in ledger.record(root, SPORT, d).values()]
ctx, boards = cli._season_ctx(root, season, DAY)
called = {e: c for cs in cli._called_by_day(root, season, boards).values() for e, c in cs.items()}
closes: dict[str, list[quotes.Quote]] = {}
for d in days:
    closes.update(audit._closes(quotes.read_day(root, SPORT, d))[0])
main = [r for r in bands.main_rows(season) if r.event_id in ctx]
unders = [r for r in main if r.market == "game_total" and r.side == "under"]
withz = sum(1 for c in ctx.values() if c.home.z is not None and c.away.z is not None)
print(f"as of {DAY}: {len(ctx)} games with context, {withz} with both z; {len(unders)} graded Unders")


def dec(a: float) -> float:
    return 1 + (a / 100 if a > 0 else 100 / -a)


def close_bet(r: ledger.Row) -> tuple[float, float] | None:
    """Flat 1u Under at the best DK/BetMGM close (highest line, then price): (pnl, pnl less no-vig expectation)."""
    c = called.get(r.event_id)
    qs = [q for q in closes.get(r.event_id, []) if q.market == "game_total" and q.side == "under"
          and q.book in EXECUTION_BOOKS and q.line is not None and q.opposite_american]
    if c is None or not qs:
        return None
    q = max(qs, key=lambda q: (q.line, dec(q.american)))
    win = dec(q.american) - 1
    p = (1 / dec(q.american)) / (1 / dec(q.american) + 1 / dec(q.opposite_american))
    pts = c.points
    pnl = win if pts < q.line else (-1.0 if pts > q.line else 0.0)
    exp = 0.0 if pts == q.line else p * win - (1 - p)
    return pnl, pnl - exp


def mse(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    return st.mean(xs), st.stdev(xs) / math.sqrt(n)


def show(label: str, rs: list[ledger.Row], rest: list[ledger.Row] | None = None) -> None:
    rs = sorted(rs, key=lambda r: (r.slate_date, r.tip_utc))
    if not rs:
        print(f"\n{label}: 0 games")
        return
    pn = [r.pnl or 0.0 for r in rs]
    ex = [(r.pnl or 0.0) - bands.market_expected(r) for r in rs]
    w, l = sum(p > 0 for p in pn), sum(p < 0 for p in pn)
    h = len(ex) // 2
    roi, roi_se = mse(pn)
    xs, xs_se = mse(ex)
    print(f"\n{label}")
    print(f"  card price: {w}-{l}-{len(pn) - w - l}, n={len(pn)}, {sum(pn):+.1f}u, ROI {roi:+.1%} ± {roi_se:.1%}; "
          f"vs no-vig {sum(ex):+.1f}u, {xs:+.1%} ± {xs_se:.1%} per bet = {xs / xs_se if xs_se else float('nan'):.2f} SE; "
          f"halves {sum(ex[:h]):+.1f} / {sum(ex[h:]):+.1f}")
    if rest is not None:
        o, o_se = mse([r.pnl or 0.0 for r in rest])
        d_se = math.hypot(roi_se, o_se)
        print(f"  every other Under: n={len(rest)}, ROI {o:+.1%} ± {o_se:.1%}; difference {roi - o:+.1%} ± {d_se:.1%} = {(roi - o) / d_se:.2f} SE")
    cb = [x for x in (close_bet(r) for r in rs) if x]
    if cb:
        cp, cx = [a for a, _ in cb], [b for _, b in cb]
        cw, cl = sum(p > 0 for p in cp), sum(p < 0 for p in cp)
        m, s = mse(cx)
        hh = len(cx) // 2
        print(f"  DK/BetMGM close: {cw}-{cl}-{len(cp) - cw - cl}, n={len(cp)}, {sum(cp):+.1f}u, ROI {st.mean(cp):+.1%}; "
              f"vs no-vig close {sum(cx):+.1f}u, {m:+.1%} ± {s:.1%} = {m / s if s else float('nan'):.2f} SE; halves {sum(cx[:hh]):+.1f} / {sum(cx[hh:]):+.1f}")
    else:
        print("  DK/BetMGM close: no timely close with a final")
    card_miss = [called[r.event_id].points - r.line for r in rs if r.event_id in called and r.line is not None]
    close_miss = [c.points - c.total for c in (called.get(r.event_id) for r in rs) if c and c.total is not None]
    move = [called[r.event_id].total - ctx[r.event_id].opener[("game_total", "over")] for r in rs
            if r.event_id in called and called[r.event_id].total is not None and ("game_total", "over") in ctx[r.event_id].opener]
    for name, xs_ in (("final less card line", card_miss), ("final less close (consensus)", close_miss), ("close less opener", move)):
        if xs_:
            m, s = mse(xs_)
            print(f"  {name}: mean {m:+.2f} ± {s:.2f} pts, median {st.median(xs_):+.1f}, n={len(xs_)}")


def overlap(c: bands.GameCtx) -> bool:
    return teamgroups._pairs(c, "power", True, False) or teamgroups._pairs(c, "pace", "fast", "slow")


for width in (0.5, 0.75):
    cell = [r for r in unders if bands._high_vs_avg(ctx[r.event_id], 1.0, 2.0, width) is not None]
    home = [r for r in cell if bands._high_vs_avg(ctx[r.event_id], 1.0, 2.0, width) == "home" and not ctx[r.event_id].neutral]
    ids = {id(r) for r in home}
    print(f"\n===== average |z| <= {width}: {len(cell)} high-vs-average Unders, {len(home)} with the high team at home (not neutral)")
    show(f"Locked cell, high team home, |z| <= {width}", home, [r for r in unders if id(r) not in ids])
    show("  inside the overlap (power-vs-mid or fast-vs-slow)", [r for r in home if overlap(ctx[r.event_id])])
    show("  leftover (neither)", [r for r in home if not overlap(ctx[r.event_id])])
    show("  power-vs-mid only tag", [r for r in home if teamgroups._pairs(ctx[r.event_id], "power", True, False)])
    show("  fast-vs-slow only tag", [r for r in home if teamgroups._pairs(ctx[r.event_id], "pace", "fast", "slow")])
    show("  road/neutral high team (logged, out of the rule)", [r for r in cell if id(r) not in ids])
