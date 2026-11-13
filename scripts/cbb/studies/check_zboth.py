"""Paper check: games where both teams have a positive Torvik AdjEM z-score, on as-of tables. Read-only.

z = (AdjOE - AdjDE - mean) / sd over every team in the team_results table saved on or before the game's tip day.
Optional 2nd arg: a team_results CSV used for every game instead (hindsight, for a preview only)."""
from __future__ import annotations

import math
import statistics as st
import sys
from datetime import date, timedelta
from pathlib import Path

from cbb_engine import audit, bands, cli
from cbb_engine.config import EXECUTION_BOOKS, SPORT, data_dir, runtime_root
from cbb_engine.data import torvik
from cbb_engine.output import teamstats
from engine_runtime import ledger, quotes

DAY = date.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else (date.today() - timedelta(days=1)).isoformat())
FIXED = Path(sys.argv[2]) if len(sys.argv) > 2 else None
root = runtime_root()
days = sorted(p.name for p in (root / "ledger" / SPORT).glob("*") if p.is_dir() and p.name <= DAY.isoformat())
season = [r for d in days for r in ledger.record(root, SPORT, d).values()]
ctx, boards = cli._season_ctx(root, season, DAY)
called = {e: c for cs in cli._called_by_day(root, season, boards).values() for e, c in cs.items()}
closes: dict[str, list[quotes.Quote]] = {}
teams: dict[str, tuple[str, str]] = {}
for d in days:
    closes.update(audit._closes(quotes.read_day(root, SPORT, d))[0])
    teams.update(cli._manifest_teams(root, date.fromisoformat(d)))
main = [r for r in bands.main_rows(season) if r.event_id in ctx]
favs = bands.ats_favourites(main)

_tables: dict[Path, dict[str, float]] = {}


def zmap(path: Path) -> dict[str, float]:
    if path not in _tables:
        em = {}
        for r in torvik.read_csv(path):
            try:
                em[r["team"]] = float(r["adjoe"]) - float(r["adjde"])
            except (KeyError, ValueError):
                continue
        m, s = st.mean(em.values()), st.pstdev(em.values())
        _tables[path] = {t: (v - m) / s for t, v in em.items()}
    return _tables[path]


zs: dict[str, tuple[float, float]] = {}
stale: list[int] = []
tips = {}
for r in season:
    tips.setdefault(r.event_id, r.tip_utc)
for eid, (home, away) in teams.items():
    if eid not in ctx or eid not in tips:
        continue
    td = bands.tip_day(tips[eid])
    if td is None:
        continue
    if FIXED:
        path = FIXED
    else:
        a = torvik.as_of(data_dir(), "team_results", td)
        if a.path is None:
            continue
        path = a.path
        stale.append(a.stale_days or 0)
    z = zmap(path)
    th, ta = teamstats.match_team(home, z), teamstats.match_team(away, z)
    if th and ta:
        zs[eid] = (z[th], z[ta])
src = f"fixed table {FIXED}" if FIXED else "as-of tables (saved before each tip day)"
print(f"as of {DAY}: {len(zs)} games with both z from {src}; table age days median "
      f"{st.median(stale) if stale else 'n/a'}, max {max(stale) if stale else 'n/a'}")


def dec(a: float) -> float:
    return 1 + (a / 100 if a > 0 else 100 / -a)


def mse(xs: list[float]) -> tuple[float, float]:
    return (st.mean(xs), st.stdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else (float("nan"), float("nan"))


def close_under(eid: str, side: str) -> tuple[float, float] | None:
    c = called.get(eid)
    qs = [q for q in closes.get(eid, []) if q.market == "game_total" and q.side == side
          and q.book in EXECUTION_BOOKS and q.line is not None and q.opposite_american]
    if c is None or not qs:
        return None
    q = max(qs, key=lambda q: ((q.line if side == "under" else -q.line), dec(q.american)))
    win = dec(q.american) - 1
    p = (1 / dec(q.american)) / (1 / dec(q.american) + 1 / dec(q.opposite_american))
    hit = c.points < q.line if side == "under" else c.points > q.line
    pnl = 0.0 if c.points == q.line else (win if hit else -1.0)
    return pnl, pnl - (0.0 if c.points == q.line else p * win - (1 - p))


def show(label: str, rs: list[ledger.Row]) -> None:
    rs = sorted(rs, key=lambda r: (r.slate_date, r.tip_utc))
    if not rs:
        print(f"  {label}: 0")
        return
    pn = [r.pnl or 0.0 for r in rs]
    ex = [(r.pnl or 0.0) - bands.market_expected(r) for r in rs]
    w, l = sum(p > 0 for p in pn), sum(p < 0 for p in pn)
    h = len(ex) // 2
    roi, rs_ = mse(pn)
    x, xs = mse(ex)
    print(f"  {label}: {w}-{l}-{len(pn) - w - l}, n={len(pn)}, {sum(pn):+.1f}u, ROI {roi:+.1%} ± {rs_:.1%}; "
          f"vs no-vig {x:+.1%} ± {xs:.1%} = {x / xs if xs else float('nan'):.2f} SE; halves {sum(ex[:h]):+.1f} / {sum(ex[h:]):+.1f}")


def ml_fav(eid: str, rs: list[ledger.Row]) -> str | None:
    ml = {r.side: r for r in rs if r.market == "game_ml" and r.event_id == eid}
    if len(ml) < 2:
        return None
    return max(ml, key=lambda s: ml[s].p_fair or 0.0)


by_event: dict[str, list[ledger.Row]] = {}
for r in main:
    by_event.setdefault(r.event_id, []).append(r)

CUTS = (
    ("both z >= +1", lambda h, a: h >= 1 and a >= 1),
    ("both z > 0", lambda h, a: h > 0 and a > 0),
    ("both z > 0, not both >= +1", lambda h, a: h > 0 and a > 0 and not (h >= 1 and a >= 1)),
    ("every other game with both z", lambda h, a: not (h > 0 and a > 0)),
)
for name, cut in CUTS:
    ids = {e for e, (h, a) in zs.items() if cut(h, a)}
    rs = [r for r in main if r.event_id in ids]
    print(f"\n===== {name}: {len(ids)} games ({sum(1 for e in ids if ctx[e].neutral)} neutral)")
    show("ATS favourite", [r for r in rs if r.market == "game_ats" and favs.get(r.event_id) == r.side])
    show("ATS underdog", [r for r in rs if r.market == "game_ats" and favs.get(r.event_id) not in (None, r.side)])
    show("ATS home", [r for r in rs if r.market == "game_ats" and r.side == "home" and not ctx[r.event_id].neutral])
    show("ATS home underdog", [r for r in rs if r.market == "game_ats" and r.side == "home" and not ctx[r.event_id].neutral
                                and favs.get(r.event_id) == "away"])
    show("ML favourite", [r for r in rs if r.market == "game_ml" and ml_fav(r.event_id, by_event[r.event_id]) == r.side])
    show("ML underdog", [r for r in rs if r.market == "game_ml" and ml_fav(r.event_id, by_event[r.event_id]) not in (None, r.side)])
    show("Over (card price)", [r for r in rs if r.market == "game_total" and r.side == "over"])
    show("Under (card price)", [r for r in rs if r.market == "game_total" and r.side == "under"])
    for side in ("under", "over"):
        cb = [x for x in (close_under(e, side) for e in sorted(ids, key=lambda e: tips[e])) if x]
        if cb:
            cp, cx = [a for a, _ in cb], [b for _, b in cb]
            m, s = mse(cx)
            print(f"  {side.title()} at DK/BetMGM close: {sum(p > 0 for p in cp)}-{sum(p < 0 for p in cp)}, n={len(cp)}, "
                  f"{sum(cp):+.1f}u; vs no-vig close {m:+.1%} ± {s:.1%} = {m / s if s else float('nan'):.2f} SE")
    miss = [called[e].points - called[e].total for e in ids if e in called and called[e].total is not None]
    if len(miss) > 1:
        m, s = mse(miss)
        print(f"  final less close total: {m:+.2f} ± {s:.2f} pts, median {st.median(miss):+.1f}, n={len(miss)}")
    cov = [(called[e].home_margin + called[e].home_spread) * (1 if favs[e] == "home" else -1)
           for e in ids if e in called and called[e].home_spread is not None and e in favs]
    if len(cov) > 1:
        m, s = mse(cov)
        print(f"  favourite margin vs close spread: {m:+.2f} ± {s:.2f} pts, n={len(cov)}")
