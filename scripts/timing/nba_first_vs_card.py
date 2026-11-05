"""NBA: close EV had each card row been taken at the first saved board price vs the card price.

Research only, reads a local copy of the engine-state tree (prices + ledger), e.g.
``git archive origin/engine-state nba/ledger nba/prices | tar x`` into an empty folder, then run this
script from that folder. Prints, per group, the mean EV under the timely no-vig
close at the first saved price minus the same at the card's price (same book and
line), with an SE clustered by game. Spends nothing and writes nothing.
"""

import csv
import glob
import gzip
import math
import statistics as st
from collections import defaultdict
from datetime import datetime


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def dec(a):
    return 1 + (a / 100 if a > 0 else 100 / -a)


def cev(a, p):
    return p * (dec(a) - 1) - (1 - p)


def ln(x):
    return float(x) if x not in ("", None) else None


first = {}
for f in sorted(glob.glob("nba/prices/2026-*/[be][ov]*_*.csv")):
    for r in csv.DictReader(open(f)):
        if not r["american"]:
            continue
        k = (r["event_id"], r["market"], r["side"], r["entity"], ln(r["line"]), r["book"])
        if k not in first or r["captured_at"] < first[k][0]:
            first[k] = (r["captured_at"], float(r["american"]))

pops = defaultdict(list)
miss = defaultdict(int)
for f in sorted(glob.glob("nba/ledger/2026-*/graded.csv.gz")):
    for r in csv.DictReader(gzip.open(f, "rt")):
        if (
            r["close_status"] != "close"
            or not r["close_fair"]
            or r["book"] not in ("draftkings", "betmgm")
            or r["outcome"] == "void"
        ):
            continue
        if r["priced_at"] >= r["tip_utc"]:
            continue
        m = r["market"]
        grp = (
            "props"
            if m.startswith("pl_")
            else ("alt lines" if m.startswith("alt_") else "game lines")
        )
        groups = [f"{grp} all sides"]
        if grp == "props" and r["side"] == "under":
            groups.append("props unders (Over-bias side)")
        if m in ("game_total", "h1_total"):
            groups.append(f"totals {r['side']}s")
        k = (r["event_id"], m, r["side"], r["entity"], ln(r["line"]), r["book"])
        fq = first.get(k)
        for g in groups:
            if fq is None or fq[0] >= r["priced_at"]:
                miss[g] += 1
                continue
            p = float(r["close_fair"])
            pops[g].append(
                (
                    r["event_id"],
                    cev(fq[1], p),
                    cev(float(r["american"]), p),
                    (ts(r["priced_at"]) - ts(fq[0])).total_seconds() / 3600,
                )
            )

print(
    f"{'group':32} {'n':>6} {'games':>5} {'hrs early':>9} {'EV@first':>9} {'EV@card':>8} {'first-card':>10} {'se':>6}"
)
for g in sorted(pops):
    rows = pops[g]
    by = defaultdict(list)
    for gid, a, b, _h in rows:
        by[gid].append(a - b)
    gm = [st.mean(v) for v in by.values()]
    se = st.stdev(gm) / math.sqrt(len(gm)) if len(gm) > 1 else float("nan")
    print(
        f"{g:32} {len(rows):6} {len(by):5} {st.median(h for *_, h in rows):9.1f} {100 * st.mean(r[1] for r in rows):+9.2f} {100 * st.mean(r[2] for r in rows):+8.2f} {100 * st.mean(gm):+10.2f} {100 * se:6.2f}  (no earlier quote: {miss[g]})"
    )
