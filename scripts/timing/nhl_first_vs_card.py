"""NHL: close EV had each row been taken at the first saved price vs the card price.

Research only, reads a local copy of the engine-state tree (prices + ledger), e.g.
``git archive origin/engine-state nhl/ledger nhl/prices | tar x`` into an empty folder, then run this
script from that folder. Prints, per group, the mean EV under the timely no-vig
close at the first saved price minus the same at the card's price (same book and
line), with an SE clustered by game. Spends nothing and writes nothing.
"""

import csv
import glob
import json
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


first: dict = {}
for f in sorted(glob.glob("nhl/prices/2026-*/*.csv")):
    for r in csv.DictReader(open(f)):
        if not r["american"]:
            continue
        k = (
            r["game_date"],
            r["matchup"],
            r["market"],
            r["side"],
            r["entity"],
            float(r["line"]) if r["line"] else None,
            r["book"],
        )
        if k not in first or r["captured_at"] < first[k][0]:
            first[k] = (r["captured_at"], float(r["american"]))

pops = defaultdict(list)
skipped = defaultdict(int)
for f in sorted(glob.glob("nhl/ledger/graded_2026-*.json")):
    for r in json.load(open(f)):
        if (
            r.get("card_tag") not in ("early", "initial")
            or r.get("close_consensus") is None
            or not r.get("puck_drop")
        ):
            continue
        pd, pa = ts(r["puck_drop"]), ts(r["priced_at"])
        if pa >= pd or not r.get("close_at") or (pd - ts(r["close_at"])).total_seconds() > 45 * 60:
            continue
        if r["outcome"] == "void":
            continue
        k = (
            r["slate_date"],
            r["matchup"],
            r["market"],
            r["side"],
            r["entity"] or "",
            None if r["line"] is None else float(r["line"]),
            r["book"],
        )
        fq = first.get(k)
        groups = []
        if r.get("pass_gate") and r["tier"] != "Pass":
            groups.append("buys")
        if r.get("sim_buy"):
            groups.append("sim would-be buys")
        if r["market"] in ("game_ml", "game_total", "game_pl") and r["model_prob"] > r["consensus"]:
            groups.append(f"model leans {r['market']}")
        if r["market"] == "game_total":
            groups.append(f"all {r['side']}s game_total")
        for g in groups:
            if fq is None or fq[0] >= r["priced_at"]:
                skipped[g] += 1
                continue
            hrs = (pa - ts(fq[0])).total_seconds() / 3600
            pops[g].append(
                (
                    r["slate_date"] + r["matchup"],
                    cev(fq[1], r["close_consensus"]),
                    cev(r["american"], r["close_consensus"]),
                    hrs,
                )
            )

print(
    f"{'group':28} {'n':>5} {'games':>5} {'hrs early':>9} {'EV@first':>9} {'EV@card':>8} {'first-card':>10} {'se':>6}"
)
for g in sorted(pops):
    rows = pops[g]
    by = defaultdict(list)
    for gid, a, b, _h in rows:
        by[gid].append(a - b)
    gm = [st.mean(v) for v in by.values()]
    se = st.stdev(gm) / math.sqrt(len(gm)) if len(gm) > 1 else float("nan")
    print(
        f"{g:28} {len(rows):5} {len(by):5} {st.median(h for *_, h in rows):9.1f} {100 * st.mean(a for _, a, _, _ in rows):+9.2f} {100 * st.mean(b for _, _, b, _ in rows):+8.2f} {100 * st.mean(gm):+10.2f} {100 * se:6.2f}  (no earlier quote: {skipped[g]})"
    )
