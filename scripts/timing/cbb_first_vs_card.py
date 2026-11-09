"""CBB: close EV at the earliest ledgered quote vs the last featured (card) quote before tip.

Research only, reads a local copy of the engine-state tree (prices + ledger), e.g.
``git archive origin/engine-state cbb/ledger | tar x`` into an empty folder, then run this
script from that folder. Prints, per group, the mean EV under the timely no-vig
close at the first saved price minus the same at the card's price (same book and
line), with an SE clustered by game. Spends nothing and writes nothing.
"""

import glob
import json
import math
import statistics as st
from collections import defaultdict


def dec(a):
    return 1 + (a / 100 if a > 0 else 100 / -a)


def cev(a, p):
    return p * (dec(a) - 1) - (1 - p)


def shard_ts(s):
    return s.split("_")[-1]


rows = [
    json.loads(line) for f in glob.glob("cbb/ledger/*/*.jsonl") for line in open(f) if line.strip()
]
close, quotes = {}, defaultdict(list)
for r in rows:
    if r["book"] not in ("draftkings", "betmgm"):
        continue
    k = (r["event_id"], r["market"], r["side"], r["line"], r["book"])
    tip = r["tip_utc"].replace("-", "").replace(":", "")
    if r["close_status"] == "pre_tip" and r["close_consensus"] is not None:
        close[k] = r["close_consensus"]
    if r["american"] is not None and shard_ts(r["quote_shard"]) < tip:
        quotes[k].append((shard_ts(r["quote_shard"]), r["pass_kind"], r["american"]))
pops = defaultdict(list)
for k, qs in quotes.items():
    if k not in close or not (0.03 < close[k] < 0.97):
        continue
    qs.sort()
    feat = [q for q in qs if q[1] == "featured"]
    if not feat or qs[0][0] >= feat[-1][0]:
        continue
    a, b = cev(qs[0][2], close[k]), cev(feat[-1][2], close[k])
    pops[f"{k[1]} all sides"].append((k[0], a, b))
    pops["all game lines"].append((k[0], a, b))
for g, v in sorted(pops.items()):
    by = defaultdict(list)
    for gid, a, b in v:
        by[gid].append(a - b)
    gm = [st.mean(x) for x in by.values()]
    print(
        f"{g:18} n={len(v):5} games={len(by):4} EV@first {100 * st.mean(x[1] for x in v):+.2f} EV@card {100 * st.mean(x[2] for x in v):+.2f} first-card {100 * st.mean(gm):+.2f} se {100 * st.stdev(gm) / math.sqrt(len(gm)):.2f}"
    )
