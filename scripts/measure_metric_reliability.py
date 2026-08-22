"""How many plate appearances each screened metric needs before it means anything.

The power screen scores eleven metrics one point apiece. That is only defensible
if the eleven stabilise at similar speeds, and they do not: a hitter's swing
speed is knowable in a dozen swings and his wOBA is not knowable in a season.

Method is split-half reliability, run on the metric itself rather than on a
proxy. For a target sample size ``n``, every hitter with at least ``2n`` plate
appearances in the window has his PA shuffled and split into two independent
halves of ``n``; the metric is computed on each half; the correlation across
hitters between the two halves is the reliability at ``n``. Repeated over a grid
of ``n`` and interpolated to find where r crosses .50 and .70.

Split-half is the right tool here rather than year-over-year correlation because
it isolates measurement noise from real change in the player: both halves come
from the same weeks, so anything that fails to repeat is noise by construction.
It is an *upper* bound on how much a metric can tell you about tonight -- a
metric that cannot even agree with itself inside one window certainly cannot
forecast the next one.

    python scripts/measure_metric_reliability.py --days 150 --reps 8

Prints a table of PA thresholds and the JSON that
``mlb_engine/features/reliability.py`` stores. Re-run it when a season of data
has accumulated; the stored numbers are one season and should move.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import date as Date

import numpy as np
import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.output.power_screen import (
    K_EVENTS,
    NO_AB,
    SWING_DESC,
    WHIFF_DESC,
    WOBA_W,
)

log = logging.getLogger("reliability")

GRID = (15, 25, 40, 60, 90, 130, 180, 250, 350)
TARGETS = (0.50, 0.70)


PA_KEY = ["batter", "game_date", "inning", "inning_topbot"]


def _pa_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per plate appearance, with everything the metrics need.

    Statcast is pitch-level and the engine's cache keeps neither the at-bat
    number nor the pitch number, so a plate appearance is keyed by hitter,
    date, inning and half-inning -- unique except for the rare inning a hitter
    bats twice, where the group's swing aggregates are shared between his turns.
    Terminal values (the batted ball, the event) come off the pitch that carried
    them; swing aggregates are summed over the pitches of the group.
    """
    swing = df["description"].isin(SWING_DESC)
    df = df.assign(
        _swing=swing,
        _whiff=df["description"].isin(WHIFF_DESC),
        _ozone=df["zone"] > 9,
        _ozone_swing=swing & (df["zone"] > 9),
    )
    grouped = df.groupby(PA_KEY, sort=False)
    agg = grouped.agg(
        pa=("events", lambda s: int(s.notna().sum())),
        events=("events", "last"),
        swings=("_swing", "sum"),
        whiffs=("_whiff", "sum"),
        ozone=("_ozone", "sum"),
        ozone_swings=("_ozone_swing", "sum"),
        bat_speed=("bat_speed", "mean"),
        launch_speed=("launch_speed", "last"),
        launch_speed_angle=("launch_speed_angle", "last"),
        xba=("estimated_ba_using_speedangle", "last"),
        xwoba=("estimated_woba_using_speedangle", "last"),
    ).reset_index()
    out = agg[agg["events"].notna() & agg["pa"].ge(1)].copy()
    share = out["pa"].clip(lower=1)
    for col in ("swings", "whiffs", "ozone", "ozone_swings"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0) / share
    for col in ("launch_speed", "launch_speed_angle", "xba", "xwoba", "bat_speed"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _rate(pa: pd.DataFrame, metric: str) -> float:
    """One metric on one bucket of plate appearances."""
    n = len(pa)
    if not n:
        return math.nan
    ev = pa["events"]
    ab = float(sum(1 for e in ev if e not in NO_AB))
    singles = float(ev.eq("single").sum())
    doubles = float(ev.eq("double").sum())
    triples = float(ev.eq("triple").sum())
    hr = float(ev.eq("home_run").sum())
    bb = float(ev.eq("walk").sum())
    hbp = float(ev.eq("hit_by_pitch").sum())
    hits = singles + doubles + triples + hr
    tb = singles + 2 * doubles + 3 * triples + 4 * hr
    ls = pa["launch_speed"].dropna()
    lsa = pa["launch_speed_angle"].dropna()

    if metric == "wrc":  # wOBA, which is what the screen's wRC+ is a rescaling of
        return (
            WOBA_W["walk"] * bb + WOBA_W["hit_by_pitch"] * hbp + WOBA_W["single"] * singles
            + WOBA_W["double"] * doubles + WOBA_W["triple"] * triples + WOBA_W["home_run"] * hr
        ) / n
    if metric == "ops":
        obp = (hits + bb + hbp) / n
        return obp + (tb / ab if ab else math.nan)
    if metric == "ba":
        return hits / ab if ab else math.nan
    if metric == "slg":
        return tb / ab if ab else math.nan
    if metric == "xba":
        x = pa["xba"].dropna()
        return float(x.sum() / ab) if ab and len(x) else math.nan
    if metric == "xslg":  # the screen's implied expected bases, not Savant's xSLG
        x = pa["xwoba"].dropna()
        return float(x.sum() / WOBA_W["single"] * 1.55 / ab) if ab and len(x) else math.nan
    if metric == "xwoba_con":
        x = pa["xwoba"].dropna()
        return float(x.mean()) if len(x) else math.nan
    if metric == "brl":
        return float((lsa == 6).mean()) if len(lsa) else math.nan
    if metric == "hh":
        return float((ls >= 95).mean()) if len(ls) else math.nan
    if metric == "ev90":
        return float(ls.quantile(0.90)) if len(ls) >= 5 else math.nan
    if metric == "osw":
        oz = float(pa["ozone"].sum())
        return float(pa["ozone_swings"].sum()) / oz if oz else math.nan
    if metric == "k":
        return float(ev.isin(K_EVENTS).mean())
    if metric == "contact":
        sw = float(pa["swings"].sum())
        return 1.0 - float(pa["whiffs"].sum()) / sw if sw else math.nan
    if metric == "bat_speed":
        b = pa["bat_speed"].dropna()
        return float(b.mean()) if len(b) else math.nan
    raise ValueError(metric)


METRICS = (
    "bat_speed", "contact", "k", "hh", "brl", "ev90", "osw",
    "xba", "xslg", "xwoba_con", "ba", "slg", "ops", "wrc",
)


def split_half(pa: pd.DataFrame, metric: str, n: int, reps: int, rng: np.random.Generator) -> float:
    """Reliability of ``metric`` at ``n`` PA, averaged over ``reps`` shuffles."""
    eligible = [g for _, g in pa.groupby("batter", sort=False) if len(g) >= 2 * n]
    if len(eligible) < 15:
        return math.nan
    rs: list[float] = []
    for _ in range(reps):
        a: list[float] = []
        b: list[float] = []
        for g in eligible:
            idx = rng.permutation(len(g))[: 2 * n]
            va = _rate(g.iloc[np.sort(idx[:n])], metric)
            vb = _rate(g.iloc[np.sort(idx[n:])], metric)
            if not (math.isnan(va) or math.isnan(vb)):
                a.append(va)
                b.append(vb)
        if len(a) >= 15 and np.std(a) > 0 and np.std(b) > 0:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(rs)) if rs else math.nan


def crossing(curve: dict[int, float], target: float) -> float:
    """PA at which reliability first reaches ``target``, linearly interpolated."""
    pts = sorted((n, r) for n, r in curve.items() if not math.isnan(r))
    for (n0, r0), (n1, r1) in zip(pts, pts[1:], strict=False):
        if r0 < target <= r1:
            return n0 + (target - r0) / (r1 - r0) * (n1 - n0)
    if pts and pts[0][1] >= target:
        return float(pts[0][0])
    return math.inf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", type=Date.fromisoformat, default=Date.today())
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config()
    repo = StatcastRepository(cfg.cache_dir)
    raw = repo.load_trailing(args.date, args.days)
    log.info("window: %d pitches", len(raw))
    pa = _pa_frame(raw)
    log.info("plate appearances: %d over %d hitters", len(pa), pa["batter"].nunique())

    rng = np.random.default_rng(args.seed)
    table: dict[str, dict[str, float]] = {}
    print(f"\n{'metric':<12}" + "".join(f"{n:>7}" for n in GRID) + "   PA@.50  PA@.70")
    for metric in METRICS:
        curve = {n: split_half(pa, metric, n, args.reps, rng) for n in GRID}
        row = "".join(
            "      -" if math.isnan(curve[n]) else f"{curve[n]:>7.2f}" for n in GRID
        )
        c50, c70 = crossing(curve, 0.50), crossing(curve, 0.70)
        table[metric] = {"pa_50": c50, "pa_70": c70}
        print(f"{metric:<12}{row}   {c50:>7.0f} {c70:>7.0f}")
    print("\n" + json.dumps(table, indent=2, default=str))


if __name__ == "__main__":
    main()
