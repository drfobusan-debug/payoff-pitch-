"""What a starter's delivery is worth once his luck term has been read.

The pitcher analogue of ``scripts/swing_stage_study.py``, and out of time in the
same way: every predictor is read strictly before an anchor date and the target is
the fortnight after it, which no predictor can see. ``features.arm`` stores what
this measures.

    python scripts/arm_stage_study.py --seasons 2025 2026          # levels/deltas
    python scripts/arm_stage_study.py --seasons 2025 2026 --cells   # the four cells

``--cells`` is the confirmatory question, and it is the reason this script exists
rather than only the regression table. Stage one flags an arm in one of two
directions -- results ran hot, so the report calls for a fade; or ran cold on a
high BABIP, so it calls for a correction -- and the delivery underneath either
agrees with that direction or argues against it. That is a 2x2 rather than a
rescue: what is being asked is whether the agreeing cells land further from the
unflagged base rate than the flag does on its own, in both directions.

Each cell is also run on the *trend* of perceived velocity beside its level,
because a falling arm is the intuitive form of the same story and the intuition
does not survive: those intervals cross zero, so nothing in ``features.arm`` is
allowed to read a delta. Intervals are bootstrap resamples of pitchers, not of
rows -- a starter appears at up to eighteen anchors.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date as Date
from datetime import timedelta

import numpy as np
import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.features.arm import (
    LEAGUE,
    MIN_LEVEL_PITCHES,
    WINDOW,
    fastballs_of,
)

log = logging.getLogger("arm-stage")

#: The physicals the panel carries. Perceived velocity is the verdict metric and
#: induced vertical break is tested beside it, never pooled into it.
METRICS = ("velo", "pvelo", "ext", "rel_x", "rel_z", "spin", "ivb", "hb")
TB_MAP = {"single": 1.0, "double": 2.0, "triple": 3.0, "home_run": 4.0}
TARGETS = (
    ("nxt_woba", "wOBA/PA"),
    ("nxt_k", "K/PA"),
    ("nxt_hit", "H/PA"),
    ("nxt_hr", "HR/PA"),
    ("nxt_tb", "TB/PA"),
)
HORIZON = 14  # days of the out-of-time target
FORM_DAYS = 42  # the trailing window stage one is read over
MIN_OUT_PA = 30
MIN_FORM_PA = 60
MIN_BABIP_DEN = 25
BL_BABIP = 0.290  # the league rate the regression report ranks BABIP against
FLAG = 0.030  # |xwOBA - wOBA| at which stage one calls a direction
BOOTS = 3000


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def physicals(df: pd.DataFrame) -> pd.DataFrame:
    """One row per tracked fastball, carrying every level the arm model reads."""
    fb = fastballs_of(df.assign(game_date=pd.to_datetime(df["game_date"]).dt.date))
    hand = fb["p_throws"].map({"L": -1.0, "R": 1.0}).fillna(1.0)
    velo = pd.to_numeric(fb["release_speed"], errors="coerce")
    ext = pd.to_numeric(fb["release_extension"], errors="coerce")
    return pd.DataFrame(
        {
            "game_date": fb["game_date"],
            "pitcher": fb["pitcher"],
            "velo": velo,
            "pvelo": velo + 1.1 * ext - 6.0,
            "ext": ext,
            "rel_x": pd.to_numeric(fb["release_pos_x"], errors="coerce") * -hand,
            "rel_z": pd.to_numeric(fb["release_pos_z"], errors="coerce"),
            "spin": pd.to_numeric(fb["release_spin_rate"], errors="coerce"),
            "ivb": pd.to_numeric(fb["pfx_z"], errors="coerce") * 12.0,
            "hb": pd.to_numeric(fb["pfx_x"], errors="coerce") * 12.0 * -hand,
        }
    ).sort_values(["pitcher", "game_date"])


def outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per plate appearance against the starter, graded."""
    pa = df[df["woba_denom"].notna()].copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"]).dt.date
    pa["tb"] = pa["events"].map(TB_MAP).fillna(0.0)
    pa["hit"] = (pa["tb"] > 0).astype(float)
    pa["hr"] = (pa["events"] == "home_run").astype(float)
    pa["k"] = pa["events"].isin(("strikeout", "strikeout_double_play")).astype(float)
    pa["woba"] = pa["woba_value"].fillna(0.0)
    pa["xwoba"] = pa["estimated_woba_using_speedangle"].fillna(pa["woba"])
    bip = pa["type"].eq("X")
    pa["babip_num"] = (bip & (pa["hit"] > 0) & (pa["hr"] == 0)).astype(float)
    pa["babip_den"] = (bip & (pa["hr"] == 0)).astype(float)
    return pa.sort_values(["pitcher", "game_date"])[
        [
            "game_date",
            "pitcher",
            "woba",
            "xwoba",
            "tb",
            "hit",
            "hr",
            "k",
            "babip_num",
            "babip_den",
        ]
    ]


def panel(repo: StatcastRepository, seasons: list[int]) -> pd.DataFrame:
    """One row per starter per anchor: the delivery before it, the fortnight after."""
    rows: list[dict[str, object]] = []
    for season in seasons:
        start = Date(season, 3, 1)
        end = min(Date(season, 11, 15), Date.today())
        df = repo.load_range(start, end)
        ph_by = dict(tuple(physicals(df).groupby("pitcher", sort=False)))
        pa_by = dict(tuple(outcomes(df).groupby("pitcher", sort=False)))
        del df
        anchor = start + timedelta(45)
        while anchor + timedelta(HORIZON) <= end:
            kept = 0
            for pid, pa in pa_by.items():
                ph = ph_by.get(pid)
                if ph is None:
                    continue
                past = ph[ph["game_date"] < anchor]
                fut = pa[
                    (pa["game_date"] >= anchor)
                    & (pa["game_date"] <= anchor + timedelta(HORIZON - 1))
                ]
                form = pa[
                    (pa["game_date"] >= anchor - timedelta(FORM_DAYS)) & (pa["game_date"] < anchor)
                ]
                if len(fut) < MIN_OUT_PA or len(form) < MIN_FORM_PA:
                    continue
                row: dict[str, object] = {
                    "season": season,
                    "anchor": anchor.isoformat(),
                    "pitcher": int(pid),
                }
                thin = False
                for metric in METRICS:
                    v = past[metric].dropna().to_numpy(dtype=float)
                    # Two windows: the level, and the block before it the delta
                    # is measured against. Below the floor the arm is unmeasured.
                    if len(v) < 2 * MIN_LEVEL_PITCHES:
                        thin = True
                        break
                    n = min(WINDOW, len(v) // 2)
                    row[f"lvl_{metric}"] = float(v[-n:].mean())
                    row[f"d_{metric}"] = float(v[-n:].mean() - v[-2 * n : -n].mean())
                if thin:
                    continue
                den = float(form["babip_den"].sum())
                if den < MIN_BABIP_DEN:
                    continue
                now_woba = float(form["woba"].mean())
                now_xwoba = float(form["xwoba"].mean())
                row["now_woba"] = now_woba
                row["now_xwoba"] = now_xwoba
                row["luck_gap"] = now_xwoba - now_woba
                row["now_babip"] = float(form["babip_num"].sum()) / den
                for key in ("woba", "tb", "hit", "hr", "k"):
                    row[f"nxt_{key}"] = float(fut[key].mean())
                rows.append(row)
                kept += 1
            log.info("%s anchor %s  n=%d", season, anchor, kept)
            anchor += timedelta(HORIZON)
    p = pd.DataFrame(rows)
    return p if p.empty else p.dropna().reset_index(drop=True)


def ols(y: np.ndarray, x: np.ndarray, cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares with standard errors clustered on ``cluster``."""
    xtx_inv = np.linalg.pinv(x.T @ x)
    b = xtx_inv @ x.T @ y
    e = y - x @ b
    meat = np.zeros((x.shape[1], x.shape[1]))
    for g in np.unique(cluster):
        m = cluster == g
        u = x[m].T @ e[m]
        meat += np.outer(u, u)
    groups = len(np.unique(cluster))
    adj = groups / (groups - 1) * (len(y) - 1) / (len(y) - x.shape[1])
    v = xtx_inv @ (meat * adj) @ xtx_inv
    r2 = 1.0 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
    return b, np.sqrt(np.diag(v)), r2


def bootstrap(
    a: pd.DataFrame, b: pd.DataFrame, col: str, rng: np.random.Generator, floor: int = 12
) -> tuple[float, float, float] | None:
    """``a`` minus ``b`` on ``col``, resampling pitchers rather than rows."""
    if len(a) < floor or len(b) < floor:
        return None
    both = pd.concat([a.assign(_side=1), b.assign(_side=0)])
    by = dict(tuple(both.groupby("pitcher", sort=False)))
    ids = np.array(list(by))
    diffs: list[float] = []
    for _ in range(BOOTS):
        draw = pd.concat([by[i] for i in rng.choice(ids, len(ids), replace=True)])
        left, right = draw[draw["_side"] == 1], draw[draw["_side"] == 0]
        if len(left) < floor or len(right) < floor:
            continue
        diffs.append(float(left[col].mean() - right[col].mean()))
    if not diffs:
        return None
    return (
        float(np.mean(diffs)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def _line(label: str, q: pd.DataFrame) -> None:
    if q.empty:
        print(f"  {label:<38} no rows")
        return
    cells = "  ".join(f"{lab} {q[col].mean():.4f}" for col, lab in TARGETS)
    print(f"  {label:<38} n {len(q):>4} | {cells}")


def levels_and_deltas(p: pd.DataFrame) -> None:
    """Which physicals survive stage one at all, as a level and as a trend."""
    cluster = p["pitcher"].to_numpy()
    base = np.column_stack([np.ones(len(p)), p["now_woba"], p["now_xwoba"], p["now_babip"]])
    print(
        f"panel: {len(p)} pitcher-windows, {p['pitcher'].nunique()} starters, "
        f"{p['anchor'].nunique()} anchors, level window {WINDOW} fastballs"
    )
    print("\nstage one alone -- the next fortnight on the trailing luck term")
    for col, label in TARGETS:
        b, se, r2 = ols(p[col].to_numpy(float), base, cluster)
        print(
            f"  {label:8s} R2 {r2:.4f} | wOBA {b[1]:+.3f} (t {b[1] / se[1]:+.2f})"
            f"  xwOBA {b[2]:+.3f} (t {b[2] / se[2]:+.2f})"
            f"  BABIP {b[3]:+.3f} (t {b[3] / se[3]:+.2f})"
        )
    for kind, prefix in (("level", "lvl_"), ("trend", "d_")):
        print(f"\nstage two by {kind}, each added on top of stage one")
        for col, label in TARGETS:
            y = p[col].to_numpy(float)
            _, _, r2b = ols(y, base, cluster)
            cells = []
            for metric in METRICS:
                x = np.column_stack([base, _z(p[f"{prefix}{metric}"])])
                b, se, r2 = ols(y, x, cluster)
                cells.append(f"{metric} t {b[4] / se[4]:+.2f} dR2 {r2 - r2b:+.5f}")
            print(f"  {label:8s} " + " | ".join(cells))


def cells(p: pd.DataFrame, rng: np.random.Generator) -> None:
    """The 2x2: does the delivery agree with the direction stage one called?"""
    p["pz"] = (p["lvl_pvelo"] - LEAGUE["pvelo"][0]) / LEAGUE["pvelo"][1]
    p["dz"] = _z(p["d_pvelo"])
    p["iz"] = (p["lvl_ivb"] - LEAGUE["ivb"][0]) / LEAGUE["ivb"][1]
    hot = p["luck_gap"] > FLAG  # deserved worse: the report calls for a fade
    cold = (p["luck_gap"] < -FLAG) & (p["now_babip"] > BL_BABIP)  # calls for a correction
    neither = ~hot & ~cold

    print("\n=== the four cells, delivery read as a level ===")
    _line("no flag either way (base rate)", p[neither])
    _line("ran hot, all", p[hot])
    _line("ran hot + below-league arm (confirm)", p[hot & (p["pz"] <= 0)])
    _line("ran hot + above-league arm (contra)", p[hot & (p["pz"] > 0)])
    _line("ran cold, all", p[cold])
    _line("ran cold + above-league arm (confirm)", p[cold & (p["pz"] > 0)])
    _line("ran cold + below-league arm (contra)", p[cold & (p["pz"] <= 0)])

    print("\nwhat the agreement is worth, pitchers resampled")
    for tag, side_a, side_b in (
        ("hot: weak arm minus strong", hot & (p["pz"] <= 0), hot & (p["pz"] > 0)),
        ("cold: strong arm minus weak", cold & (p["pz"] > 0), cold & (p["pz"] <= 0)),
        ("hot + weak arm minus unflagged", hot & (p["pz"] <= 0), neither),
        ("cold + strong arm minus unflagged", cold & (p["pz"] > 0), neither),
    ):
        for col, label in TARGETS:
            r = bootstrap(p[side_a], p[side_b], col, rng)
            if r is not None:
                print(f"  {tag:34s} {label:8s} {r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}]")

    print("\nthe same cells read as a trend -- the form that does not survive")
    _line("ran hot + falling pVelo", p[hot & (p["dz"] <= 0)])
    _line("ran hot + rising pVelo", p[hot & (p["dz"] > 0)])
    _line("ran cold + rising pVelo", p[cold & (p["dz"] > 0)])
    _line("ran cold + falling pVelo", p[cold & (p["dz"] <= 0)])
    for tag, side_a, side_b in (
        ("hot: falling minus rising", hot & (p["dz"] <= 0), hot & (p["dz"] > 0)),
        ("cold: rising minus falling", cold & (p["dz"] > 0), cold & (p["dz"] <= 0)),
        (
            "hot + weak level: falling minus rising",
            hot & (p["pz"] <= 0) & (p["dz"] <= 0),
            hot & (p["pz"] <= 0) & (p["dz"] > 0),
        ),
    ):
        r = bootstrap(p[side_a], p[side_b], "nxt_woba", rng)
        if r is not None:
            print(f"  {tag:40s} wOBA/PA {r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}]")

    print("\nride (IVB) in the same cells, which is why it is not the verdict metric")
    _line("ran hot + low ride", p[hot & (p["iz"] <= 0)])
    _line("ran hot + high ride", p[hot & (p["iz"] > 0)])

    print("\nby season, the two agreeing cells on wOBA/PA")
    for season, g in p.groupby("season"):
        g_hot = g[g["luck_gap"] > FLAG]
        g_cold = g[(g["luck_gap"] < -FLAG) & (g["now_babip"] > BL_BABIP)]
        weak = g_hot[g_hot["pz"] <= 0]["nxt_woba"]
        strong = g_cold[g_cold["pz"] > 0]["nxt_woba"]
        print(
            f"  {season}: hot+weak {weak.mean():.4f} (n {len(weak)}) | "
            f"cold+strong {strong.mean():.4f} (n {len(strong)}) | "
            f"unflagged {g[(g['luck_gap'].abs() <= FLAG)]['nxt_woba'].mean():.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--cells", action="store_true", help="the confirmatory 2x2")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    cfg = load_config()
    p = panel(StatcastRepository(cfg.cache_dir), args.seasons)
    if p.empty:
        raise SystemExit("no pitcher-windows: the cache does not cover these seasons")
    levels_and_deltas(p)
    if args.cells:
        cells(p, np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
