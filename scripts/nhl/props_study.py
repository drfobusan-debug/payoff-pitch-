"""SOG pseudo-line study (master plan §5.5 Phase-3 gate).

Walk-forward over a finished season: before each game, project a skater's SOG
mean from his log to date (``features.props.project_skater``, previous season
as the prior), set the pseudo-line at ``round(mean) + 0.5`` and score P(over)
from the negative binomial against the outcome. The comparison is the league
base rate of going over that same pseudo-line. Also fits the NB variance slope
(var = v x mean) and reports pooled per-60 league means by position.

    python scripts/nhl/props_study.py --season 2024 --logs ~/.nhl_engine/cache/gamelogs
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from nhl_engine.data.skaters import parse_skater_log
from nhl_engine.features import props as propf
from nhl_engine.models import props as propm


def _load(dir_: Path, season: int):
    out = {}
    for p in dir_.glob(f"*_{season}.json"):
        pid = int(p.stem.split("_")[0])
        try:
            data = json.loads(p.read_text())
        except ValueError:
            continue
        log = parse_skater_log(pid, data)
        if log:
            out[pid] = log
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--logs", type=Path, default=Path.home() / ".nhl_engine/cache/gamelogs")
    ap.add_argument("--positions", type=Path, default=None, help="json {player_id: 'F'|'D'}")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cur = _load(a.logs, a.season)
    prev = _load(a.logs, a.season - 1)
    positions: dict[int, str] = {}
    if a.positions and a.positions.exists():
        positions = {int(k): v for k, v in json.loads(a.positions.read_text()).items()}

    # Pooled means from the previous season (what the engine would know).
    league = propf.league_means([(positions.get(pid, "F"), log) for pid, log in prev.items()])

    preds: list[tuple[float, int, float]] = []  # (mean, sog, line)
    for pid, log in cur.items():
        pos = positions.get(pid, "F")
        plog = prev.get(pid, [])
        for i, g in enumerate(log):
            if i < 5:
                continue  # need a little current-season TOI to project
            proj = propf.project_skater(
                player_id=pid,
                name="",
                team=g.team,
                position=pos,
                current=log[:i],
                previous=plog,
                slate=g.game_date,
                league=league[pos],
            )
            mean = proj.mean("sog")
            line = math.floor(mean) + 0.5
            preds.append((mean, g.sog, line))

    means = np.array([m for m, _, _ in preds])
    sog = np.array([s for _, s, _ in preds])
    lines = np.array([ln for _, _, ln in preds])
    n = len(preds)

    # Variance slope: var(sog | mean bucket) regressed on mean bucket, through origin.
    bins = np.clip(np.round(means * 2) / 2, 0.5, 6.0)
    xs, ys, ws = [], [], []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() < 200:
            continue
        xs.append(float(means[m].mean()))
        ys.append(float(sog[m].var()))
        ws.append(int(m.sum()))
    xs_, ys_, ws_ = np.array(xs), np.array(ys), np.array(ws)
    v = float((ws_ * xs_ * ys_).sum() / (ws_ * xs_ * xs_).sum())

    def brier(p: np.ndarray, y: np.ndarray) -> float:
        return float(((p - y) ** 2).mean())

    y = (sog > lines).astype(float)
    p_model = np.array(
        [propm.over_under(propm.negbin_pmf(m, v), ln, "over").win for m, ln in zip(means, lines, strict=True)]
    )
    p_pois = np.array(
        [propm.over_under(propm.poisson_pmf(m), ln, "over").win for m, ln in zip(means, lines, strict=True)]
    )
    base: dict[float, float] = defaultdict(float)
    for ln in np.unique(lines):
        base[float(ln)] = float(y[lines == ln].mean())
    p_base = np.array([base[float(ln)] for ln in lines])

    b_model, b_pois, b_base = brier(p_model, y), brier(p_pois, y), brier(p_base, y)
    # Paired bootstrap on the Brier gap.
    rng = np.random.default_rng(1)
    gaps = []
    d = (p_base - y) ** 2 - (p_model - y) ** 2
    for _ in range(400):
        idx = rng.integers(0, n, n)
        gaps.append(float(d[idx].mean()))
    lo, hi = np.percentile(gaps, [2.5, 97.5])

    # Calibration by model-prob decile.
    order = np.argsort(p_model)
    dec = []
    for chunk in np.array_split(order, 10):
        dec.append((float(p_model[chunk].mean()), float(y[chunk].mean()), int(len(chunk))))

    report = {
        "season": a.season,
        "player_games": n,
        "players": len(cur),
        "league_means_prev_season": league,
        "nb_var_slope": v,
        "var_fit_points": [(x, yy, int(w)) for x, yy, w in zip(xs_, ys_, ws_, strict=True)],
        "brier": {"nb": b_model, "poisson": b_pois, "base_rate": b_base},
        "brier_gain_vs_base": {"mean": b_base - b_model, "ci95": [0.0, 0.0]},
        "calibration_deciles": dec,
        "mean_sog_actual": float(sog.mean()),
        "mean_sog_projected": float(means.mean()),
    }
    report["brier_gain_vs_base"]["ci95"] = [float(lo), float(hi)]
    text = json.dumps(report, indent=2, default=float)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
