"""Phase 3: grade the shadow and cross-book rows, and the morning block that reads them.

Grading writes only the ``result`` column of the two shadow ledgers. Nothing is
classified, staked or carried to the card. The block's last line is the gate.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from datetime import date as Date
from pathlib import Path

import numpy as np

from mlb_engine.audit import tilt
from mlb_engine.audit.cluster import clustered_mean_ci
from mlb_engine.audit.grade import PUSH, grade_batter, grade_pitcher
from mlb_engine.audit.shadow import (
    CROSSBOOK_FIELDS,
    CROSSBOOK_LEDGER,
    HIST_STALE,
    HYP_WEIGHTS,
    SHADOW_FIELDS,
    SHADOW_LEDGER,
)
from mlb_engine.data.results import GameResult
from mlb_engine.market import keys
from mlb_engine.recommendations import Recommendation

GATE_LINE = "Gate shut: no alternate keys on the tape. 200-row rule is a 2027 rule."
YES, NO = "yes", "no"


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _ou(actual: float, line: float) -> str:
    """``yes`` when the shadow's "yes" side (over / home / home covers) landed; pushes excluded."""
    return YES if actual > line else NO if actual < line else PUSH


def _game_result(row: Mapping[str, str], res: GameResult) -> str:
    market, line = row["market"], row.get("line", "")
    h, a = res.home_runs, res.away_runs
    if market == "game_total" and line:
        return _ou(h + a, float(line))
    if market == "game_ml":
        return _ou(h - a, 0.0)
    if market == "game_rl" and line:
        return _ou(h - a + float(line), 0.0)
    return ""


def _prop_result(res: GameResult, market: str, pid: int, stat: str, line: float) -> str:
    grader = grade_batter if market.startswith("batter_") else grade_pitcher
    g = grader(res, pid, stat, line, "over")
    if g is None:
        return "void"
    return PUSH if g == PUSH else YES if g == "win" else NO


def grade_shadow(
    audit_dir: Path,
    slate_date: Date,
    results: Mapping[int, GameResult | None],
    recs: list[Recommendation],
) -> tuple[int, int]:
    """Fill ``result`` on the date's shadow and cross-book rows; returns rows graded.

    A prop's name is resolved through ``keys.canonical`` to the prediction row's
    ``player_id``, and every read is graded on its own line -- a cross-book row
    at 6.5 is the stat against 6.5, never whether the pin's 5.5 won.
    """
    iso = slate_date.isoformat()
    canon = {
        (r.game_pk, r.market, keys.canonical(r.selection)): r
        for r in recs if r.player_id is not None and r.stat
    }
    n_s = n_c = 0
    rows = _read(audit_dir / SHADOW_LEDGER)
    for row in rows:
        if row.get("date") != iso or not row.get("game_pk"):
            continue
        res = results.get(int(row["game_pk"]))
        if res is None or not res.final:
            continue
        if row["market"].startswith(("batter_", "pitcher_")):
            rec = canon.get((int(row["game_pk"]), row["market"], keys.canonical(row["selection"])))
            if rec is None or rec.player_id is None or rec.stat is None or not row.get("line"):
                continue
            row["result"] = _prop_result(res, row["market"], rec.player_id, rec.stat, float(row["line"]))
        else:
            row["result"] = _game_result(row, res)
        n_s += bool(row["result"])
    _write(audit_dir / SHADOW_LEDGER, SHADOW_FIELDS, rows)
    cross = _read(audit_dir / CROSSBOOK_LEDGER)
    for row in cross:
        if row.get("date") != iso or not row.get("game_pk") or not row.get("player_id"):
            continue
        res = results.get(int(row["game_pk"]))
        if res is None or not res.final:
            continue
        row["result"] = _prop_result(
            res, row["market"], int(row["player_id"]), row["stat"], float(row["read_line"])
        )
        n_c += 1
    if cross:
        _write(audit_dir / CROSSBOOK_LEDGER, CROSSBOOK_FIELDS, cross)
    return n_s, n_c


def _brier(rows: list[dict[str, str]], col: str) -> tuple[float, int]:
    pairs = [(float(r[col]), r["result"] == YES) for r in rows
             if r.get("result") in (YES, NO) and r.get(col)]
    if not pairs:
        return float("nan"), 0
    p, y = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs], dtype=float)
    return float(np.mean((p - y) ** 2)), len(pairs)


def _diff_ci(rows: list[dict[str, str]], a: str, b: str) -> str:
    use = [r for r in rows if r.get("result") in (YES, NO) and r.get(a) and r.get(b)]
    if len(use) < 2:
        return "n/a"
    y = np.array([r["result"] == YES for r in use], dtype=float)
    d = (np.array([float(r[a]) for r in use]) - y) ** 2 - (np.array([float(r[b]) for r in use]) - y) ** 2
    lo, hi = clustered_mean_ci(d, [r["date"] for r in use])
    return f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"


def _fmt(rows: list[dict[str, str]], label: str, cols: tuple[str, ...]) -> str:
    parts = []
    n = 0
    for c in cols:
        b, n = _brier(rows, c)
        parts.append(f"{c}={b:.4f}" if n else f"{c}=n/a")
    return f"  {label}: n={n} Brier " + " ".join(parts)


def morning_block(audit_dir: Path) -> list[str]:
    """The shadow's lines for the morning report, cumulative over every graded date."""
    rows = _read(audit_dir / SHADOW_LEDGER)
    cross = _read(audit_dir / CROSSBOOK_LEDGER)
    out = ["Shadow (measurement only -- nothing here is a bet)"]
    games = {(r["date"], r["game_key"]): r for r in rows}
    status = [r.get("pin_status", "") for r in games.values()]
    out.append(
        f"  games: {len(games)}  pinned {status.count(tilt.PINNED)}  "
        f"unpinned {status.count(tilt.UNPINNED)}  stale {status.count(HIST_STALE)}"
    )
    rl = [r for r in rows if r["market"] == "game_rl"]
    out.append(_fmt(rl, "run-line read", ("p_tilt_1pass", "fair_prob", "raw_sim_prob")))
    out.append(f"    tilt - book Brier (date-clustered 95%): {_diff_ci(rl, 'p_tilt_1pass', 'fair_prob')}")
    res = [float(r["rl_residual"]) for r in games.values() if r.get("rl_residual")]
    if res:
        out.append(f"    mean run-line residual {np.mean(res):+.4f} (n={len(res)})")
    reads = [
        tilt.GameRead(_NO_PIN, float(r["rl_residual"]), float(r["ml_holdout_gap"]), None, "")
        for r in games.values() if r.get("rl_residual") and r.get("ml_holdout_gap")
    ]
    out.append(f"  moneyline-holdout sign violations: {tilt.sign_violations(reads)} (expected 0)")
    props = [r for r in rows if r["market"].startswith(("batter_", "pitcher_"))]
    out.append(_fmt(props, "props at the pin", ("p_tilt_1pass", "p_tilt_ipf", "fair_prob")))
    drift = [abs(float(r["total_drift_1pass"])) for r in props if r.get("total_drift_1pass")]
    if drift:
        out.append(f"    mean |total drift| one-pass {np.mean(drift):.4f}")
    out.append(_fmt(cross, "cross-book (different line)", ("p_tilt_1pass", "p_tilt_ipf", "read_fair", "raw_sim_prob")))
    out.append(f"    tilt - book Brier (date-clustered 95%): {_diff_ci(cross, 'p_tilt_1pass', 'read_fair')}")
    out.append("  -- under the fold: receipts at weight zero --")
    tot = [r for r in rows if r["market"] == "game_total"]
    out.append(_fmt(tot, "unshifted sim, totals", ("raw_sim_prob", "fair_prob")))
    for market, (col, w) in HYP_WEIGHTS.items():
        sub = [r for r in rows if r["market"] == market]
        b, n = _brier(sub, col)
        if n:
            out.append(f"  old weight {market} w={w}: n={n} Brier {b:.4f}")
    th = [float(r["total_holdout_gap"]) for r in games.values() if r.get("total_holdout_gap")]
    th_st = [r.get("total_holdout_status", "") for r in games.values() if r.get("total_holdout_status")]
    out.append(
        f"  total holdout: n={len(th)} mean gap {np.mean(th) if th else float('nan'):+.4f} "
        f"unpinned {th_st.count(tilt.UNPINNED)}/{len(th_st)}"
    )
    out.append(f"  worksheet gap rows: {sum(1 for r in tot if r.get('worksheet_gap'))}  "
               f"totals SUM rows: {sum(1 for r in tot if r.get('totals_sum'))}")
    out.append(GATE_LINE)
    return out


_NO_PIN = tilt.Pin2D(tilt.PINNED, None, None, "", None, None, None, None, 0, None, None)
