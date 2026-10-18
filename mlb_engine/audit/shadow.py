"""The shadow: the simulator pinned to the market, read where the market is not.

Reads only what is already on disk -- the quote tape, the raw histograms, the
predictions file and the worksheet/totals receipts -- and writes two ledgers:
``shadow_ledger.csv`` (one row per pinned main-line market) and
``crossbook_ledger.csv`` (a prop read at a *different* two-sided line another
book posted). It makes no request, classifies nothing and buys nothing; the
live card never reads either file. Gaps, ``SUM`` and the old anchor weights ride
along as receipts at weight zero.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from datetime import date as Date
from pathlib import Path
from typing import Any

import numpy as np

from mlb_engine.audit import histograms as hg
from mlb_engine.audit import tilt
from mlb_engine.audit.clv import close_from_tape
from mlb_engine.audit.tape import TapeRow, load_tape, tape_path
from mlb_engine.market import keys
from mlb_engine.market.ev import anchor_to_market
from mlb_engine.recommendations import Recommendation, load_json

SHADOW_LEDGER = "shadow_ledger.csv"
CROSSBOOK_LEDGER = "crossbook_ledger.csv"
HIST_STALE = "hist_stale"
NO_HIST = "no_hist"
NO_MARKET = "no_market"
# The old anchor weights, kept as named leftovers and scored -- never shipped.
HYP_WEIGHTS = {
    "game_total": ("hyp_total_078", 0.78),
    "pitcher_bb": ("hyp_pbb_050", 0.50),
    "pitcher_h": ("hyp_ph_081", 0.81),
    "batter_2b": ("hyp_2b_061", 0.61),
}
SHADOW_FIELDS = [
    "date", "game_key", "event_id", "game_pk", "matchup", "market", "selection", "line",
    "side", "main_line", "american", "book", "fair_prob", "raw_sim_prob", "live_model_prob",
    "pin_status", "solver", "lam", "alpha", "beta", "alpha_newton", "beta_newton",
    "alpha_lsq", "beta_lsq", "n_landed", "residual_1", "residual_2", "ess", "tail_paths",
    "thin", "p_tilt_1pass", "total_drift_1pass", "p_tilt_ipf", "total_drift_ipf",
    "rl_residual", "ml_holdout_gap", "total_holdout_gap", "total_holdout_status",
    "f5_3way_home", "f5_3way_tie", "hyp_total_078", "hyp_pbb_050", "hyp_ph_081",
    "hyp_2b_061", "worksheet_gap", "worksheet_side_fair", "totals_sum", "sim_median_total",
    "posted_total", "result", "close_prob", "clv_quarantine",
]
SHADOW_KEY = ("date", "game_key", "market", "selection", "line")
CROSSBOOK_FIELDS = [
    "date", "game_key", "event_id", "game_pk", "matchup", "market", "player_id", "stat",
    "pin_line", "pin_fair", "read_book", "read_line", "read_side", "read_fair",
    "p_tilt_1pass", "p_tilt_ipf", "raw_sim_prob", "ess", "tail_paths", "result",
]
CROSSBOOK_KEY = ("date", "game_key", "market", "player_id", "pin_line", "read_book", "read_line")


def _f(x: float | None, nd: int = 6) -> str:
    return "" if x is None or not np.isfinite(x) else f"{round(float(x), nd)}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_ledger(
    path: Path, fields: list[str], rows: list[dict[str, Any]], iso: str
) -> None:
    """Replace this date's rows (re-run safe) and keep every other date as it was."""
    kept = [r for r in _read_csv(path) if r.get("date") != iso]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in [*kept, *rows]:
            w.writerow({k: r.get(k, "") for k in fields})


def _fair(rows: list[TapeRow]) -> tuple[float | None, float | None, str]:
    tc = close_from_tape(rows)
    if tc.quote is None:
        return None, None, tc.reason or NO_MARKET
    return tc.quote.no_vig_prob, tc.quote.american, ""


def _main_line(by_line: dict[float, list[TapeRow]]) -> float | None:
    """The line the most books hang, two-sided; ties go to the price nearest even."""
    paired = {ln: rs for ln, rs in by_line.items() if any(r.opposite_american is not None for r in rs)}
    if not paired:
        return None
    def score(ln: float) -> tuple[int, float]:
        fair = _fair(paired[ln])[0]
        return (-len({r.book for r in paired[ln]}), abs((fair or 0.0) - 0.5))
    return min(paired, key=score)


def _game_rows(rows: list[TapeRow], market: str, side: str) -> dict[float, list[TapeRow]]:
    out: dict[float, list[TapeRow]] = {}
    for r in rows:
        if r.market == market and r.side == side and r.line is not None:
            out.setdefault(r.line, []).append(r)
    return out


def _sel_rows(rows: list[TapeRow], market: str, selection: str) -> list[TapeRow]:
    return [r for r in rows if r.market == market and r.selection == selection]


def run_shadow(audit_dir: Path, slate_date: Date) -> tuple[int, int]:
    """Write the date's shadow and cross-book rows; returns how many of each."""
    iso = slate_date.isoformat()
    tape_rows = [
        r for r in load_tape(tape_path(audit_dir, slate_date))
        if r.hours_to_first_pitch is not None and r.hours_to_first_pitch > 0
    ]
    hists = hg.load_histograms(hg.hist_path(audit_dir, slate_date))
    pred_path = audit_dir / f"predictions_{iso}.json"
    recs = load_json(pred_path) if pred_path.exists() else []
    hashes = hg.pred_hashes(recs)
    by_game: dict[int, list[Recommendation]] = {}
    for rec in recs:
        by_game.setdefault(rec.game_pk, []).append(rec)
    worksheet = {r["game_pk"]: r for r in _read_csv(audit_dir / "worksheet_ledger.csv") if r.get("date") == iso}
    totals = {r["game_pk"]: r for r in _read_csv(audit_dir / "totals_ledger.csv") if r.get("date") == iso}
    tape_by_key: dict[str, list[TapeRow]] = {}
    for row in tape_rows:
        tape_by_key.setdefault(row.game_key, []).append(row)

    shadow: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    for gk, entry in sorted(hists.items()):
        pk = int(entry.get("game_pk") or 0)
        base = {"date": iso, "game_key": gk, "event_id": entry.get("event_id") or "",
                "game_pk": pk, "matchup": entry.get("matchup", "")}
        if hashes.get(pk) != entry.get("pred_hash"):
            shadow.append({**base, "market": "game", "selection": "", "line": "",
                           "pin_status": HIST_STALE})
            continue
        s_rows, c_rows = _shadow_game(base, entry, tape_by_key.get(gk, []), by_game.get(pk, []),
                                      worksheet.get(str(pk)), totals.get(str(pk)))
        shadow.extend(s_rows)
        cross.extend(c_rows)
    _write_ledger(audit_dir / SHADOW_LEDGER, SHADOW_FIELDS, shadow, iso)
    _write_ledger(audit_dir / CROSSBOOK_LEDGER, CROSSBOOK_FIELDS, cross, iso)
    return len(shadow), len(cross)


def _shadow_game(
    base: dict[str, Any],
    entry: dict[str, Any],
    rows: list[TapeRow],
    recs: Iterable[Recommendation],
    ws: dict[str, str] | None,
    tot: dict[str, str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts = hg.dense(entry["full"])
    p = counts / counts.sum()
    s, m = tilt.grid_axes(p.shape)
    matchup = str(base["matchup"])
    home = matchup.split(" @ ")[-1]
    live = {(r.market, r.selection): r.model_prob for r in recs}
    canon = {(r.market, keys.canonical(r.selection)): r for r in recs}
    receipts = {
        "worksheet_gap": (ws or {}).get("gap", ""),
        "worksheet_side_fair": (ws or {}).get("fav_implied", ""),
        "totals_sum": (tot or {}).get("sum_pts", ""),
        "posted_total": (tot or {}).get("line", ""),
        "sim_median_total": _f(float(np.median(np.repeat(s.ravel(), counts.ravel().astype(int)))), 2),
    }
    out: list[dict[str, Any]] = []

    tot_line = _main_line(_game_rows(rows, "game_total", "over"))
    tot_sel = keys.game_total(True, tot_line) if tot_line is not None else ""
    over_fair, over_am, _ = _fair(_sel_rows(rows, "game_total", tot_sel)) if tot_sel else (None, None, "")
    ml_sel = keys.game_ml(home)
    ml_fair, ml_am, _ = _fair(_sel_rows(rows, "game_ml", ml_sel))
    rl_line = _main_line(_game_rows(rows, "game_rl", home))
    rl_sel = keys.game_rl(home, rl_line) if rl_line is not None else ""
    rl_fair, rl_am, _ = _fair(_sel_rows(rows, "game_rl", rl_sel)) if rl_sel else (None, None, "")

    if tot_line is None or over_fair is None or ml_fair is None:
        out.append({**base, **receipts, "market": "game", "selection": "", "line": "",
                    "pin_status": NO_MARKET})
        return out, []
    read = tilt.read_game(p, tot_line, over_fair, ml_fair, rl_line, rl_fair)
    pin = read.pin
    q = pin.q
    w = np.where(p > 0, (q if q is not None else p) / np.where(p > 0, p, 1.0), 0.0)
    game_ess = tilt.ess(counts, w) if q is not None else None
    pin_cols = {
        "pin_status": pin.status, "solver": pin.solver, "alpha": _f(pin.alpha),
        "beta": _f(pin.beta), "alpha_newton": _f(pin.alpha_newton),
        "beta_newton": _f(pin.beta_newton), "alpha_lsq": _f(pin.alpha_lsq),
        "beta_lsq": _f(pin.beta_lsq), "n_landed": pin.n_landed,
        "residual_1": _f(pin.residual_1, 8), "residual_2": _f(pin.residual_2, 8),
        "ess": _f(game_ess, 1), "rl_residual": _f(read.rl_residual),
        "ml_holdout_gap": _f(read.ml_holdout_gap),
        "total_holdout_gap": _f(read.total_holdout_gap),
        "total_holdout_status": read.total_holdout_status,
    }
    game_mk: list[tuple[str, str, float | None, str, float, float | None, tilt.TwoWay, int]] = [
        ("game_total", tot_sel, tot_line, "over", over_fair, over_am,
         tilt.total_market(p.shape, tot_line, over_fair), int(counts[s > tot_line].sum())),
        ("game_ml", ml_sel, None, home, ml_fair, ml_am,
         tilt.ml_market(p.shape, ml_fair), int(counts[m > 0].sum())),
    ]
    if rl_line is not None and rl_fair is not None:
        game_mk.append(("game_rl", rl_sel, rl_line, home, rl_fair, rl_am,
                        tilt.rl_market(p.shape, rl_line, rl_fair),
                        int(counts[m + rl_line > 0].sum())))
    for market, sel, line, side, fair, am, tw, tail in game_mk:
        raw = tw.prob(p)
        row = {**base, **receipts, **pin_cols, "market": market, "selection": sel,
               "line": _f(line, 1), "side": side, "main_line": _f(line, 1),
               "american": _f(am, 0), "book": "consensus", "fair_prob": _f(fair),
               "raw_sim_prob": _f(raw), "live_model_prob": _f(live.get((market, sel))),
               "tail_paths": tail,
               "p_tilt_1pass": _f(tw.prob(q)) if q is not None else ""}
        if market in HYP_WEIGHTS:
            col, wt = HYP_WEIGHTS[market]
            row[col] = _f(anchor_to_market(raw, fair, wt))
        out.append(row)
    f5 = hg.dense(entry["f5"])
    fh, fa = np.indices(f5.shape)
    out[0]["f5_3way_home"] = _f(float(f5[fh > fa].sum() / f5.sum()))
    out[0]["f5_3way_tie"] = _f(float(f5[fh == fa].sum() / f5.sum()))

    cross: list[dict[str, Any]] = []
    if q is None:
        return out, cross
    p_tilt_total = np.bincount(s.ravel().astype(int), weights=q.ravel())
    props: dict[tuple[str, str], list[TapeRow]] = {}
    for r in rows:
        if r.market.startswith(("batter_", "pitcher_")) and r.side == "over":
            props.setdefault((r.market, keys.player_from_selection(r.selection)), []).append(r)
    for (market, _player), prow in sorted(props.items()):
        by_line: dict[float, list[TapeRow]] = {}
        for r in prow:
            if r.line is not None:
                by_line.setdefault(r.line, []).append(r)
        main = _main_line(by_line)
        if main is None:
            continue
        pin_rows = by_line[main]
        sel = pin_rows[0].selection
        rec = canon.get((market, keys.canonical(sel)))
        if rec is None or rec.player_id is None or rec.stat is None:
            continue
        hkey = hg.prop_key(market.split("_")[0], rec.player_id, rec.stat)
        if hkey not in entry["props"]:
            continue
        kc = hg.dense(entry["props"][hkey])
        pfair, am, _ = _fair(pin_rows)
        if pfair is None:
            continue
        fair = pfair
        kv = np.arange(kc.shape[0], dtype=float)
        one = tilt.conditional_prop(kc, p_tilt_total, main, fair, tot_line)
        ipf = tilt.conditional_prop(kc, p_tilt_total, main, fair, tot_line, ipf_steps=5)
        raw = tilt.prop_over(kv, kc.sum(axis=1), main)
        row = {**base, **receipts, "market": market, "selection": sel, "line": _f(main, 1),
               "side": "over", "main_line": _f(main, 1), "american": _f(am, 0),
               "book": "consensus", "fair_prob": _f(fair), "raw_sim_prob": _f(raw),
               "live_model_prob": _f(live.get((market, sel)) or rec.model_prob),
               "pin_status": one.status, "lam": _f(one.lam), "ess": _f(one.ess, 1),
               "tail_paths": int(kc[kv > main].sum()),
               "p_tilt_1pass": _f(tilt.prop_over(kv, one.q, main)) if one.q is not None else "",
               "total_drift_1pass": _f(one.total_drift),
               "p_tilt_ipf": _f(tilt.prop_over(kv, ipf.q, main)) if ipf.q is not None else "",
               "total_drift_ipf": _f(ipf.total_drift)}
        if market in HYP_WEIGHTS:
            col, wt = HYP_WEIGHTS[market]
            row[col] = _f(anchor_to_market(raw, fair, wt))
        out.append(row)
        # A second book's *different* two-sided line is the only shape read; the
        # same line at another book is consensus minus that book and is not written.
        for read_line, rrows in sorted(by_line.items()):
            if read_line == main:
                continue
            for r in rrows:
                if r.opposite_american is None:
                    continue
                rf, _, _ = _fair([r])
                cross.append({
                    "date": base["date"], "game_key": base["game_key"],
                    "event_id": base["event_id"], "game_pk": base["game_pk"],
                    "matchup": matchup, "market": market, "player_id": rec.player_id,
                    "stat": rec.stat, "pin_line": _f(main, 1), "pin_fair": _f(fair),
                    "read_book": r.book, "read_line": _f(read_line, 1), "read_side": "over",
                    "read_fair": _f(rf),
                    "p_tilt_1pass": _f(tilt.prop_over(kv, one.q, read_line)) if one.q is not None else "",
                    "p_tilt_ipf": _f(tilt.prop_over(kv, ipf.q, read_line)) if ipf.q is not None else "",
                    "raw_sim_prob": _f(tilt.prop_over(kv, kc.sum(axis=1), read_line)),
                    "ess": _f(one.ess, 1), "tail_paths": int(kc[kv > read_line].sum()),
                })
    return out, cross


def status_counts(path: Path, iso: str) -> Counter[str]:
    return Counter(r.get("pin_status", "") for r in _read_csv(path) if r.get("date") == iso)
