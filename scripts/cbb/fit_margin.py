"""Fit the v0 margin shape (spreads, moneylines, first halves) from last season's closes.

Input CSV columns: game_date, home_spread (home-side consensus close), close_total,
home_pts, away_pts, h1_home, h1_away (``cbb-engine history --export`` writes them).

    .venv/bin/python -m scripts.cbb.fit_margin closes_2024_25.csv --out cbb_engine/params/margin.json
    .venv/bin/python -m scripts.cbb.fit_margin closes_2024_25.csv --holdout 2025-01-25 --out /dev/null

* full game: SD of ``final margin − (−home_spread)`` per month (``margin.phase``), and the
  key-number ratio observed/expected of |final margin| where |z| > 3 (expected = the normal
  around each game's close with no mass at 0: overtime means no ties);
* moneyline: one multiplier on that SD, fitted on who won (``ml_scale``);
* first half: half margin = share × close margin, half total = a + b × close total (OLS),
  each residual SD, and half-margin key ratios with ties allowed.

``--holdout DATE`` fits on games before DATE and scores the games from DATE on. This
script is the only thing allowed to write ``margin.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, pstdev

from cbb_engine.models.margin import PARAMS, MarginWidth, p_side, p_total, phase, pmf

KEY_Z = 3.0
KEY_MAX = 30


def _games(rows: list[dict[str, str]]) -> list[dict[str, float | str | None]]:
    out: list[dict[str, float | str | None]] = []
    for r in rows:
        try:
            spread, hp, ap = float(r["home_spread"]), int(float(r["home_pts"])), int(float(r["away_pts"]))
        except (KeyError, ValueError):
            continue
        g: dict[str, float | str | None] = {"day": r["game_date"], "mu": -spread, "m": float(hp - ap), "t": None, "h1m": None, "h1t": None}
        try:
            g["t"] = float(r["close_total"])
        except (KeyError, ValueError):
            pass
        try:
            h, a = int(float(r["h1_home"])), int(float(r["h1_away"]))
            g["h1m"], g["h1t"] = float(h - a), float(h + a)
        except (KeyError, ValueError):
            pass
        out.append(g)
    return out


def _ratios(pairs: list[tuple[float, float, float]], *, tie: bool) -> dict[str, float]:
    """Observed/expected mass at each |k| (k ≤ KEY_MAX) where the binomial z clears KEY_Z."""
    n = len(pairs)
    seen: Counter[int] = Counter()
    expected: dict[int, float] = defaultdict(float)
    for mu, sd, obs in pairs:
        seen[abs(int(obs))] += 1
        for k, v in pmf(mu, sd, {}, tie=tie).items():
            if abs(k) <= KEY_MAX:
                expected[abs(k)] += v
    out = {}
    for k in sorted(expected):
        p = expected[k] / n
        if expected[k] < 5:
            continue
        z = (seen[k] / n - p) / math.sqrt(max(p * (1 - p), 1e-12) / n)
        if abs(z) > KEY_Z:
            out[str(k)] = round(seen[k] / expected[k], 4)
    return out


def _ols(xs: list[float], ys: list[float], *, intercept: bool) -> tuple[float, float]:
    if not intercept:
        return 0.0, sum(x * y for x, y in zip(xs, ys, strict=True)) / sum(x * x for x in xs)
    mx, my = mean(xs), mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def _ml_scale(games: list[dict[str, float | str | None]], sd_of: dict[str, float]) -> float:
    """The width multiplier that maximises the moneyline log likelihood: margins are fatter-tailed
    than normal, so who wins is tighter around the spread than the cover width says."""
    def ll(scale: float) -> float:
        tot = 0.0
        for g in games:
            mu, sd = float(g["mu"]), scale * sd_of[phase(str(g["day"]))]  # type: ignore[arg-type]
            tie = _cdf(0.5, mu, sd) - _cdf(-0.5, mu, sd)
            p = min(max((1 - _cdf(0.5, mu, sd)) / max(1 - tie, 1e-12), 1e-9), 1 - 1e-9)
            tot += math.log(p if float(g["m"]) > 0 else 1 - p)  # type: ignore[arg-type]
        return tot

    return max((round(0.70 + 0.01 * i, 2) for i in range(61)), key=ll)


def _cdf(x: float, mu: float, sd: float) -> float:
    return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))


def fit(rows: list[dict[str, str]], *, min_n: int = 200) -> dict[str, object]:
    games = _games(rows)
    if len(games) < min_n:
        raise SystemExit(f"only {len(games)} games; need {min_n} before a margin width is written")
    resid: dict[str, list[float]] = defaultdict(list)
    for g in games:
        r = float(g["m"]) - float(g["mu"])  # type: ignore[arg-type]
        resid["all"].append(r)
        resid[phase(str(g["day"]))].append(r)
    sd_by = {k: round(pstdev(v), 3) for k, v in resid.items() if len(v) >= min_n or k == "all"}
    sd_of = {k: sd_by.get(k, sd_by["all"]) for k in resid}
    key = _ratios([(float(g["mu"]), sd_of[phase(str(g["day"]))], float(g["m"])) for g in games], tie=False)  # type: ignore[arg-type]
    out: dict[str, object] = {
        "sd_by_phase": sd_by, "key_ratio": key, "ml_scale": _ml_scale(games, sd_of), "n": len(games), "fitted_on": date.today().isoformat(),
    }
    halves = [g for g in games if g["h1m"] is not None and g["t"] is not None]
    if len(halves) >= min_n:
        mus = [float(g["mu"]) for g in halves]  # type: ignore[arg-type]
        h1m = [float(g["h1m"]) for g in halves]  # type: ignore[arg-type]
        _, share = _ols(mus, h1m, intercept=False)
        m_sd = pstdev([y - share * x for x, y in zip(mus, h1m, strict=True)])
        tots = [float(g["t"]) for g in halves]  # type: ignore[arg-type]
        h1t = [float(g["h1t"]) for g in halves]  # type: ignore[arg-type]
        a, b = _ols(tots, h1t, intercept=True)
        t_sd = pstdev([y - (a + b * x) for x, y in zip(tots, h1t, strict=True)])
        out |= {
            "h1_margin_share": round(share, 4), "h1_margin_sd": round(m_sd, 3),
            "h1_key_ratio": _ratios([(share * x, m_sd, y) for x, y in zip(mus, h1m, strict=True)], tie=True),
            "h1_total_a": round(a, 3), "h1_total_b": round(b, 4), "h1_total_sd": round(t_sd, 3), "h1_n": len(halves),
        }
    return out


def _mw(fitted: dict[str, object]) -> MarginWidth:
    return MarginWidth(
        sd_by_phase=dict(fitted["sd_by_phase"]),  # type: ignore[call-overload]
        key_ratio={int(k): v for k, v in dict(fitted["key_ratio"]).items()},  # type: ignore[call-overload]
        ml_scale=float(fitted.get("ml_scale", 1.0)),  # type: ignore[arg-type]
        h1_margin_share=float(fitted.get("h1_margin_share", 0.5)),  # type: ignore[arg-type]
        h1_margin_sd=float(fitted.get("h1_margin_sd", 0.0)),  # type: ignore[arg-type]
        h1_key_ratio={int(k): v for k, v in dict(fitted.get("h1_key_ratio", {})).items()},  # type: ignore[call-overload]
        h1_total_a=float(fitted.get("h1_total_a", 0.0)),  # type: ignore[arg-type]
        h1_total_b=float(fitted.get("h1_total_b", 0.5)),  # type: ignore[arg-type]
        h1_total_sd=float(fitted.get("h1_total_sd", 0.0)),  # type: ignore[arg-type]
    )


def holdout(rows: list[dict[str, str]], cut: str) -> dict[str, object]:
    """Fit before ``cut``; on games from ``cut`` on: calibration and log loss of the home-win and
    half-lead probabilities, and the half total's error, against simple baselines."""
    train = [r for r in rows if r.get("game_date", "") < cut]
    test = _games([r for r in rows if r.get("game_date", "") >= cut])
    mw = _mw(fit(train))
    full_single = MarginWidth(sd_by_phase={"all": mw.sd_by_phase["all"]})

    def score(ps: list[tuple[float, int]]) -> dict[str, object]:
        ps = [(min(max(p, 1e-6), 1 - 1e-6), y) for p, y in ps]
        bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for p, y in ps:
            bins[min(int(p * 10), 9)].append((p, y))
        cal = {
            f"{b * 10}-{b * 10 + 10}%": {"n": len(v), "pred": round(mean(p for p, _ in v), 3), "won": round(mean(y for _, y in v), 3)}
            for b, v in sorted(bins.items())
        }
        return {
            "n": len(ps),
            "log_loss": round(-mean(math.log(p if y else 1 - p) for p, y in ps), 4),
            "brier": round(mean((p - y) ** 2 for p, y in ps), 4),
            "calibration": cal,
        }

    def win_p(m: MarginWidth, g: dict[str, float | str | None]) -> float:
        sd = (m.sd(str(g["day"])) or m.sd_by_phase["all"]) * m.ml_scale
        return p_side(pmf(float(g["mu"]), sd, m.key_ratio, tie=False), "home", None) or 0.5  # type: ignore[arg-type]

    out: dict[str, object] = {
        "cut": cut, "train_n": len(train),
        "ml_fitted": score([(win_p(mw, g), int(float(g["m"]) > 0)) for g in test]),  # type: ignore[arg-type]
        "ml_one_width_no_keys": score([(win_p(full_single, g), int(float(g["m"]) > 0)) for g in test]),  # type: ignore[arg-type]
    }
    halves = [g for g in test if g["h1m"] is not None and g["t"] is not None and g["h1m"] != 0]
    if halves and mw.h1_margin_sd:
        lead = [
            (p_side(pmf(mw.h1_margin_share * float(g["mu"]), mw.h1_margin_sd, mw.h1_key_ratio, tie=True), "home", None) or 0.5,  # type: ignore[arg-type]
             int(float(g["h1m"]) > 0))  # type: ignore[arg-type]
            for g in halves
        ]
        half_share = [(p_side(pmf(0.5 * float(g["mu"]), mw.h1_margin_sd, {}, tie=True), "home", None) or 0.5, y)  # type: ignore[arg-type]
                      for g, (_, y) in zip(halves, lead, strict=True)]
        pred_t = [mw.h1_total_a + mw.h1_total_b * float(g["t"]) for g in halves]  # type: ignore[arg-type]
        err = [float(g["h1t"]) - p for g, p in zip(halves, pred_t, strict=True)]  # type: ignore[arg-type]
        err_half = [float(g["h1t"]) - 0.5 * float(g["t"]) for g in halves]  # type: ignore[arg-type]
        over = [(p_total(p, mw.h1_total_sd, "over", round(p * 2) / 2 + 0.5 if (p * 2) % 1 else p), int(float(g["h1t"]) > p)) for g, p in zip(halves, pred_t, strict=True)]  # type: ignore[arg-type]
        out |= {
            "h1_lead_fitted": score(lead),
            "h1_lead_half_share": score(half_share),
            "h1_total": {
                "n": len(halves), "bias_fitted": round(mean(err), 3), "rmse_fitted": round(math.sqrt(mean(e * e for e in err)), 3),
                "bias_half_of_full": round(mean(err_half), 3), "rmse_half_of_full": round(math.sqrt(mean(e * e for e in err_half)), 3),
                "over_rate_at_fitted": round(mean(y for _, y in over), 3),
            },
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--out", default=str(PARAMS))
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--holdout", help="YYYY-MM-DD: fit before, score from (writes nothing)")
    args = ap.parse_args(argv)
    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.holdout:
        print(json.dumps(holdout(rows, args.holdout), indent=2))
        return 0
    out = fit(rows, min_n=args.min_n)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
