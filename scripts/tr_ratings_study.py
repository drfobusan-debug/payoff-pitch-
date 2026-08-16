"""Do TeamRankings' luck and consistency ratings explain any of our error?

Their picks are a benchmark; these two ratings are the only part of their site
that is not either already in our model or already inside the market price we
anchor to:

* **consistency** is game-to-game variability (lower is steadier). We model a
  team's mean run scoring, never the spread of it -- and the spread is what a
  total and a run line are priced off, since a volatile club both blows out and
  gets blown out.
* **luck** is wins above what the run differential implies -- an outside reading
  of the season luck gap ``features.team_form`` scaffolds but has never trusted.

The test is the one that has condemned every signal we have measured properly,
including our own edge: conditional on the market price, does it carry anything?

    logit(win) ~ a + b*logit(model) + c*logit(market) + d*rating

fitted on graded game-market rows the engine actually priced. A rating that only
looks good on hit rate is a rating that is agreeing with the favourite; the test
that matters is whether it survives *next to* the price. For totals the rating
enters as the two clubs' combined consistency, because a total is a property of
the game, not of one side.

Nothing here changes a price. It reports what it can and, below the sample where
that is honest, says so and stops.

Usage::

    python scripts/tr_ratings_study.py                    # every captured day
    python scripts/tr_ratings_study.py --since 2026-08-16

Ratings are as-of-today with no archive, so the sample can only start on the
first day ``mlb-engine teamrankings`` stored them, and grows by one slate a day.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mlb_engine.audit.grade import WIN
from mlb_engine.audit.ledger import ENGINE, load_ledger
from mlb_engine.config import load_config
from mlb_engine.data.teamrankings import TeamRating, load_ratings

_GAME_MARKETS = ("game_total", "game_rl", "game_ml")
# Below this a coefficient is noise dressed as a finding. A game market is close
# to a coin flip, so the standard error on 100 rows swamps any real effect.
_MIN_ROWS = 150


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _ratings_by_day(audit_dir: Path, since: str | None) -> dict[str, dict[str, TeamRating]]:
    out: dict[str, dict[str, TeamRating]] = {}
    for path in sorted(audit_dir.glob("tr_ratings_*.json")):
        day = path.stem.replace("tr_ratings_", "")
        if since and day < since:
            continue
        rows = load_ratings(path)
        if rows:
            out[day] = {r.team: r for r in rows}
    return out


def _teams(matchup: str) -> tuple[str, str] | None:
    away, _, home = matchup.partition(" @ ")
    return (away.strip(), home.strip()) if away and home else None


def _fit(rows: list[tuple[float, float, float, int]]) -> tuple[float, float] | None:
    """Newton fit of ``win ~ model + market + rating``; returns (coef, se) on rating.

    Deliberately not sklearn: the engine already depends on it, but a 3-feature
    logit written out is auditable, and the standard error is the whole point of
    running this at all.
    """
    n, k = len(rows), 4
    beta = [0.0] * k
    for _ in range(60):
        grad = [0.0] * k
        hess = [[0.0] * k for _ in range(k)]
        for model, market, rating, win in rows:
            x = [1.0, model, market, rating]
            eta = sum(b * xi for b, xi in zip(beta, x, strict=True))
            p = 1.0 / (1.0 + math.exp(-max(min(eta, 30.0), -30.0)))
            w = max(p * (1 - p), 1e-9)
            for i in range(k):
                grad[i] += (win - p) * x[i]
                for j in range(k):
                    hess[i][j] += w * x[i] * x[j]
        step = _solve(hess, grad)
        if step is None:
            return None
        beta = [b + s for b, s in zip(beta, step, strict=True)]
        if max(abs(s) for s in step) < 1e-8:
            break
    cov = _invert(hess)
    if cov is None or cov[3][3] <= 0 or n <= k:
        return None
    return beta[3], math.sqrt(cov[3][3])


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    inv = _invert(a)
    if inv is None:
        return None
    return [sum(inv[i][j] * b[j] for j in range(len(b))) for i in range(len(b))]


def _invert(a: list[list[float]]) -> list[list[float]] | None:
    n = len(a)
    m = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for r in range(n):
            if r != col and m[r][col]:
                factor = m[r][col]
                m[r] = [v - factor * w for v, w in zip(m[r], m[col], strict=True)]
    return [row[n:] for row in m]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", help="ignore captures before this date (YYYY-MM-DD)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    cfg = load_config()
    ratings = _ratings_by_day(cfg.audit_dir, args.since)
    if not ratings:
        print(
            "No TeamRankings ratings captured yet. They are stored by "
            "`mlb-engine teamrankings` and cannot be backfilled -- the sample "
            "starts the first night it runs."
        )
        return 1

    entries = [
        e
        for e in load_ledger(cfg.audit_dir / "ledger.csv")
        if e.source == ENGINE
        and e.market in _GAME_MARKETS
        and e.date in ratings
        and e.result in ("win", "loss")
        and e.model_prob
        and e.fair_prob
    ]
    days = sorted({e.date for e in entries})
    report: dict[str, object] = {"days": len(days), "rows": len(entries)}
    print(f"TeamRankings ratings: {len(ratings)} days captured, {len(entries)} graded engine rows")

    for market in _GAME_MARKETS:
        for name in ("consistency", "luck"):
            rows: list[tuple[float, float, float, int]] = []
            for e in (x for x in entries if x.market == market):
                teams = _teams(e.matchup)
                if teams is None:
                    continue
                day = ratings[e.date]
                sides = [day.get(t) for t in teams]
                values = [getattr(s, name) for s in sides if s is not None]
                if len(values) != 2 or any(v is None for v in values):
                    continue
                # A total is a property of the game, so both clubs count; a side
                # market is about the team backed, which is the one we can name
                # only for the run line and money line.
                figure = sum(v for v in values if v is not None)
                rows.append(
                    (
                        _logit(e.model_prob),
                        _logit(e.fair_prob),
                        figure,
                        1 if e.result == WIN else 0,
                    )
                )
            key = f"{market}.{name}"
            if len(rows) < _MIN_ROWS:
                print(f"  {key:<24} {len(rows):>4} rows -- too few to read (need {_MIN_ROWS})")
                report[key] = {"rows": len(rows), "verdict": "insufficient"}
                continue
            fit = _fit(rows)
            if fit is None:
                print(f"  {key:<24} {len(rows):>4} rows -- fit did not converge")
                report[key] = {"rows": len(rows), "verdict": "no-fit"}
                continue
            coef, se = fit
            sigma = coef / se if se else 0.0
            verdict = "adds nothing" if abs(sigma) < 2 else "carries information"
            print(f"  {key:<24} {len(rows):>4} rows  coef {coef:+.4f}  {sigma:+.2f} SE  {verdict}")
            report[key] = {"rows": len(rows), "coef": coef, "se": se, "sigma": sigma}

    if args.json:
        print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
