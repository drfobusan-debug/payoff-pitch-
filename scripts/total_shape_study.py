"""Ask what shape the simulator's run distribution has, not just where it sits.

Every over in the ledger realizes below the model and every under above it, in
every batter market, which reads like a run-environment level bias: if the
simulator scored a game too high, each over would inherit the error. This tests
that directly, by inverting the four posted game-total lines into the model's own
implied distribution for each game and comparing it to the final score.

The level is not the problem. What is left is the shape: real MLB totals are
right-skewed -- their mean sits above their median because a blowout has no
ceiling -- while the model's implied distribution is near-symmetric, so it prices
the median as if it were the mean and every over is dear by construction.

Nothing here changes a probability. It writes a table and a joined CSV.

    python scripts/total_shape_study.py [--ledger PATH] [--out PATH]

Box scores are cached under ``~/.mlb_engine/cache/boxscores``, the same place the
audit keeps them, so a second run costs no requests.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm, skew

log = logging.getLogger("total_shape")

BASE = "https://statsapi.mlb.com/api/v1"
LINES = (7.5, 8.5, 9.5, 10.5)
# A probability is clipped before it is inverted: the simulator reports 0 and 1
# for tails it never sampled, and an infinite quantile would carry one game's
# Monte-Carlo edge into the fitted mean for all of them.
CLIP = 0.005


def _schedule(date: str, cache: Path, session: requests.Session) -> list[int]:
    p = cache / f"schedule_{date}.json"
    if p.exists():
        return [int(x) for x in json.loads(p.read_text())]
    params = {"sportId": "1", "date": date}
    r = session.get(f"{BASE}/schedule", params=params, timeout=30).json()
    pks = [
        int(g["gamePk"])
        for d in r.get("dates", [])
        for g in d.get("games", [])
        if str(g.get("status", {}).get("abstractGameState", "")).lower() == "final"
    ]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pks))
    return pks


def _final(game_pk: int, cache: Path, session: requests.Session) -> tuple[str, int] | None:
    """``("AWY @ HOM", total runs)`` for a completed game, or None."""
    p = cache / f"{game_pk}.json"
    if p.exists():
        data = json.loads(p.read_text())
    else:
        data = {
            "linescore": session.get(f"{BASE}/game/{game_pk}/linescore", timeout=30).json(),
            "boxscore": session.get(f"{BASE}/game/{game_pk}/boxscore", timeout=30).json(),
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
    teams = (data.get("boxscore", {}) or {}).get("teams", {})
    innings = (data.get("linescore", {}) or {}).get("innings", [])
    if not innings or len(innings) < 9:
        return None
    abbr = {
        side: str(((teams.get(side, {}) or {}).get("team", {}) or {}).get("abbreviation", ""))
        for side in ("home", "away")
    }
    if not abbr["home"] or not abbr["away"]:
        return None
    total = sum(
        int((i.get(side, {}) or {}).get("runs", 0) or 0)
        for i in innings
        for side in ("home", "away")
    )
    return f"{abbr['away']} @ {abbr['home']}", total


def load_totals(ledger: Path) -> pd.DataFrame:
    """One row per game, the model's probability at each of the four over lines."""
    d = pd.read_csv(ledger, low_memory=False)
    if "source" in d.columns:
        d = d[(d["source"].isna()) | (d["source"] == "engine")]
    g = d[(d["market"] == "game_total") & (d["selection"].str.startswith("Over"))]
    wide = g.pivot_table(index=["date", "matchup"], columns="line", values="model_prob")
    return wide.reindex(columns=list(LINES)).dropna().reset_index()


def join_finals(games: pd.DataFrame, cache: Path) -> pd.DataFrame:
    session = requests.Session()
    finals: dict[tuple[str, str], int] = {}
    for date in sorted(games["date"].unique()):
        for pk in _schedule(str(date), cache, session):
            try:
                got = _final(pk, cache, session)
            except requests.RequestException as exc:
                log.warning("no box score for %s: %s", pk, exc)
                continue
            if got is not None:
                finals[(str(date), got[0])] = got[1]
        log.info("%s: %d finals", date, sum(1 for k in finals if k[0] == str(date)))
    games["total"] = [
        finals.get((str(d), str(m)), np.nan)
        for d, m in zip(games["date"], games["matchup"], strict=True)
    ]
    return games[games["total"].notna()].copy()


def fit_implied(games: pd.DataFrame) -> pd.DataFrame:
    """Least-squares mean and width of the normal the four probabilities imply.

    ``P(over line) = Phi((mu - line) / sd)``, so regressing the lines on their own
    inverted probabilities recovers both at once. The fit is read for its centre
    and its width only -- a normal cannot represent skew, which is the point: the
    residual between this and the realized distribution is the skew the model is
    missing.
    """
    lines = np.array(LINES)
    mus: list[float] = []
    sds: list[float] = []
    medians: list[float] = []
    for _, row in games.iterrows():
        probs = np.clip(np.array([float(row[line]) for line in LINES]), CLIP, 1 - CLIP)
        design = np.vstack([np.ones(len(lines)), -norm.ppf(probs)]).T
        beta, *_ = np.linalg.lstsq(design, lines, rcond=None)
        mus.append(float(beta[0]))
        sds.append(float(beta[1]))
        # The line the model itself puts at even money, read off its own curve
        # rather than from the fit, so the median is not forced to equal the mean.
        medians.append(float(np.interp(0.5, probs[::-1], lines[::-1])))
    games["mu"] = mus
    games["sd"] = sds
    games["median"] = medians
    return games


def report(games: pd.DataFrame) -> None:
    total = games["total"]
    bias = games["mu"] - total
    print(f"\n{len(games)} games with both a model curve and a final score")
    print(
        f"  level:  model mean {games['mu'].mean():.3f}  realized {total.mean():.3f}"
        f"  bias {bias.mean():+.3f} (se {bias.std() / np.sqrt(len(games)):.3f})"
    )
    print(
        f"  shape:  realized mean {total.mean():.2f} median {total.median():.1f}"
        f" skew {skew(total):+.2f}  |  model implied median {games['median'].mean():.2f}"
    )
    print(
        f"  width:  model implied sd {games['sd'].mean():.2f}"
        f"  realized sd {total.std():.2f}  sd of model means {games['mu'].std():.2f}"
        f"  corr(mean, realized) {games['mu'].corr(total):+.3f}"
    )

    print("\nwhat each over line was worth")
    for line in LINES:
        model = games[line].mean()
        real = (total > line).mean()
        print(
            f"  o{line:<5} model {model:.3f}  realized {real:.3f}"
            f"  miss {100 * (real - model):+.1f} pts"
        )

    # A normal centred on the realized mean with the realized width misses the
    # same way the model does, which is the evidence that the defect is skew and
    # not the level or the width.
    print("\nrealized tails against a normal of the same mean and width")
    fitted = norm(total.mean(), total.std())
    for line in (3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5):
        print(
            f"  P(total > {line:<5} realized {(total > line).mean():.3f}"
            f"  normal {1 - fitted.cdf(line):.3f}"
        )

    games = games.assign(bucket=pd.qcut(games["mu"], 5, duplicates="drop"))
    print("\ndoes a higher model mean find a higher final score?")
    print(
        games.groupby("bucket", observed=True)
        .agg(n=("total", "size"), model=("mu", "mean"), realized=("total", "mean"))
        .round(2)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=Path.home() / ".mlb_engine/audit/ledger.csv")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".mlb_engine/cache/boxscores")
    ap.add_argument("--out", type=Path, help="write the joined rows here")
    args = ap.parse_args()

    games = load_totals(args.ledger)
    print(f"{len(games)} games priced at all four lines")
    joined = fit_implied(join_finals(games, args.cache))
    if args.out is not None:
        joined.to_csv(args.out, index=False)
        print(f"rows -> {args.out}")
    report(joined)


if __name__ == "__main__":
    main()
