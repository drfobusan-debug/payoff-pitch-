"""Which first-five model should price the F5 markets: the Markov chain or the sim?

The engine runs two first-five models on every game and prices with one of them.
``models.markov_f5.f5_from_lineups`` -- a per-lineup-slot, TTO-aware Markov chain
-- prices ``f5_ml``, ``f5_total`` and ``f5_rl``; the Monte Carlo used for every
other market also produces first-five run arrays (``home_runs_f5``), and until
``MLBE_F5_FROM_SIM`` they were computed and discarded.

They disagree in one direction, so one of them is hot. This settles it against
the first fives that were actually scored, which needs no odds at all -- the box
score is the grader -- so the replay runs with the API key stripped and costs no
credits. Both models are captured on the same game in the same process, making
the comparison paired: the chain is left in place and the simulator's arrays are
read from the result it was handed, then the two are scored on the F5 total lines
the engine actually offers (4.5 and 5.5) and on the F5 side.

    python -m scripts.f5_model_study --dates 2026-08-06:2026-08-16
    python -m scripts.f5_model_study --grade   # score what was captured

Finding, 10 slates / 133 games (2026-08-06..08-16): both models are hot on raw
probabilities -- the chain by 1.14 runs, the simulator by 0.85 -- but the
simulator is closer on every cut, and its Brier on the two total lines is 0.0053
better with a bootstrap interval clear of zero (+0.0011..+0.0094) when whole
games are resampled. The F5 *side* is untouched (0.2744 vs 0.2747), so the chain's
defect is a run level, not a lean. Graded production cards, whose probabilities
have been through the calibrator, say the same thing an order smaller: 5.05
projected against 4.80 scored, which is how much of the raw error the isotonic
map is already absorbing -- and the reason flipping the flag needs the F5 map
refit rather than inherited.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

LINES = (4.5, 5.5)
OUT_DIR = Path(os.path.expanduser("~/.mlb_engine/f5_study"))


def capture(date: str, sims: int) -> int:
    """Replay one slate, recording both first-five models per game."""
    os.environ["MLBE_ODDS_CACHE_TTL"] = "99999999"
    os.environ["MLBE_STATE_SYNC"] = "0"
    os.environ.pop("THE_ODDS_API_KEY", None)
    os.environ.pop("ODDS_API_KEY", None)

    from mlb_engine.models import markov_f5, montecarlo

    rows: list[dict] = []
    pending: dict = {}
    _sim = montecarlo.MonteCarlo.simulate
    _chain = markov_f5.f5_from_lineups

    def sim_hook(self, home, away):  # type: ignore[no-untyped-def]
        res = _sim(self, home, away)
        pending["res"] = res
        return res

    def chain_hook(*a, **kw):  # type: ignore[no-untyped-def]
        chain = _chain(*a, **kw)
        sim = markov_f5.f5_from_sim(
            pending["res"].home_runs_f5, pending["res"].away_runs_f5
        )
        rows.append(
            {
                "date": date,
                "chain_mean": float(sum(i * p for i, p in enumerate(chain.total_dist))),
                "sim_mean": float(sum(i * p for i, p in enumerate(sim.total_dist))),
                "chain_over": {str(x): chain.p_total_over(x) for x in LINES},
                "sim_over": {str(x): sim.p_total_over(x) for x in LINES},
                "chain_home": chain.p_home_ml,
                "sim_home": sim.p_home_ml,
            }
        )
        return chain

    montecarlo.MonteCarlo.simulate = sim_hook  # type: ignore[method-assign]
    import mlb_engine.pipeline as pipeline

    pipeline.f5_from_lineups = chain_hook  # type: ignore[assignment]
    _mk = pipeline.Pipeline._mk

    def mk(self, *a, **kw):  # type: ignore[no-untyped-def]
        rec = _mk(self, *a, **kw)
        if rec.market == "f5_ml" and rows:
            rows[-1].setdefault("matchup", rec.matchup)
        return rec

    pipeline.Pipeline._mk = mk  # type: ignore[method-assign]

    from mlb_engine.cli import main

    try:
        main(["run", "--date", date, "--sims", str(sims)])
    except SystemExit:
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{date}.json").write_text(json.dumps(rows))
    return len(rows)


def _actual_first_fives(box_dir: Path) -> pd.DataFrame:
    """First-five runs per game, from cached box scores (``data.results`` writes them)."""
    out = []
    for path in glob.glob(str(box_dir / "*.json")):
        d = json.loads(Path(path).read_text())
        innings = (d.get("linescore") or {}).get("innings") or []
        teams = (d.get("boxscore") or {}).get("teams") or {}
        if not innings or not teams:
            continue

        def runs(side: str, innings: list = innings) -> int:
            return sum(
                int((i.get(side, {}) or {}).get("runs", 0) or 0)
                for i in innings
                if i.get("num", 99) <= 5
            )

        home = (teams["home"]["team"] or {}).get("abbreviation")
        away = (teams["away"]["team"] or {}).get("abbreviation")
        out.append(
            {
                "date": d.get("date"),
                "matchup": f"{away} @ {home}",
                "f5": runs("home") + runs("away"),
                "h5": runs("home"),
                "a5": runs("away"),
            }
        )
    return pd.DataFrame(out)


def grade(box_dir: Path) -> None:
    rows = [
        r
        for path in glob.glob(str(OUT_DIR / "*.json"))
        for r in json.loads(Path(path).read_text())
        if "matchup" in r
    ]
    if not rows:
        raise SystemExit(f"nothing captured in {OUT_DIR}; run --dates first")
    cap = pd.DataFrame(rows)
    cap["matchup"] = cap["matchup"].str.replace(" at ", " @ ", regex=False)
    d = cap.merge(_actual_first_fives(box_dir), on=["date", "matchup"], how="inner")
    print(f"slates {cap['date'].nunique()}, captured {len(cap)}, graded {len(d)}")
    print(
        f"  level: chain {d['chain_mean'].mean():.2f} | sim {d['sim_mean'].mean():.2f} "
        f"| actual {d['f5'].mean():.2f}"
    )
    print(
        f"  bias: chain {(d['chain_mean'] - d['f5']).mean():+.2f} runs, "
        f"sim {(d['sim_mean'] - d['f5']).mean():+.2f}"
    )

    losses = []
    for line in LINES:
        key = str(line)
        y = (d["f5"] > line).astype(int)
        pc = d["chain_over"].apply(lambda x, k=key: x[k])
        ps = d["sim_over"].apply(lambda x, k=key: x[k])
        print(
            f"  over {line}: actual {y.mean():.3f} | chain {pc.mean():.3f} "
            f"(brier {np.mean((pc - y) ** 2):.4f}) | sim {ps.mean():.3f} "
            f"(brier {np.mean((ps - y) ** 2):.4f})"
        )
        losses.append(pd.DataFrame({"game": d.index, "gain": (pc - y) ** 2 - (ps - y) ** 2}))

    gains = pd.concat(losses)
    games = gains["game"].unique()
    rng = np.random.default_rng(0)
    boot = [
        gains[gains["game"].isin(rng.choice(games, len(games), replace=True))]["gain"].mean()
        for _ in range(2000)
    ]
    print(
        f"  brier gain for the sim: {gains['gain'].mean():+.4f} "
        f"(95% {np.percentile(boot, 2.5):+.4f}..{np.percentile(boot, 97.5):+.4f}, games resampled)"
    )

    side = d[d["h5"] != d["a5"]]
    y = (side["h5"] > side["a5"]).astype(int)
    print(
        f"  F5 side, ties dropped (n={len(side)}): chain brier "
        f"{np.mean((side['chain_home'] - y) ** 2):.4f} | sim "
        f"{np.mean((side['sim_home'] - y) ** 2):.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", help="single date or START:END, replayed one slate at a time")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--grade", action="store_true")
    ap.add_argument(
        "--box-dir",
        default="~/.mlb_engine/boxscores",
        help="cached box scores (data.results writes them under the data dir)",
    )
    args = ap.parse_args()

    if args.dates:
        start, _, end = args.dates.partition(":")
        day, last = Date.fromisoformat(start), Date.fromisoformat(end or start)
        while day <= last:
            print(f"{day}: {capture(day.isoformat(), args.sims)} games captured")
            day += timedelta(days=1)
    if args.grade:
        grade(Path(os.path.expanduser(args.box_dir)))


if __name__ == "__main__":
    main()
