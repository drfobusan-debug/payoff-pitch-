"""Is the removal hazard stable out of time, and does the platoon term earn a place?

The tables show slot, inning, the starter's exit and same-handedness all moving
the hazard. This fits them jointly, then refits on chronological blocks: a term
that only works in-sample, or flips sign between halves of the season, is not
something a simulator should branch on.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mlb_engine.data.pbp import plate_appearances

PBP = Path.home() / ".mlb_engine" / "cache" / "pbp"
FEATURES = ["sp_out", "late", "slot", "same_hand", "same_hand_x_slot"]


def rows() -> list[dict]:
    out: list[dict] = []
    for path in sorted(PBP.glob("*.json")):
        try:
            plays = json.loads(path.read_text()).get("allPlays") or []
        except (json.JSONDecodeError, OSError):
            continue
        plays = plate_appearances(plays)
        if not plays or max(int(p["about"]["inning"]) for p in plays) < 8:
            continue
        date = str(plays[0]["about"].get("startTime", ""))[:10]
        n_pa: dict[str, int] = defaultdict(int)
        first: dict[tuple[str, int], int] = {}
        hand: dict[tuple[str, int], str] = {}
        gone: dict[tuple[str, int], bool] = defaultdict(bool)
        sp: dict[str, int] = {}
        for play in plays:
            about, mu = play["about"], play["matchup"]
            team = "away" if about["isTopInning"] else "home"
            pit_team = "home" if about["isTopInning"] else "away"
            pitcher = int(mu["pitcher"]["id"])
            sp.setdefault(pit_team, pitcher)
            slot = n_pa[team] % 9
            n_pa[team] += 1
            key = (team, slot)
            batter = int(mu["batter"]["id"])
            orig = first.setdefault(key, batter)
            orig_hand = hand.setdefault(key, mu["batSide"]["code"])
            removed = batter != orig
            if not gone[key]:
                same = int(orig_hand == mu["pitchHand"]["code"])
                out.append(
                    {
                        "date": date,
                        "y": int(removed),
                        "sp_out": float(pitcher != sp[pit_team]),
                        "late": float(int(about["inning"]) >= 7),
                        "slot": (slot + 1 - 5) / 4.0,
                        "same_hand": float(same),
                        "same_hand_x_slot": same * (slot + 1 - 5) / 4.0,
                    }
                )
            gone[key] = gone[key] or removed
    return out


def logit(x: np.ndarray, y: np.ndarray, iters: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Newton-Raphson logistic regression with an intercept, returning (beta, se)."""
    x = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-x @ beta))
        w = np.clip(p * (1 - p), 1e-9, None)
        hess = x.T @ (x * w[:, None])
        step = np.linalg.solve(hess, x.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1.0 / (1.0 + np.exp(-x @ beta))
    w = np.clip(p * (1 - p), 1e-9, None)
    cov = np.linalg.inv(x.T @ (x * w[:, None]))
    return beta, np.sqrt(np.diag(cov))


def _score(b: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Brier and log loss of a fitted hazard on games it was not fitted on."""
    p = np.clip(1.0 / (1.0 + np.exp(-(b[0] + x @ b[1:]))), 1e-6, 1 - 1e-6)
    return (
        float(np.mean((p - y) ** 2)),
        float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    )


def design(data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[r[f] for f in FEATURES] for r in data], dtype=float)
    y = np.array([r["y"] for r in data], dtype=float)
    return x, y


def main() -> None:
    data = [r for r in rows() if r["date"]]
    data.sort(key=lambda r: r["date"])
    dates = sorted({r["date"] for r in data})
    print(f"{len(data):,} at-risk appearances, {dates[0]} .. {dates[-1]}")

    x, y = design(data)
    beta, se = logit(x, y)
    print(f"\n{'term':<18}{'coef':>9}{'z':>8}")
    print(f"{'intercept':<18}{beta[0]:9.3f}{beta[0] / se[0]:8.2f}")
    for i, f in enumerate(FEATURES, start=1):
        print(f"{f:<18}{beta[i]:9.3f}{beta[i] / se[i]:8.2f}")

    # Chronological blocks: fit on everything before a cut, score the games after.
    print("\nout of time, 4 folds (fit on the past, score the future)")
    folds = 4
    edges = [dates[int(len(dates) * k / (folds + 1))] for k in range(1, folds + 1)]
    for cut in edges:
        train = [r for r in data if r["date"] < cut]
        test = [r for r in data if r["date"] >= cut]
        if len(train) < 2000 or len(test) < 1000:
            continue
        xtr, ytr = design(train)
        xte, yte = design(test)
        b_full, _ = logit(xtr, ytr)
        # The same fit without the two platoon terms, to price what they add.
        keep = [FEATURES.index(f) for f in ("sp_out", "late", "slot")]
        b_base, _ = logit(xtr[:, keep], ytr)

        bf, lf = _score(b_full, xte, yte)
        bb, lb = _score(b_base, xte[:, keep], yte)
        i = FEATURES.index("same_hand") + 1
        print(
            f"  cut {cut}  n_test={len(test):<6}"
            f" same_hand {b_full[i]:+.3f}"
            f"  brier {bf:.5f} vs {bb:.5f}"
            f"  logloss {lf:.5f} vs {lb:.5f}"
            f"  {'platoon helps' if lf < lb else 'platoon hurts'}"
        )


if __name__ == "__main__":
    main()
