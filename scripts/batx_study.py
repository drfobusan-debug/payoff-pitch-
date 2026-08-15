"""Evaluate THE BAT X daily projections against our model and against the market.

THE BAT X (Derek Carty) publishes a per-game projected batting line -- PA, AB,
1B, 2B, 3B, HR, BB, K, R, RBI -- which is exactly the input our batter prop
markets need. This script turns that line into over/under probabilities and then
asks the only question worth asking of a new signal: conditional on the market
price, does it carry information our model does not?

The test is deliberately the same one that condemned our own edge: fit

    logit(win) ~ a + b*logit(model) + c*logit(market) + d*logit(batx)

on graded rows. A useful forecast scores a positive coefficient *next to* the
price. Absolute hit rate and ROI are not evidence -- they are dominated by which
props happened to be offered.

Usage::

    # game day, before the slate: turn the BAT X csv into probabilities
    python scripts/batx_study.py price --hitters DKHitters.csv --date 2026-08-10 \\
        --out ~/.mlb_engine/batx/2026-08-10.csv

    # after the slate is graded: join to the ledger and run the head-to-head
    python scripts/batx_study.py grade --probs '~/.mlb_engine/batx/*.csv'

The projections are a *mean* line. Turning a mean into P(over) needs a
distribution, so:

* plate appearances use a two-point distribution straddling the projected mean
  (4.15 PA -> 85% chance of 4, 15% chance of 5), which matches the mean exactly
  and reflects that a hitter takes a whole number of turns;
* hits and total bases come from an exact convolution over those PA of the
  per-PA outcome vector (1B/2B/3B/HR/other), so they share the right joint --
  a home run lifts both;
* runs and RBI have no per-PA decomposition in the feed, so they need a
  distribution imposed on the projected mean, and a Poisson is measurably the
  wrong one -- RBI arrive in clumps (a three-run homer is one swing), which
  gives more zero-RBI games than a Poisson allows. They use negative binomials
  whose dispersion is fitted to our own ledger's realised rates
  (``--rbi-dispersion``, ``--r-dispersion``).

``batter_hrr`` (H+R+RBI) is the weakest of the set: it convolves the exact hit
distribution with independent run and RBI distributions, and hits/runs/RBI are
emphatically not independent -- a home run scores all three at once. Independence
understates the variance, so P(over) on the combo is biased toward the mean. It
is reported, but it is the one number here not to trust.

Which markets are a clean read on BAT X, and which are a read on our own
assumptions, is worth keeping straight when the verdict lands:

===============================  ==================================================
clean -- feed only               ``batter_h``, ``batter_1b``, ``batter_2b``,
                                 ``batter_hr``, ``batter_tb``
assumption-dependent             ``batter_r``, ``batter_rbi``, ``batter_hrr``,
                                 ``pitcher_outs``, ``pitcher_er``, and (through
                                 the out distribution) ``pitcher_k``,
                                 ``pitcher_bb``, ``pitcher_h``
===============================  ==================================================

The pitcher export carries the same shape of line -- TBF, K, BB, H, ER, OUTS --
and splits cleanly in two:

* strikeouts, walks and hits allowed are per-batter-faced events, so their count
  is binomial over the batters actually faced;
* outs and earned runs have no such decomposition, and both are badly
  overdispersed relative to a Poisson -- a starter's night ends early far more
  often than a Poisson on 19 outs allows, and crooked innings give ER a fat right
  tail. They are priced as negative binomials whose spread is an *assumption of
  ours*, not something the feed states (``--outs-sd``, ``--er-dispersion``).

Crucially the first group is not independent of the second. Batters faced is a
consequence of how long the starter lasts, and an early hook is what actually
kills a strikeout over -- so TBF is not held at its projected 24.5 but scaled off
the same overdispersed out distribution before the binomial is applied. That
couples every pitcher market to ``--outs-sd``: a verdict on pitcher props is
partly a verdict on the spread we imposed, in a way the batter markets are not.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binom, nbinom, poisson

LEDGER = os.path.expanduser("~/.mlb_engine/audit/ledger.csv")

# Per-PA outcome columns we need off the BAT X hitters export, mapped to the
# base value each one is worth. Anything else in the row is a rate, a context
# flag, or a DFS scoring artifact and is ignored.
HIT_BASES = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}

# Ledger market -> the BAT X quantity that prices it.
BATTER_MARKETS = (
    "batter_h",
    "batter_1b",
    "batter_2b",
    "batter_hr",
    "batter_tb",
    "batter_r",
    "batter_rbi",
    "batter_hrr",
)
PITCHER_MARKETS = ("pitcher_k", "pitcher_bb", "pitcher_h", "pitcher_er", "pitcher_outs")
MARKETS = (*BATTER_MARKETS, *PITCHER_MARKETS)

# Pitcher counting stats that resolve once per batter faced, so their count is
# binomial over TBF: ledger market -> column in the pitcher export.
PER_TBF = {"pitcher_k": "K", "pitcher_bb": "BB", "pitcher_h": "H"}

# A starter's out total is far more spread than a Poisson on its mean: the mean
# is dragged down by short outings that a Poisson cannot produce. 5.5 outs of
# standard deviation at a ~19-out mean is the shape of a typical starter's
# distribution, and it is an assumption of ours -- the feed projects only a mean.
DEFAULT_OUTS_SD = 5.5

# Earned runs are overdispersed for the same reason in reverse: most starts give
# up one or two, a few give up seven in one inning. Variance ~ 1.6x the mean.
DEFAULT_ER_DISPERSION = 1.6

# RBI are the clumpiest counting stat a hitter has -- they arrive two and three at
# a time on one swing -- so a Poisson badly understates how often a hitter drives
# in nobody. At 2.0 the slate's mean P(RBI >= 1) lands on the 0.274 the ledger
# actually realised, against 0.367 for a Poisson. Runs need far less help (a
# hitter scores at most one per trip), and 1.25 matches the realised 0.354.
# Both are fitted to our own outcomes, so they are OUR parameters, not BAT X's.
DEFAULT_RBI_DISPERSION = 2.0
DEFAULT_R_DISPERSION = 1.25

# Lines to price. The book's line varies night to night, so cover the range the
# ledger has ever shown plus room either side; unjoined lines simply go unused.
PITCHER_LINES = {
    "pitcher_k": (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5),
    "pitcher_bb": (1.5, 2.5, 3.5),
    "pitcher_h": (3.5, 4.5, 5.5, 6.5, 7.5),
    "pitcher_er": (1.5, 2.5, 3.5, 4.5),
    "pitcher_outs": (12.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5),
}

# Trailing tokens the engine appends to a player's name in ``selection``.
_SEL_SUFFIX = re.compile(r"\s+(H\+R\+RBI|1B|2B|3B|HR|TB|H|R|RBI|Ks|Walks|Hits|ER|Outs)\s+[ou][\d.]+$")
_SEL_SIDE = re.compile(r"\s([ou])[\d.]+$")

# Markets the BAT X feed prices on its own numbers alone. The rest impose a
# distribution of ours on their projected mean (see the module docstring), so a
# verdict pooled over all of them grades our assumptions as much as their feed.
CLEAN_MARKETS = frozenset({"batter_h", "batter_1b", "batter_2b", "batter_hr", "batter_tb"})


def norm_name(name: str) -> str:
    """Strip accents and case so 'Julio Rodríguez' joins to 'Julio Rodriguez'."""
    n = unicodedata.normalize("NFKD", str(name))
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = re.sub(r"[.'`-]", "", n)
    n = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", n.strip().lower())
    return re.sub(r"\s+", " ", n)


def player_from_selection(selection: str) -> str:
    return norm_name(_SEL_SUFFIX.sub("", str(selection)))


def is_under(selection: str) -> bool:
    match = _SEL_SIDE.search(str(selection))
    return match is not None and match.group(1) == "u"


def pa_distribution(mean_pa: float) -> dict[int, float]:
    """Two-point distribution over whole plate appearances matching ``mean_pa``."""
    if not math.isfinite(mean_pa) or mean_pa <= 0:
        return {0: 1.0}
    lo = int(math.floor(mean_pa))
    frac = mean_pa - lo
    if frac <= 1e-9:
        return {lo: 1.0}
    return {lo: 1.0 - frac, lo + 1: frac}


def hit_tb_distribution(mean_pa: float, rates: dict[str, float]) -> dict[tuple[int, int], float]:
    """Exact joint distribution of (hits, total bases) over the PA distribution.

    ``rates`` holds per-PA probabilities for 1B/2B/3B/HR; the remainder is any
    non-hit outcome, which advances neither count.
    """
    p_hit = {b: max(0.0, rates.get(k, 0.0)) for k, b in HIT_BASES.items()}
    p_none = max(0.0, 1.0 - sum(p_hit.values()))

    joint: dict[tuple[int, int], float] = {}
    for n_pa, w_pa in pa_distribution(mean_pa).items():
        state: dict[tuple[int, int], float] = {(0, 0): 1.0}
        for _ in range(n_pa):
            nxt: dict[tuple[int, int], float] = {}
            for (h, tb), w in state.items():
                nxt[(h, tb)] = nxt.get((h, tb), 0.0) + w * p_none
                for bases, p in p_hit.items():
                    if p <= 0.0:
                        continue
                    key = (h + 1, tb + bases)
                    nxt[key] = nxt.get(key, 0.0) + w * p
            state = nxt
        for key, w in state.items():
            joint[key] = joint.get(key, 0.0) + w_pa * w
    return joint


def _raw_count_pmf(mean: float, var: float, cap: int) -> np.ndarray:
    mean = max(1e-6, float(mean))
    k = np.arange(cap + 1)
    if var <= mean * (1.0 + 1e-6):
        pmf = poisson.pmf(k, mean)
    else:
        p = mean / var
        r = mean * mean / (var - mean)
        pmf = nbinom.pmf(k, r, p)
    total = pmf.sum()
    return pmf / total if total > 0 else pmf


def _moments(pmf: np.ndarray) -> tuple[float, float]:
    k = np.arange(len(pmf))
    m = float(k @ pmf)
    return m, float(max(0.0, (k * k) @ pmf - m * m) ** 0.5)


def dist_mean(dist: dict[int, float]) -> float:
    return float(sum(k * w for k, w in dist.items()))


def dist_sd(dist: dict[int, float]) -> float:
    m = dist_mean(dist)
    return float(max(0.0, sum(k * k * w for k, w in dist.items()) - m * m) ** 0.5)


def _match_mean(target_mean: float, var: float, cap: int) -> np.ndarray:
    """Solve for the underlying mean whose truncated distribution hits the target."""
    lo, hi = target_mean, float(cap) * 4.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if _moments(_raw_count_pmf(mid, max(var, mid * 1.000001), cap))[0] < target_mean:
            lo = mid
        else:
            hi = mid
    return _raw_count_pmf(hi, max(var, hi * 1.000001), cap)


def overdispersed_pmf(mean: float, sd: float, cap: int) -> dict[int, float]:
    """Count distribution on ``0..cap`` matching both ``mean`` and ``sd``.

    Truncation is not a detail here. A negative binomial matched to a starter's
    19.1 projected outs with a 5.5 spread puts ~5% of its mass beyond the 27 outs
    a complete game allows; renormalising that away drops the mean by two full
    outs *and* squeezes the spread to 4.4. Both errors push in the same direction
    -- they would understate every out and strikeout over, and the shortfall would
    look like BAT X being wrong rather than us being sloppy.

    So both moments are solved for on the truncated support: an outer search on
    the underlying spread, an inner search on the underlying mean.
    """
    mean = max(1e-6, float(mean))
    lo, hi = mean**0.5, float(cap)
    pmf = _match_mean(mean, max(sd, mean**0.5) ** 2, cap)
    for _ in range(40):
        mid = (lo + hi) / 2.0
        pmf = _match_mean(mean, mid * mid, cap)
        if _moments(pmf)[1] < sd:
            lo = mid
        else:
            hi = mid
    return {k: float(w) for k, w in enumerate(pmf)}


def tbf_distribution(mean_tbf: float, outs: dict[int, float], mean_outs: float) -> dict[int, float]:
    """Spread batters faced along the out distribution.

    A starter who records 12 outs does not face his projected 24 batters. Holding
    TBF at its mean would price strikeout overs as if the hook never comes early,
    which is precisely how those overs lose.
    """
    if mean_outs <= 0:
        return {int(round(mean_tbf)): 1.0}
    per_out = mean_tbf / mean_outs
    out: dict[int, float] = {}
    for o, w in outs.items():
        n = max(1, int(round(o * per_out)))
        out[n] = out.get(n, 0.0) + w
    return out


def binomial_count_pmf(trials: dict[int, float], rate: float, cap: int) -> dict[int, float]:
    """Distribution of a per-trial Bernoulli count, averaged over the trial count."""
    rate = min(max(rate, 0.0), 1.0)
    out: dict[int, float] = {}
    for n, w in trials.items():
        for k in range(min(n, cap) + 1):
            out[k] = out.get(k, 0.0) + w * float(binom.pmf(k, n, rate))
    return out


def p_at_least(dist: dict[int, float], threshold: int) -> float:
    return float(sum(w for k, w in dist.items() if k >= threshold))


def marginal(joint: dict[tuple[int, int], float], axis: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, w in joint.items():
        out[key[axis]] = out.get(key[axis], 0.0) + w
    return out


def binomial_at_least_one(mean_pa: float, rate: float) -> float:
    """P(count >= 1) for a per-PA Bernoulli, averaged over the PA distribution."""
    if rate <= 0.0:
        return 0.0
    return float(sum(w * (1.0 - (1.0 - min(rate, 1.0)) ** n) for n, w in pa_distribution(mean_pa).items()))


def convolve(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for ka, wa in a.items():
        for kb, wb in b.items():
            out[ka + kb] = out.get(ka + kb, 0.0) + wa * wb
    return out


def price_row(row: pd.Series, r_dispersion: float, rbi_dispersion: float) -> dict[str, float]:
    """All batter-prop over probabilities implied by one BAT X projected line."""
    pa = float(row["PA"])
    rates = {k: (float(row[k]) / pa if pa > 0 else 0.0) for k in HIT_BASES}
    joint = hit_tb_distribution(pa, rates)
    hits = marginal(joint, 0)
    bases = marginal(joint, 1)

    r_mean, rbi_mean = float(row["R"]), float(row["RBI"])
    runs = overdispersed_pmf(r_mean, (r_dispersion * r_mean) ** 0.5, cap=6)
    rbis = overdispersed_pmf(rbi_mean, (rbi_dispersion * rbi_mean) ** 0.5, cap=8)
    hrr = convolve(convolve(hits, runs), rbis)

    return {
        "batter_h@0.5": p_at_least(hits, 1),
        "batter_h@1.5": p_at_least(hits, 2),
        "batter_1b@0.5": binomial_at_least_one(pa, rates["1B"]),
        "batter_2b@0.5": binomial_at_least_one(pa, rates["2B"]),
        "batter_hr@0.5": binomial_at_least_one(pa, rates["HR"]),
        "batter_tb@0.5": p_at_least(bases, 1),
        "batter_tb@1.5": p_at_least(bases, 2),
        "batter_tb@2.5": p_at_least(bases, 3),
        "batter_tb@3.5": p_at_least(bases, 4),
        "batter_r@0.5": p_at_least(runs, 1),
        "batter_rbi@0.5": p_at_least(rbis, 1),
        "batter_hrr@1.5": p_at_least(hrr, 2),
        "batter_hrr@2.5": p_at_least(hrr, 3),
    }


def price_pitcher_row(
    row: pd.Series, outs_sd: float, er_dispersion: float
) -> tuple[dict[str, float], tuple[float, float]]:
    """Pitcher-prop over probabilities, plus the realised (mean, sd) of the outs fit."""
    mean_outs = float(row["OUTS"])
    mean_tbf = float(row["TBF"])
    # 27 outs is a complete game; the distribution cannot run past it.
    outs = overdispersed_pmf(mean_outs, outs_sd, cap=27)
    trials = tbf_distribution(mean_tbf, outs, mean_outs)

    probs: dict[str, float] = {}
    for market, col in PER_TBF.items():
        rate = float(row[col]) / mean_tbf if mean_tbf > 0 else 0.0
        dist = binomial_count_pmf(trials, rate, cap=20)
        for line in PITCHER_LINES[market]:
            probs[f"{market}@{line}"] = p_at_least(dist, math.ceil(line))

    mean_er = float(row["ER"])
    er = overdispersed_pmf(mean_er, (er_dispersion * mean_er) ** 0.5, cap=15)
    for line in PITCHER_LINES["pitcher_er"]:
        probs[f"pitcher_er@{line}"] = p_at_least(er, math.ceil(line))
    for line in PITCHER_LINES["pitcher_outs"]:
        probs[f"pitcher_outs@{line}"] = p_at_least(outs, math.ceil(line))
    return probs, (dist_mean(outs), dist_sd(outs))


def read_pitchers(path: str) -> pd.DataFrame:
    """Load the BAT X pitchers export."""
    df = pd.read_csv(os.path.expanduser(path))
    df.columns = [str(c).strip().upper() for c in df.columns]
    name_col = next((c for c in ("PLAYER", "NAME", "PLAYER_NAME") if c in df.columns), None)
    if name_col is None:
        raise SystemExit(f"no player-name column in {path}: {list(df.columns)[:12]}")
    missing = [c for c in ("TBF", "OUTS", "ER", *PER_TBF.values()) if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing projection columns {missing}")
    df = df.rename(columns={name_col: "NAME"})
    return drop_non_players(df)


def drop_non_players(df: pd.DataFrame) -> pd.DataFrame:
    """Discard footer/legend rows the export appends after the last player.

    The pitchers file ends with a row whose name is ``1`` and whose columns hold
    what look like column indices -- 44 strikeouts, 56 earned runs. It parses
    perfectly happily and would be priced as a real starter.

    The test is deliberately just "has letters in it": anything stricter starts
    throwing away real players. Requiring two whole words dropped ``A.J. Ewing``,
    ``Tyler O'Neill`` and ``Travis d'Arnaud``.
    """
    named = df["NAME"].astype(str).str.count(r"[A-Za-z]") >= 3
    dropped = int((~named).sum())
    if dropped:
        print(f"dropped {dropped} non-player row(s): {list(df.loc[~named, 'NAME'])[:5]}")
    out = df[named].copy()
    out["player"] = out["NAME"].map(norm_name)
    return out


def check_pitcher_schema(df: pd.DataFrame) -> None:
    """Outs and innings pitched must agree, and hits must not exceed batters faced."""
    if {"IP", "OUTS"} <= set(df.columns):
        # IP is a plain decimal here (6.36 IP = 19.1 outs), not the scoreboard
        # convention where .1 and .2 mean thirds of an inning.
        err = (df["IP"] * 3 - df["OUTS"]).abs()
        print(f"schema check: IP vs OUTS median abs error {err.median():.2f} outs")
        if err.median() > 0.5:
            print("  WARNING: IP and OUTS disagree -- the columns are probably misaligned")
            raise SystemExit(1)
    bad = int((df["H"] + df["BB"] > df["TBF"]).sum())
    if bad:
        print(f"  WARNING: {bad} rows have H+BB above TBF, which cannot happen")


# DraftKings MLB hitter scoring. Used only to verify the export's columns are
# mapped correctly: the projected components must reproduce the projected FPTS.
# A misaligned header silently produces plausible-looking nonsense otherwise.
DK_POINTS = {"1B": 3, "2B": 5, "3B": 8, "HR": 10, "RBI": 2, "R": 2, "BB": 2, "HBP": 2, "SB": 5}


def check_schema(df: pd.DataFrame) -> None:
    """Reconcile projected FPTS against the projected components."""
    if "FPTS" not in df.columns:
        print("no FPTS column -- cannot verify the column mapping")
        return
    cols = [c for c in DK_POINTS if c in df.columns]
    rebuilt = sum(df[c].fillna(0.0) * DK_POINTS[c] for c in cols)
    err = (rebuilt - df["FPTS"].fillna(0.0)).abs()
    ok = float((err < 0.25).mean())
    print(f"schema check: rebuilt DK points from {cols}")
    print(f"  median abs error {err.median():.3f} pts, {ok:.1%} of rows within 0.25")
    if ok < 0.9:
        print("  WARNING: components do not reproduce FPTS -- the columns are probably")
        print("  misaligned, and every probability below would be nonsense. Stopping.")
        raise SystemExit(1)


def read_hitters(path: str) -> pd.DataFrame:
    """Load the BAT X hitters export, tolerating case and whitespace in headers."""
    df = pd.read_csv(os.path.expanduser(path))
    df.columns = [str(c).strip().upper() for c in df.columns]
    name_col = next((c for c in ("NAME", "PLAYER", "PLAYER_NAME") if c in df.columns), None)
    if name_col is None:
        raise SystemExit(f"no player-name column in {path}: {list(df.columns)[:12]}")
    missing = [c for c in ("PA", "R", "RBI", *HIT_BASES) if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing projection columns {missing}")
    df = df.rename(columns={name_col: "NAME"})
    return drop_non_players(df)


def _emit(date: str, player: str, team: str, probs: dict[str, float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, prob in probs.items():
        market, line = key.split("@")
        rows.append(
            {
                "date": date,
                "player": player,
                "team": team,
                "market": market,
                "line": float(line),
                "batx_prob": round(prob, 6),
            }
        )
    return rows


def cmd_price(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []

    hitters = read_hitters(args.hitters)
    check_schema(hitters)
    for _, row in hitters.iterrows():
        if not np.isfinite(row.get("PA", np.nan)) or float(row["PA"]) <= 0:
            continue
        probs = price_row(row, args.r_dispersion, args.rbi_dispersion)
        rows.extend(_emit(args.date, str(row["player"]), str(row.get("TEAM", "")), probs))
    print(f"{len(hitters)} hitters -> {len(rows)} selections")

    if args.pitchers:
        pitchers = read_pitchers(args.pitchers)
        check_pitcher_schema(pitchers)
        before = len(rows)
        drift: list[float] = []
        for _, row in pitchers.iterrows():
            if not np.isfinite(row.get("OUTS", np.nan)) or float(row["OUTS"]) <= 0:
                continue
            probs, (fit_mean, fit_sd) = price_pitcher_row(row, args.outs_sd, args.er_dispersion)
            drift.append(abs(fit_mean - float(row["OUTS"])))
            drift.append(abs(fit_sd - args.outs_sd))
            rows.extend(_emit(args.date, str(row["player"]), str(row.get("TM", "")), probs))
        print(f"{len(pitchers)} pitchers -> {len(rows) - before} selections")
        # The out distribution is capped at 27, so confirm the moment solve still
        # landed on the projected mean and the spread we asked for.
        print(f"  outs fit: worst moment miss {max(drift):.4f} (target sd {args.outs_sd})")

    dest = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"wrote {len(rows)} priced selections -> {dest}")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_logit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plain logistic regression with an intercept. Returns (coefs, std errors)."""
    design = np.column_stack([np.ones(len(y)), x])

    def nll(beta: np.ndarray) -> float:
        z = design @ beta
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    fit = minimize(nll, np.zeros(design.shape[1]), method="BFGS")
    pred = 1.0 / (1.0 + np.exp(-(design @ fit.x)))
    w = pred * (1 - pred)
    cov = np.linalg.pinv(design.T @ (design * w[:, None]))
    return fit.x, np.sqrt(np.diag(cov))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def cmd_grade(args: argparse.Namespace) -> None:
    paths = sorted(glob.glob(os.path.expanduser(args.probs)))
    if not paths:
        raise SystemExit(f"no BAT X probability files match {args.probs}")
    batx = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    key = ["date", "player", "market", "line"]
    dupes = int(batx.duplicated(key).sum())
    if dupes:
        # Same player twice on a date (doubleheaders, or overlapping exports).
        # Keeping both would double-count those rows in the fit.
        print(f"dropping {dupes} duplicate projections on {key}")
        batx = batx.drop_duplicates(key)

    led = pd.read_csv(os.path.expanduser(args.ledger))
    led = led[led.market.isin(MARKETS) & led.result.isin(["win", "loss"])].copy()
    led["player"] = led.selection.map(player_from_selection)
    led["y"] = (led.result == "win").astype(float)

    df = led.merge(batx, on=["date", "player", "market", "line"], how="inner")
    if df.empty:
        raise SystemExit("no ledger rows joined to a BAT X projection -- check dates and name normalisation")

    # The ledger's model_prob and fair_prob are the probability of the side that
    # was recommended; batx_prob is always P(over). On an under row the two are
    # complements, so comparing them unflipped grades BAT X against its own
    # mirror image -- and every counting prop the engine fades is an under.
    df["under"] = df.selection.map(is_under)
    df.loc[df.under, "batx_prob"] = 1.0 - df.loc[df.under, "batx_prob"]
    if int(df.under.sum()):
        print(f"  of which under rows (batx flipped to P(under)): {int(df.under.sum())}")

    joined_dates = sorted(df.date.unique())
    print(f"joined {len(df)} graded rows over {len(joined_dates)} slates ({joined_dates[0]}..{joined_dates[-1]})")
    print(f"  unmatched ledger rows: {len(led) - len(df)}")

    print("\ncalibration (mean predicted vs actual)")
    print(f"  {'source':<8} {'mean p':>8} {'actual':>8} {'gap':>8} {'brier':>8}")
    for label, col in (("batx", "batx_prob"), ("model", "model_prob"), ("market", "fair_prob")):
        if col not in df.columns:
            continue
        sub = df[np.isfinite(df[col])]
        if sub.empty:
            print(f"  {label:<8} {'--':>8} (no rows)")
            continue
        p, yy = sub[col].to_numpy(), sub.y.to_numpy()
        print(f"  {label:<8} {p.mean():8.3f} {yy.mean():8.3f} {p.mean() - yy.mean():+8.3f} {brier(p, yy):8.4f}")

    priced = df[np.isfinite(df.fair_prob) & np.isfinite(df.model_prob) & np.isfinite(df.batx_prob)]
    print(f"\nhead-to-head on {len(priced)} rows carrying a real market price")
    if len(priced) < 100:
        print("  too few priced rows to read anything into the coefficients")
    if priced.empty:
        return
    _head_to_head(priced, "all markets")
    _head_to_head(priced[priced.market.isin(CLEAN_MARKETS)], "feed only")
    _head_to_head(priced[~priced.market.isin(CLEAN_MARKETS)], "our assumptions")
    print("\n  a forecast with information the others lack scores a positive coefficient here")
    print("  read 'feed only': the other rows grade our own distributions as much as theirs")


FIT_COLS = ("model_prob", "fair_prob", "batx_prob")


def _design(frame: pd.DataFrame, markets: list[str]) -> np.ndarray:
    """Logit forecasts plus a per-market intercept.

    Without the market dummies the fit is free to score a forecast for knowing
    that a home run is rarer than a hit, which every one of them knows. Only
    variation *within* a market is information.
    """
    x = np.column_stack([logit(frame[c].to_numpy()) for c in FIT_COLS])
    if len(markets) < 2:
        return x
    dummies = pd.get_dummies(frame.market).reindex(columns=markets).fillna(0.0)
    return np.column_stack([x, dummies.to_numpy(float)[:, 1:]])


def _head_to_head(frame: pd.DataFrame, label: str, draws: int = 300) -> None:
    """Fit the three forecasts against each other, bootstrapped by player-date.

    One hitter contributes up to five rows on a slate off a single projected
    line, so the rows are anything but independent and the plain standard error
    is roughly half what it should be. Resampling whole player-dates prices that
    in: on the first four slates it moved the BAT X interval from comfortably
    positive to touching zero.
    """
    if len(frame) < 150:
        print(f"  {label:<16} n={len(frame):<5} too few rows to read")
        return
    markets = sorted(frame.market.unique())
    beta, _ = fit_logit(_design(frame, markets), frame.y.to_numpy())
    groups = [group for _, group in frame.groupby(frame.player + "|" + frame.date)]
    rng = np.random.default_rng(4)
    sampled: list[np.ndarray] = []
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        boot = pd.concat([groups[i] for i in pick], ignore_index=True)
        try:
            fit, _unused = fit_logit(_design(boot, markets), boot.y.to_numpy())
        except (ValueError, np.linalg.LinAlgError):
            continue
        sampled.append(fit[1 : 1 + len(FIT_COLS)])
    spread = np.array(sampled)
    print(f"  {label:<16} n={len(frame):<5} ({len(groups)} player-dates)")
    print(f"    {'term':<12} {'coef':>7} {'95% CI':>18} {'P(>0)':>7}")
    for i, name in enumerate(FIT_COLS):
        lo, hi = np.percentile(spread[:, i], [2.5, 97.5])
        share = float(np.mean(spread[:, i] > 0))
        print(f"    {name:<12} {beta[i + 1]:+7.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>18} {share:7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("price", help="turn a BAT X hitters export into over probabilities")
    p.add_argument("--hitters", required=True, help="path to the BAT X hitters CSV export")
    p.add_argument("--pitchers", help="path to the BAT X pitchers CSV export")
    p.add_argument("--date", required=True, help="slate date the export covers (YYYY-MM-DD)")
    p.add_argument("--out", required=True, help="where to write the priced selections")
    p.add_argument(
        "--outs-sd",
        type=float,
        default=DEFAULT_OUTS_SD,
        help="assumed standard deviation of a starter's out total (feed gives only a mean)",
    )
    p.add_argument(
        "--er-dispersion",
        type=float,
        default=DEFAULT_ER_DISPERSION,
        help="assumed ratio of earned-run variance to its mean",
    )
    p.add_argument(
        "--rbi-dispersion",
        type=float,
        default=DEFAULT_RBI_DISPERSION,
        help="assumed ratio of RBI variance to its mean (RBI arrive in clumps)",
    )
    p.add_argument(
        "--r-dispersion",
        type=float,
        default=DEFAULT_R_DISPERSION,
        help="assumed ratio of runs-scored variance to its mean",
    )
    p.set_defaults(func=cmd_price)

    g = sub.add_parser("grade", help="join priced selections to graded ledger rows")
    g.add_argument("--probs", required=True, help="glob of files written by 'price'")
    g.add_argument("--ledger", default=LEDGER)
    g.set_defaults(func=cmd_grade)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
