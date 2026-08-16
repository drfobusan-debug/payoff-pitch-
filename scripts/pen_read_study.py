"""What survives in a three-week bullpen read, measured out of time.

Three questions the daily preview and the simulator were answering without
evidence, each scored the way the number is used:

``forward``
    Take a 21-day relief window per team and score it against the **next** 21
    days of relief wOBA allowed. Compares the raw read, the flat 60-PA prior
    that used to ship, and the fitted per-outcome pen priors, and asks whether
    wOBA or xwOBA is the better forecast. The slope of what happened next on
    what the window said is the shrinkage the read has earned: 1.0 means it is
    used at the right strength, 0.15 means at six times its worth.

``spread``
    The per-arm wOBA standard deviation the preview used to call volatility,
    across two adjacent windows, against the binomial noise floor for arms with
    ~33 batters faced.

``fatigue``
    The "gassed arms" workload proxy (rebuilt from pitch rows, same rule as the
    StatsAPI boxscore version), scored against the relief wOBA the pen went on
    to allow that night.

``mlgate``
    The same proxy scored against the thing the moneyline gate spends on it:
    did the side it would demote go on to lose more often than that team
    usually does?

Usage::

    python -m scripts.pen_read_study forward --cache ~/.mlb_engine/cache/statcast_2026-04-01_2026-07-27.pkl
    python -m scripts.pen_read_study spread  --cache ...
    python -m scripts.pen_read_study fatigue --cache ...
    python -m scripts.pen_read_study mlgate  --cache ...

The cache is a Statcast pickle the engine has already downloaded; the wider the
span, the more window pairs ``forward`` can score.
"""

from __future__ import annotations

import argparse
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from mlb_engine.features.ml_gate import DEFAULT_PEN_DEPLETED as ML_PEN_DEPLETED
from mlb_engine.features.ml_gate import DEFAULT_PEN_EDGE as ML_PEN_EDGE
from mlb_engine.features.rolling import (
    MIN_ARM_PA,
    PEN_PRIOR_STRENGTH,
    PRIOR_STRENGTH,
    _pa_rows,
    bullpen_relief_frame,
    rates_from_events,
    woba_from_rates,
)

MIN_WINDOW_PA = 80  # a pen window thinner than this is not a read at all
MIN_BBE = 30  # matches build_bullpen_profile's xwOBA floor
BB_WOBA = 0.690  # the walk's wOBA weight, to put contact xwOBA on the PA scale


def _load(cache: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_pickle(cache)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    teams = sorted(set(df["home_team"].dropna()) | set(df["away_team"].dropna()))
    print(f"{cache.name}: {df['game_date'].min()}..{df['game_date'].max()}, {len(teams)} teams")
    return df, teams


def _window(df: pd.DataFrame, team: str, as_of: Date, days: int = 21) -> dict[str, float] | None:
    """Every read of one pen window, plus the realised wOBA that scores it."""
    relief = bullpen_relief_frame(df, team, as_of, days, 6)
    if not len(relief):
        return None
    pa = _pa_rows(relief)
    if len(pa) < MIN_WINDOW_PA:
        return None
    events = pa["events"]
    out = {
        "pa": float(len(pa)),
        "raw": woba_from_rates(rates_from_events(events, prior_strength=0.0).as_dict()),
        "flat60": woba_from_rates(
            rates_from_events(events, prior_strength=PRIOR_STRENGTH).as_dict()
        ),
        "fitted": woba_from_rates(
            rates_from_events(events, prior_strength=PEN_PRIOR_STRENGTH).as_dict()
        ),
        "k": rates_from_events(events, prior_strength=0.0).as_dict().get("K", 0.0),
    }
    xw = relief["estimated_woba_using_speedangle"].dropna()
    if len(xw) >= MIN_BBE:
        walks = rates_from_events(events, prior_strength=0.0).as_dict().get("BB", 0.0)
        # Contact xwOBA covers batted balls only; credit walks and charge the
        # rest as zero to land on the same per-PA scale as wOBA allowed.
        out["xwoba"] = float(xw.mean()) * (len(xw) / len(pa)) + BB_WOBA * walks
    return out


def _pairs(df: pd.DataFrame, teams: list[str], step: int = 7, days: int = 21) -> pd.DataFrame:
    rows = []
    as_of = min(df["game_date"]) + timedelta(days=days)
    end = max(df["game_date"]) - timedelta(days=days)
    while as_of <= end:
        for team in teams:
            now = _window(df, team, as_of, days)
            later = _window(df, team, as_of + timedelta(days=days), days)
            if now and later:
                rows.append({"team": team, "as_of": as_of, **now, "next": later["raw"],
                             "next_k": later["k"]})
        as_of += timedelta(days=step)
    return pd.DataFrame(rows)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.cov(x, y)[0, 1] / np.var(x))


def forward(df: pd.DataFrame, teams: list[str]) -> None:
    d = _pairs(df, teams)
    print(f"\n{len(d)} team-windows over {d.as_of.nunique()} start dates, "
          f"mean {d.pa.mean():.0f} relief PA, next-window mean wOBA {d['next'].mean():.4f}\n")
    y = d["next"].to_numpy()
    print("  read              sd     slope on next window   RMSE as a forecast")
    for col in ("raw", "flat60", "fitted"):
        x = d[col].to_numpy()
        print(f"  {col:14s} {x.std():.4f}          {_slope(x, y):5.2f}          "
              f"{float(np.sqrt(((y - x) ** 2).mean())):.4f}")
    league = float(np.sqrt(((y - y.mean()) ** 2).mean()))
    print(f"  {'league mean':14s} {0.0:.4f}            n/a          {league:.4f}")

    print("\n  wOBA or xwOBA, against the next 21 days of wOBA allowed:")
    have = d.dropna(subset=["xwoba"])
    print(f"    n={len(have)}  wOBA r = {have.raw.corr(have['next']):+.3f}   "
          f"xwOBA r = {have.xwoba.corr(have['next']):+.3f}   "
          f"K% r = {have.k.corr(have['next']):+.3f}")
    z = have[["raw", "xwoba"]].to_numpy()
    z = np.column_stack([np.ones(len(z)), (z - z.mean(0)) / z.std(0)])
    beta, *_ = np.linalg.lstsq(z, have["next"].to_numpy(), rcond=None)
    print(f"    joint, standardised: wOBA {beta[1]:+.4f}, xwOBA {beta[2]:+.4f} per sd")
    print("\n  by start date:")
    for as_of, g in have.groupby("as_of"):
        print(f"    {as_of}  n={len(g):2d}  wOBA r={g.raw.corr(g['next']):+.3f}  "
              f"xwOBA r={g.xwoba.corr(g['next']):+.3f}  K%->K% r={g.k.corr(g.next_k):+.3f}")


def _arm_wobas(df: pd.DataFrame, team: str, as_of: Date, days: int = 21) -> list[tuple[float, int]]:
    relief = bullpen_relief_frame(df, team, as_of, days, 6)
    if not len(relief):
        return []
    out = []
    for _, arm in relief.groupby("pitcher"):
        events = arm["events"].dropna()
        if len(events) < MIN_ARM_PA:
            continue
        out.append((woba_from_rates(rates_from_events(events).as_dict()), len(events)))
    return out


def spread(df: pd.DataFrame, teams: list[str], days: int = 21) -> None:
    """Is a pen's arm-to-arm spread a property of the pen, or of the sample?"""
    end = max(df["game_date"])
    b_as_of, a_as_of = end, end - timedelta(days=days)
    rows, noise = [], []
    for team in teams:
        a, b = _arm_wobas(df, team, a_as_of, days), _arm_wobas(df, team, b_as_of, days)
        if len(a) < 2 or len(b) < 2:
            continue
        rows.append((team, float(np.std([w for w, _ in a])), float(np.std([w for w, _ in b]))))
        # Binomial noise floor: what spread appears among identical arms, after
        # the same shrinkage the per-arm estimate carries.
        for _, n in a + b:
            keep = (n / (n + PRIOR_STRENGTH)) ** 2
            noise.append(keep * 0.300 * 0.700 * 1.6 / n)  # wOBA-scaled per-PA variance
    d = pd.DataFrame(rows, columns=["team", "a", "b"])
    print(f"\npens with 2+ qualified arms in both windows: {len(d)}")
    print(f"  spread: window A {d.a.mean():.4f}, window B {d.b.mean():.4f}")
    print(f"  binomial noise floor: {float(np.sqrt(np.mean(noise))):.4f}")
    print(f"  split-half r of the spread: {d.a.corr(d.b):+.3f}")


def _relief_pitch_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relief rows and their per-arm daily pitch counts, keyed by fielding team."""
    df = df.copy()
    df["fielding"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])
    starters = set(
        map(tuple, df.loc[df["inning"] <= 1, ["game_date", "pitcher"]].dropna().to_numpy())
    )
    rel = df[[(d, p) not in starters for d, p in zip(df["game_date"], df["pitcher"], strict=False)]]
    rel = rel[rel["inning"] >= 6]
    pitch = rel.groupby(["fielding", "game_date", "pitcher"]).size().rename("pitches").reset_index()
    return rel, pitch


def _workload(prior: pd.DataFrame, as_of: Date) -> tuple[float, float]:
    """The StatsAPI proxy's rule, rebuilt from pitch rows: (0-100 score, three-day load)."""
    last2 = sorted(prior["game_date"].unique())[-2:]
    recent = prior[prior["game_date"].isin(last2)]
    gassed = 0
    for _, arm in recent.groupby("pitcher"):
        last_day = arm.loc[arm["game_date"] == last2[-1], "pitches"].sum()
        if arm["game_date"].nunique() >= 2 or arm["pitches"].sum() >= 40 or last_day >= 30:
            gassed += 1
    w21 = prior[prior["game_date"] >= as_of - timedelta(days=21)]
    w3 = prior[prior["game_date"] >= as_of - timedelta(days=3)]
    expect = len(w21) / 21 * 3 if len(w21) else 0
    return min(100.0, gassed * 20.0), (len(w3) / expect if expect else float("nan"))


def fatigue(df: pd.DataFrame, _teams: list[str]) -> None:
    """Does the workload proxy predict the relief wOBA that follows it?"""
    rel, pitch = _relief_pitch_counts(df)

    out = []
    for (team, gd), g in _pa_rows(rel).groupby(["fielding", "game_date"]):
        if len(g) < 6:
            continue
        out.append(
            (team, gd, woba_from_rates(rates_from_events(g["events"], prior_strength=0.0).as_dict()))
        )
    res = pd.DataFrame(out, columns=["team", "date", "woba"])
    by_team = {t: g for t, g in pitch.groupby("fielding")}

    rows = []
    for _, r in res.iterrows():
        g = by_team.get(r.team)
        if g is None:
            continue
        prior = g[g["game_date"] < r.date]
        if prior.empty:
            continue
        score, load = _workload(prior, r.date)
        rows.append((r.woba, int(round(score / 20)), score, load))

    d = pd.DataFrame(rows, columns=["woba", "gassed", "fatigue", "load"]).dropna()
    print(f"\nteam-games: {len(d)}, mean relief wOBA allowed {d.woba.mean():.3f}")
    print(f"  r(workload score, wOBA allowed) = {d.fatigue.corr(d.woba):+.3f}")
    print(f"  r(three-day load, wOBA allowed) = {d.load.corr(d.woba):+.3f}")
    print(d.groupby("gassed").agg(games=("woba", "size"), woba=("woba", "mean")).round(3).to_string())
    dep, ok = d[d.fatigue >= 60], d[d.fatigue < 60]
    se = float((dep.woba.std() ** 2 / len(dep) + ok.woba.std() ** 2 / len(ok)) ** 0.5)
    print(f"  depleted n={len(dep)} {dep.woba.mean():.3f} vs rest n={len(ok)} {ok.woba.mean():.3f}"
          f" -> {dep.woba.mean() - ok.woba.mean():+.4f} +/- {se:.4f}")


def _finals(start: Date, end: Date) -> pd.DataFrame:
    """One row per side of every decided game in the range, from the MLB schedule."""
    js = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "startDate": str(start), "endDate": str(end), "hydrate": "team"},
        timeout=60,
    ).json()
    rows = []
    for day in js.get("dates", []):
        gd = pd.to_datetime(day["date"]).date()
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            home, away = g["teams"]["home"], g["teams"]["away"]
            if "score" not in home or "score" not in away or home["score"] == away["score"]:
                continue
            ht, at = home["team"]["abbreviation"], away["team"]["abbreviation"]
            rows.append((gd, ht, at, int(home["score"] > away["score"])))
            rows.append((gd, at, ht, int(away["score"] > home["score"])))
    return pd.DataFrame(rows, columns=["date", "team", "opp", "win"])


def mlgate(df: pd.DataFrame, _teams: list[str]) -> None:
    """Does the moneyline depletion gate demote losers, or just bets?

    Rebuilds the proxy before each game and asks whether the side the gate would
    have passed on lost more than that same team usually does -- the team's own
    rate standing in for the price, which we do not have back this far.
    """
    _, pitch = _relief_pitch_counts(df)
    dates = sorted(df["game_date"].unique())
    rows = []
    for team, g in pitch.groupby("fielding"):
        for gd in dates:
            prior = g[g["game_date"] < gd]
            if prior.empty or (gd - max(prior["game_date"])).days > 3:
                continue
            rows.append((team, gd, _workload(prior, gd)[0]))
    fat = pd.DataFrame(rows, columns=["team", "date", "fatigue"])

    res = _finals(min(dates), max(dates))
    d = res.merge(fat, on=["date", "team"]).merge(
        fat.rename(columns={"team": "opp", "fatigue": "opp_fatigue"}), on=["date", "opp"]
    )
    d["gated"] = (d.fatigue >= ML_PEN_DEPLETED) & ((d.fatigue - d.opp_fatigue) >= ML_PEN_EDGE)
    d = d.join(d.groupby("team")["win"].mean().rename("own_rate"), on="team")

    print(f"\nteam-games with a read on both pens: {len(d)}")
    for name, sub in (("gate would demote", d[d.gated]), ("gate lets through", d[~d.gated])):
        se = float(np.sqrt(sub.win.mean() * (1 - sub.win.mean()) / max(len(sub), 1)))
        print(f"  {name}: n={len(sub)} won {sub.win.mean():.3f} vs those teams' own"
              f" {sub.own_rate.mean():.3f} -> {sub.win.mean() - sub.own_rate.mean():+.3f}"
              f" +/- {se:.3f}")
    print(f"  r(own fatigue, win) = {d.fatigue.corr(d.win):+.3f}")
    print(f"  r(fatigue gap, win) = {(d.fatigue - d.opp_fatigue).corr(d.win):+.3f}")
    print(
        d.groupby(pd.cut(d.fatigue, [-1, 19, 39, 59, 79, 101]), observed=True)
        .agg(games=("win", "size"), won=("win", "mean"), own_rate=("own_rate", "mean"))
        .round(3)
        .to_string()
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("forward", "spread", "fatigue", "mlgate"))
    ap.add_argument("--cache", required=True, type=Path, help="Statcast pickle from the engine cache")
    args = ap.parse_args()
    df, teams = _load(args.cache)
    {"forward": forward, "spread": spread, "fatigue": fatigue, "mlgate": mlgate}[args.mode](
        df, teams
    )


if __name__ == "__main__":
    main()
