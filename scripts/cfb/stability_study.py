"""Does the roster-stability haircut earn its place, and should it fade?

The engine scales every rating gap by ``regression_factor`` (0.90) and then again
by :func:`cfb_engine.data.preseason.stability_factor`, which keeps as little as
75% of the gap when both rosters are volatile. Neither term has a week clock, so
a rebuilt roster is still discounted in week 14 after nine games of evidence.
This script measures both, walk-forward, and the answer is that the week clock is
not the problem -- the haircut is.

Instruments
-----------
Point-in-time ratings are the whole difficulty: SP+, FPI and Sagarin publish only
today's numbers, so a week-by-week study cannot use them. Two that *are*
retrievable:

* **TeamRankings predictive** (``--source tr``), an actual ensemble member,
  served dated via ``?date=``. Each game is matched to the last Saturday snapshot
  before kickoff.
* **The repo's own PPA ridge** (``--source eff``), refit each week on games played
  strictly before the slate week.

Roster continuity is CFBD returning production, used as terciles. The baked-in
VSiN 0-19 stability score exists for one season only, so it has no history to fit
against; the two agree at r=+0.43 on 2026, which is what makes returning
production the usable stand-in.

What it found (2014-2025, 6,818 TR-matched non-neutral games)
-------------------------------------------------------------
The fraction of the published gap the data supports, ``margin ~ a + b*gap``::

    weeks 1-2  3-4    5-7    8-10   11-15    all
    1.029   0.922  0.870  0.909  0.963    0.936

Shallow U, not a ramp: the gap is worth *most* in September, when the shipped
0.90 shrink is closest to wrong in the opposite direction. A fitted per-week
shrink is worth 0.011 MAE walk-forward, which is nothing.

By continuity, and this is the finding::

    low continuity   b=0.916    weeks 1-4 0.960   5-9 0.853   10-15 0.917
    high continuity  b=0.968    weeks 1-4 0.972   5-9 0.914   10-15 1.019

The effect is real, in the expected direction, and about 0.05 of the gap wide --
against a shipped haircut up to 0.25 wide. It is also *smallest in September*
(0.012) and largest in November (0.102), which is backwards from a term whose
justification is preseason uncertainty. Walk-forward, more haircut is monotonically
worse::

    policy              rating MAE   blend MAE   ATS n    win%
    no haircut (flat)      12.549      12.321    3336   49.46%
    keep floor 0.95        12.556      12.323    3348   49.49%
    keep floor 0.90        12.572      12.329    3393   49.43%
    keep floor 0.75 (ship) 12.654      12.365    3482   49.45%

so the haircut now defaults off. The PPA-ridge instrument agrees (13.173 flat vs
13.243 shipped). Nothing here beats the market: market-only MAE is 12.127 against
12.31-12.37 for every blended policy, and no policy clears break-even ATS.

Usage::

    CFBD_API_KEY=... python scripts/cfb/stability_study.py --source tr
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import requests

from cfb_engine.data.efficiency import EfficiencyBook, fit_efficiency
from cfb_engine.data.teamnames import school_key

CACHE = pathlib.Path(os.getenv("CFBE_STUDY_CACHE", pathlib.Path.home() / "cfb_cache"))
TR_URL = "https://www.teamrankings.com/college-football/ranking/predictive-by-other"
CFBD = "https://api.collegefootballdata.com"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120"}
SEASONS = list(range(2014, 2026))

BLEND = 0.35  # cfb_engine.config.ModelParams.market_blend
FLAT = 0.90  # cfb_engine.config.FeatureParams.regression_factor
SHIPPED_FLOOR = 0.75  # cfb_engine.data.preseason stability floor


# -- data ------------------------------------------------------------------
def _cfbd(path: str, name: str, **params: str | int) -> list[dict]:
    dest = CACHE / f"{name}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    key = os.environ["CFBD_API_KEY"]
    for attempt in range(4):
        resp = requests.get(
            f"{CFBD}{path}", params=params, headers={"Authorization": f"Bearer {key}"}, timeout=120
        )
        if resp.ok:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(resp.json()))
            return resp.json()
        time.sleep(3 * (attempt + 1))
    raise SystemExit(f"CFBD request failed: {name}")


def _field(row: dict, *keys: str):
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def games(season: int) -> list[dict]:
    out = []
    for row in _cfbd("/games", f"games_{season}", year=season, seasonType="regular"):
        home_pts = _field(row, "homePoints", "home_points")
        away_pts = _field(row, "awayPoints", "away_points")
        start = _field(row, "startDate", "start_date")
        if home_pts is None or away_pts is None or start is None:
            continue
        out.append(
            {
                "id": int(_field(row, "id", "gameId")),
                "season": season,
                "week": int(_field(row, "week") or 0),
                "date": dt.date.fromisoformat(str(start)[:10]),
                "home": school_key(str(_field(row, "homeTeam", "home_team"))),
                "away": school_key(str(_field(row, "awayTeam", "away_team"))),
                "margin": float(home_pts) - float(away_pts),
                "neutral": bool(_field(row, "neutralSite", "neutral_site") or False),
            }
        )
    return out


def spreads(season: int) -> dict[int, float]:
    """Median closing spread per game, home perspective (negative = home favoured)."""
    out: dict[int, float] = {}
    for row in _cfbd("/lines", f"lines_{season}", year=season, seasonType="regular"):
        gid = _field(row, "id", "gameId")
        vals = []
        for line in row.get("lines") or []:
            try:
                vals.append(float(line["spread"]))
            except (KeyError, TypeError, ValueError):
                continue
        if gid is not None and vals:
            out[int(gid)] = float(np.median(vals))
    return out


def returning(season: int) -> dict[str, float]:
    rows = _cfbd("/player/returning", f"returning_{season}", year=season)
    return {
        school_key(str(r["team"])): float(r["percentPPA"])
        for r in rows
        if r.get("team") and r.get("percentPPA") is not None
    }


def saturdays(season: int) -> list[str]:
    day = dt.date(season, 8, 20)
    day += dt.timedelta(days=(5 - day.weekday()) % 7)
    out = []
    while day < dt.date(season, 12, 10):
        out.append(day.isoformat())
        day += dt.timedelta(days=7)
    return out


def tr_snapshot(date: str) -> dict[str, float]:
    path = CACHE / "tr" / f"{date}.html"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(TR_URL, params={"date": date}, headers=UA, timeout=60)
        resp.raise_for_status()
        path.write_text(resp.text)
        time.sleep(1.0)
    try:
        table = pd.read_html(path)[0]
    except ValueError:
        return {}
    names = table.Team.astype(str).str.replace(r"\s*\(\d+-\d+(-\d+)?\)$", "", regex=True)
    ratings = pd.to_numeric(table.Rating, errors="coerce")
    return {school_key(n): float(v) for n, v in zip(names, ratings, strict=False) if pd.notna(v)}


def tr_rows() -> list[dict]:
    rows = []
    for season in SEASONS:
        books = {d: tr_snapshot(d) for d in saturdays(season)}
        dates = sorted(books)
        line, ret = spreads(season), returning(season)
        for game in games(season):
            i = bisect.bisect_left(dates, game["date"].isoformat()) - 1
            if i < 0:
                continue
            book = books[dates[i]]
            home, away = book.get(game["home"]), book.get(game["away"])
            if home is None or away is None:
                continue
            rows.append(
                {
                    **game,
                    "gap": home - away,
                    "spread": line.get(game["id"]),
                    "ret_home": ret.get(game["home"]),
                    "ret_away": ret.get(game["away"]),
                }
            )
    return rows


def eff_rows() -> list[dict]:
    from cfb_engine.data.efficiency import POINTS_PER_PPA

    rows = []
    for season in SEASONS:
        season_games = games(season)
        home_of = {g["id"]: g["home"] for g in season_games}
        per_week: dict[int, list[tuple[str, str, float, float]]] = {}
        for row in _cfbd("/ppa/games", f"ppa_{season}", year=season):
            gid = _field(row, "gameId", "game_id")
            ppa = (row.get("offense") or {}).get("overall")
            if gid is None or ppa is None or not row.get("team") or not row.get("opponent"):
                continue
            team = school_key(str(row["team"]))
            site = 1.0 if home_of.get(int(gid)) == team else -1.0
            week = int(_field(row, "week") or 0)
            per_week.setdefault(week, []).append(
                (team, school_key(str(row["opponent"])), site, float(ppa))
            )
        books: dict[int, EfficiencyBook | None] = {}
        seen: list[tuple[str, str, float, float]] = []
        for week in sorted(per_week):
            books[week] = fit_efficiency(list(seen)) if seen else None
            seen.extend(per_week[week])
        line, ret = spreads(season), returning(season)
        for game in season_games:
            book = books.get(game["week"])
            if book is None:
                continue
            home, away = book.get(game["home"]), book.get(game["away"])
            if home is None or away is None:
                continue
            rows.append(
                {
                    **game,
                    "gap": (home.net - away.net) * POINTS_PER_PPA,
                    "spread": line.get(game["id"]),
                    "ret_home": ret.get(game["home"]),
                    "ret_away": ret.get(game["away"]),
                }
            )
    return rows


# -- fitting ---------------------------------------------------------------
def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    """Slope, intercept, t(slope) and n for ``y ~ a + b x``."""
    n = len(y)
    if n < 30:
        return float("nan"), float("nan"), float("nan"), n
    design = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    var = float(resid @ resid) / (n - 2)
    cov = var * np.linalg.inv(design.T @ design)
    return float(coef[1]), float(coef[0]), float(coef[1] / np.sqrt(cov[1, 1])), n


def _line(rows: list[dict], label: str) -> None:
    b, a, t, n = ols(np.array([r["gap"] for r in rows]), np.array([r["margin"] for r in rows]))
    print(f"{label:<18} n={n:5d}  b={b:.3f} (t {t:+.1f})  hfa={a:+.2f}")


def by_week(rows: list[dict]) -> None:
    print("\n=== margin ~ a + b*gap  (b = fraction of the rating gap the data supports)")
    _line(rows, "all weeks")
    for lo, hi in ((1, 2), (3, 4), (5, 7), (8, 10), (11, 15)):
        sel = [r for r in rows if lo <= r["week"] <= hi]
        if len(sel) >= 50:
            _line(sel, f"weeks {lo}-{hi}")


def by_continuity(rows: list[dict]) -> None:
    have = [r for r in rows if r["ret_home"] is not None and r["ret_away"] is not None]
    cont = np.array([(r["ret_home"] + r["ret_away"]) / 2 for r in have])
    lo_cut, hi_cut = np.percentile(cont, [33, 67])
    print("\n=== b by roster continuity (returning production, terciles)")
    for label, mask in (
        ("low continuity", cont <= lo_cut),
        ("mid continuity", (cont > lo_cut) & (cont < hi_cut)),
        ("high continuity", cont >= hi_cut),
    ):
        sel = [r for r, m in zip(have, mask, strict=True) if m]
        _line(sel, label)
        for lo, hi in ((1, 4), (5, 9), (10, 15)):
            sub = [r for r in sel if lo <= r["week"] <= hi]
            if len(sub) >= 50:
                _line(sub, f"  weeks {lo}-{hi}")


POLICIES = ("no haircut", "floor 0.95", "floor 0.90", "floor 0.75 (shipped)", "fitted week")


def _predict(policy: str, row: dict, tier: int, fits: dict) -> float:
    gap, hfa = row["gap"], fits["hfa"]
    if policy == "fitted week":
        bucket = 0 if row["week"] <= 4 else (1 if row["week"] <= 9 else 2)
        b, a = fits["per_week"][bucket]
        return b * gap + a
    if policy == "no haircut":
        return FLAT * gap + hfa
    floor = SHIPPED_FLOOR if "shipped" in policy else float(policy.split()[1])
    reliability = (0.25, 0.6, 0.95)[tier]
    return FLAT * (floor + (1 - floor) * reliability) * gap + hfa


def walk_forward(rows: list[dict]) -> None:
    graded = [
        r
        for r in rows
        if r["spread"] is not None and r["ret_home"] is not None and r["ret_away"] is not None
    ]
    stats: dict[str, dict[str, list[float]]] = {
        p: {"rating": [], "blend": [], "ats": []} for p in POLICIES
    }
    market_abs: list[float] = []
    for season in SEASONS[3:]:
        train = [r for r in graded if r["season"] < season]
        test = [r for r in graded if r["season"] == season]
        if len(train) < 500 or not test:
            continue
        b_all, hfa, _, _ = ols(
            np.array([r["gap"] for r in train]), np.array([r["margin"] for r in train])
        )
        per_week = {}
        for bucket, (lo, hi) in enumerate(((1, 4), (5, 9), (10, 15))):
            sel = [r for r in train if lo <= r["week"] <= hi]
            per_week[bucket] = (
                ols(np.array([r["gap"] for r in sel]), np.array([r["margin"] for r in sel]))[:2]
                if len(sel) >= 200
                else (b_all, hfa)
            )
        fits = {"hfa": hfa, "b_all": b_all, "per_week": per_week}
        cont = np.array([(r["ret_home"] + r["ret_away"]) / 2 for r in train])
        cuts = np.percentile(cont, [33, 67])
        for row in test:
            market = -row["spread"]
            market_abs.append(abs(market - row["margin"]))
            avg = (row["ret_home"] + row["ret_away"]) / 2
            tier = 0 if avg <= cuts[0] else (2 if avg >= cuts[1] else 1)
            for policy in POLICIES:
                pred = _predict(policy, row, tier, fits)
                blend = (1 - BLEND) * pred + BLEND * market
                stats[policy]["rating"].append(abs(pred - row["margin"]))
                stats[policy]["blend"].append(abs(blend - row["margin"]))
                edge = blend - market
                if abs(edge) >= 1.0:
                    cover = (row["margin"] - market) * (1.0 if edge > 0 else -1.0)
                    if abs(cover) > 1e-9:
                        stats[policy]["ats"].append(1.0 if cover > 0 else 0.0)
    print(f"\n=== walk-forward, market-only MAE {np.mean(market_abs):.3f} (n={len(market_abs)})")
    print(f"{'policy':<22}{'rating MAE':>11}{'blend MAE':>11}{'ATS n':>8}{'win%':>8}{'ROI':>8}")
    for policy in POLICIES:
        s = stats[policy]
        ats = np.array(s["ats"])
        win = float(ats.mean()) if len(ats) else float("nan")
        roi = (win * (100 / 110) - (1 - win)) * 100
        print(
            f"{policy:<22}{np.mean(s['rating']):>11.3f}{np.mean(s['blend']):>11.3f}"
            f"{len(ats):>8d}{win * 100:>7.2f}%{roi:>7.2f}%"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("tr", "eff"), default="tr")
    args = parser.parse_args(argv)
    rows = [r for r in (tr_rows() if args.source == "tr" else eff_rows()) if not r["neutral"]]
    print(f"{len(rows)} graded non-neutral games with a point-in-time {args.source} rating")
    by_week(rows)
    by_continuity(rows)
    walk_forward(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
