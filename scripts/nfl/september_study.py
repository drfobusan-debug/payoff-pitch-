"""What the engine does about the first month, and the three answers it got.

The rating's decay clock counts weeks that appear in the panel, so the offseason
is one tick and a week-1 rating is "how you finished last season". This script
tests the three ways of fixing that, walk-forward over 2013-2025 with every
rating fitted on prior weeks only, and scored on weeks 1-4 specifically.

1. ``--offseason`` charge the season boundary extra age. **Worse, monotonically.**

       extra weeks   0       2       4       6       8      12      20
       weeks 1-4   10.306  10.322  10.339  10.359  10.392  10.475  10.698

   Forgetting last season faster is the intuitive fix and it is the wrong one:
   the prior-season evidence is carrying real signal in September.

2. ``--memory`` lengthen the memory instead. Mildly better in September and worse
   over the season, so the shipped 8 weeks stays:

       half-life     6       8      12      16      24      36
       weeks 1-4   10.369  10.306  10.260  10.265  10.314  10.391

3. ``--qb`` add the offseason information the panel already carries: who is
   taking the snaps. This is the one that ships, and the finding is a split.
   Classifying each starter against last season's primary and this season's
   incumbent (:mod:`nfl_engine.features.quarterback`, prior weeks only), and
   regressing the rating's signed error on ``away fill-in - home fill-in``:

       window        n     rating slope      closing line
       all          911   -2.891 (t -6.77)  -0.818 (t -1.98)
       2013-2019    447   -3.528 (t -5.70)  -1.126 (t -1.86)
       2020-2025    464   -2.252 (t -3.82)  -0.458 (t -0.81)
       weeks 5+     729   -3.888 (t -8.21)  -1.640 (t -3.57)
       weeks 1-4    182   +1.083 (t +1.17)  +2.450 (t +2.71)

   A team starting a man who is neither last year's starter nor this year's
   incumbent is overrated by ~4 points, in both eras -- **except in September,
   where the sign reverses**, because before a team has an incumbent "not last
   year's man" means the new starter rather than a backup. A genuine offseason
   change carries no bias at all (home t -0.00, away t -0.06). So the correction
   is conditioned on the week and ships at 75% of the fitted slope, which is what
   the walk-forward full-sample MAE picks.

   It is not a bet: fading the fill-in side covers 50.90% of 890 games, and
   38.20% in weeks 1-4.

Usage::

    python scripts/nfl/september_study.py --offseason
    python scripts/nfl/september_study.py --memory
    python scripts/nfl/september_study.py --qb
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nfl_engine.data import nflverse
from nfl_engine.features import quarterback as qb_mod
from nfl_engine.features import ratings as ratings_mod
from nfl_engine.models import expectation
from scripts.nfl.ratings_study import PANEL_FIRST_SEASON, build_panel, tstat

BREAK_EVEN = 0.524


def grade(
    panel: pd.DataFrame,
    *,
    first: int,
    half_life: float = ratings_mod.HALF_LIFE_WEEKS,
    ridge: float = ratings_mod.RIDGE,
    offseason: float = 0.0,
) -> pd.DataFrame:
    """Rate and grade every game from ``first`` on; ratings-only, no simulator."""
    weeks = panel[["season", "week", "week_index"]].drop_duplicates().sort_values("week_index")
    rows: list[dict[str, object]] = []
    for entry in weeks.itertuples():
        if entry.season < first:
            continue
        history = panel[panel.week_index < entry.week_index]
        book = ratings_mod.fit(
            history,
            asof=entry.week_index,
            asof_season=int(entry.season),
            half_life=half_life,
            ridge=ridge,
            offseason=offseason,
        )
        if not book.is_usable():
            continue
        slate = panel[(panel.week_index == entry.week_index) & panel.is_home].drop_duplicates(
            "game_id"
        )
        for game in slate.itertuples():
            if pd.isna(game.spread_line) or pd.isna(game.total_line):
                continue
            home, away = str(game.home_team), str(game.away_team)
            rows.append(
                {
                    "season": int(game.season),
                    "week": int(game.week),
                    "home": home,
                    "away": away,
                    "home_qb": getattr(game, "home_qb_id", None),
                    "away_qb": getattr(game, "away_qb_id", None),
                    "rating": expectation.rating_margin(book, home, away),
                    "market": float(game.spread_line),
                    "margin": float(game.home_score - game.away_score),
                }
            )
    graded = pd.DataFrame(rows)
    if not graded.empty:
        graded["rating_err"] = graded.rating - graded.margin
        graded["market_err"] = graded.market - graded.margin
    return graded


def _early(graded: pd.DataFrame) -> float:
    early = graded[graded.week <= 4]
    return float(np.abs(early.rating - early.margin).mean())


def _all(graded: pd.DataFrame) -> float:
    return float(np.abs(graded.rating - graded.margin).mean())


def offseason_grid(panel: pd.DataFrame, first: int) -> None:
    print("extra offseason age, scored on out-of-sample margin MAE")
    for extra in (0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0):
        graded = grade(panel, first=first, offseason=extra)
        if graded.empty:
            continue
        print(
            f"  +{extra:4.1f} weeks per boundary  n={len(graded):5d}"
            f"  weeks 1-4 {_early(graded):.3f}  all weeks {_all(graded):.3f}"
        )


def memory_grid(panel: pd.DataFrame, first: int) -> None:
    print("half-life, scored on September against the whole season")
    for half_life in (6.0, 8.0, 12.0, 16.0, 24.0, 36.0):
        graded = grade(panel, first=first, half_life=half_life)
        if graded.empty:
            continue
        print(
            f"  half-life {half_life:5.1f} weeks  n={len(graded):5d}"
            f"  weeks 1-4 {_early(graded):.3f}  all weeks {_all(graded):.3f}"
        )


def _signal(graded: pd.DataFrame, book: qb_mod.StarterBook) -> pd.DataFrame:
    """Signed fill-in indicator: +1 the away side is a fill-in, -1 the home side."""
    out = graded.copy()
    out["home_fill"] = [
        book.is_fill_in(int(r.season), int(r.week), str(r.home), r.home_qb)
        for r in out.itertuples()
    ]
    out["away_fill"] = [
        book.is_fill_in(int(r.season), int(r.week), str(r.away), r.away_qb)
        for r in out.itertuples()
    ]
    out["signal"] = out.away_fill.astype(float) - out.home_fill.astype(float)
    return out


def _windows(tab: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("all", tab),
        ("2013-2019", tab[tab.season <= 2019]),
        ("2020-2025", tab[tab.season >= 2020]),
        ("weeks 5+", tab[tab.week >= 5]),
        ("weeks 1-4", tab[tab.week <= 4]),
    ]


def _status_split(tab: pd.DataFrame, book: qb_mod.StarterBook) -> None:
    """Separate a genuine change from a displaced incumbent, retrospectively.

    'This man is the team's own starter this season' is not knowable at kickoff,
    so this table is a diagnostic and not the shipped test -- but it is the one
    that says *which* kind of change the rating cannot price. The shipped rule
    conditions on the week instead, which is available.
    """
    games = nflverse.games()
    games = games[games.home_score.notna()]
    long = pd.concat(
        [
            games[["season", "home_team", "home_qb_id"]].rename(
                columns={"home_team": "team", "home_qb_id": "qb"}
            ),
            games[["season", "away_team", "away_qb_id"]].rename(
                columns={"away_team": "team", "away_qb_id": "qb"}
            ),
        ]
    ).dropna()
    counts = long.groupby(["season", "team", "qb"]).size().rename("n").reset_index()
    primary = counts.sort_values("n", ascending=False).drop_duplicates(["season", "team"])
    own = {(int(r.season), str(r.team)): str(r.qb) for r in primary.itertuples()}

    def kind(season: int, week: int, team: str, qb: object) -> str:
        status = book.status(season, week, team, qb if isinstance(qb, str) else None)
        if status != qb_mod.FILL_IN:
            return status
        return "new_regular" if own.get((season, team)) == qb else "fill_in"

    rows = []
    for r in tab.itertuples():
        rows.append(
            {
                "rating_err": r.rating_err,
                "market_err": r.market_err,
                "home": kind(int(r.season), int(r.week), str(r.home), r.home_qb),
                "away": kind(int(r.season), int(r.week), str(r.away), r.away_qb),
            }
        )
    split = pd.DataFrame(rows)
    print("\nwhich kind of change? (positive = the rating was too high on the home side)")
    for label, mask in (
        ("home new regular", (split.home == "new_regular") & (split.away == qb_mod.INCUMBENT)),
        ("away new regular", (split.away == "new_regular") & (split.home == qb_mod.INCUMBENT)),
        ("home fill-in", (split.home == "fill_in") & (split.away == qb_mod.INCUMBENT)),
        ("away fill-in", (split.away == "fill_in") & (split.home == qb_mod.INCUMBENT)),
        ("both incumbents", (split.home == qb_mod.INCUMBENT) & (split.away == qb_mod.INCUMBENT)),
    ):
        part = split[mask]
        if len(part) < 40:
            print(f"  {label:18s} n={len(part):4d}  too thin")
            continue
        out = []
        for col in ("rating_err", "market_err"):
            v = part[col]
            se = float(v.std(ddof=1) / np.sqrt(len(v)))
            out.append(f"{float(v.mean()):+6.3f} (t {float(v.mean()) / se:+5.2f})")
        print(f"  {label:18s} n={len(part):4d}  rating {out[0]}   market {out[1]}")


def qb_study(panel: pd.DataFrame, first: int) -> None:
    graded = grade(panel, first=first)
    if graded.empty:
        raise SystemExit("nothing graded")
    book = qb_mod.build(nflverse.games())
    tab = _signal(graded, book)
    live = tab[tab.signal != 0]
    print(f"n={len(tab)} graded, {len(live)} with a fill-in on exactly one side\n")

    print("does the rating overrate the fill-in side, and does the line?")
    for label, part in _windows(live):
        if len(part) < 60:
            print(f"  {label:10s} n={len(part):4d}  too thin")
            continue
        signal = part.signal.to_numpy(dtype=float)
        r_slope, r_t = tstat(signal, part.rating_err.to_numpy(dtype=float))
        m_slope, m_t = tstat(signal, part.market_err.to_numpy(dtype=float))
        print(
            f"  {label:10s} n={len(part):4d}  rating {r_slope:+6.3f} (t {r_t:+5.2f})"
            f"   market {m_slope:+6.3f} (t {m_t:+5.2f})"
        )

    _status_split(tab, book)

    print("\nATS, backing the healthy side -- this is why it is not a bet")
    for label, part in _windows(live):
        keep = part[part.margin != part.market]
        if len(keep) < 60:
            print(f"  {label:10s} n={len(keep):4d}  too thin")
            continue
        won = np.where(keep.signal > 0, keep.margin > keep.market, keep.margin < keep.market)
        rate = float(won.mean())
        roi = rate * (100 / 110) - (1 - rate)
        verdict = "clears" if rate > BREAK_EVEN else "fails"
        print(
            f"  {label:10s} n={len(keep):4d}  win {rate * 100:.2f}%"
            f"  ROI {roi * 100:+.2f}%  {verdict} break-even"
        )

    print("\nthe charge, fitted on prior seasons only and scored on every game")
    for share in (0.25, 0.5, 0.75, 1.0):
        before, after, touched = [], [], 0
        for season in range(first + 3, int(tab.season.max()) + 1):
            hist = tab[(tab.season < season) & (tab.week >= qb_mod.FILL_IN_MIN_WEEK)]
            hist = hist[hist.signal != 0]
            fwd = tab[tab.season == season]
            if len(hist) < 60 or fwd.empty:
                continue
            slope, _ = tstat(
                hist.signal.to_numpy(dtype=float),
                hist.rating_err.to_numpy(dtype=float),
            )
            applied = np.where(fwd.week >= qb_mod.FILL_IN_MIN_WEEK, slope * share * fwd.signal, 0.0)
            touched += int((applied != 0).sum())
            before.extend(np.abs(fwd.rating_err).tolist())
            after.extend(np.abs(fwd.rating_err - applied).tolist())
        if not before:
            continue
        print(
            f"  {int(share * 100):3d}% of the fitted slope ({slope * share:+5.2f} pts"
            f" as of the last season)  margin MAE {np.mean(before):.4f}"
            f" -> {np.mean(after):.4f}  ({touched} games touched)"
        )
    print(f"  shipped: {qb_mod.FILL_IN_MARGIN_POINTS:+.1f} pts from week {qb_mod.FILL_IN_MIN_WEEK}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=2013, help="first graded season")
    parser.add_argument("--last", type=int, default=2025)
    parser.add_argument("--offseason", action="store_true", help="extra-age grid")
    parser.add_argument("--memory", action="store_true", help="half-life grid")
    parser.add_argument("--qb", action="store_true", help="fill-in quarterback study")
    args = parser.parse_args()

    panel = build_panel(PANEL_FIRST_SEASON, args.last)
    print(f"panel: {len(panel)} team-games, {panel.game_id.nunique()} games")
    ran = False
    if args.offseason:
        offseason_grid(panel, args.first)
        ran = True
    if args.memory:
        memory_grid(panel, args.first)
        ran = True
    if args.qb or not ran:
        qb_study(panel, args.first)


if __name__ == "__main__":
    main()
