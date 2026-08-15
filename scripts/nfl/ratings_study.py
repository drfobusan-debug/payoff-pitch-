"""The phase-3 gate: does the rating beat the closing line, out of sample?

Walk-forward over 2010-2025. Every rating is fitted on prior weeks only, priced
through the phase-2 possession simulator, and scored on the four things that
decide whether a forecast is worth betting:

1. **Accuracy** against the closing spread and total (MAE, RMSE).
2. **Calibration** of the moneyline probability (Brier, log loss, decile table).
3. **Record** on ATS, moneyline and totals against the 52.4% a -110 needs.
4. **The residual test** -- regress the closing line's own error on the rating's
   disagreement with it. This is the only one that can distinguish a good
   forecast from a profitable one, because a rating that is right about football
   but agrees with the market where the market is wrong wins nothing.

The verdict, and the reason ``MARKET_WEIGHT`` ships at 1.0:

    n = 3,450 games, 2013-2025 (5,431 games of panel behind them)
    margin MAE      rating 10.282    market  9.905
    total MAE       rating 10.739    market 10.471
    ATS  best slice          50.88%  (2+ points of edge, n=1,712)
    total best slice         52.10%  (3+ points of edge, n=1,192)
    margin residual on rating edge   slope +0.016   t +0.25
    total residual on rating edge    slope +0.080   t +1.16
    moneyline Brier 0.2221 (base rate 0.2470), log loss 0.6349

**The gate fails, and it is not close.** The rating is a competent forecaster --
10.28 points of margin error is what a public EPA model gets, and priced through
the possession simulator its win probabilities are well calibrated -- and it is
still worthless as a bet. It loses to the line it is trying to beat by 0.38
points, its disagreements with that line explain none of the line's error, and no
blend weight beats simply taking the market. The best single slice, totals on 3+
points of edge, is 52.10% against the 52.40% a -110 needs, on the sample most
favourable to it.

Two honest caveats, both of which flatter the rating rather than the market. The
ratings-to-points coefficients are fitted on this whole window rather than
walk-forward (``--fit``), so the rating gets an in-sample advantage the market
does not; and the situational block's wind and divisional coefficients are fitted
on overlapping history. It loses anyway. The one slice with a pulse is weeks 1-4
(slope +0.238, t +1.76), consistent with the market being thinner in September
and not significant.

Usage::

    python scripts/nfl/ratings_study.py                  # the gate table
    python scripts/nfl/ratings_study.py --grid           # refit half-life/ridge
    python scripts/nfl/ratings_study.py --fit            # refit points coefficients
    python scripts/nfl/ratings_study.py --first 2016     # a shorter window
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_engine.data import nflverse
from nfl_engine.features import panel as panel_mod
from nfl_engine.features import ratings as ratings_mod
from nfl_engine.features.adjustments import Situation
from nfl_engine.models.drives import DriveSim
from nfl_engine.models.expectation import forecast

BREAK_EVEN = 0.524
PANEL_FIRST_SEASON = 2006


@dataclass(frozen=True)
class Row:
    """One graded game: what the rating said, what the market said, what happened."""

    season: int
    week: int
    rating_margin: float
    rating_total: float
    market_margin: float
    market_total: float
    margin: float
    total: float
    d_off_epa: float
    d_def_epa: float
    s_off_epa: float
    s_def_epa: float
    home_win_prob: float | None = None
    home_won: float = 0.0


def build_panel(first_season: int, last_season: int) -> pd.DataFrame:
    seasons = list(range(first_season, last_season + 1))
    frame = panel_mod.panel(seasons)
    if frame.empty:
        raise SystemExit("no play-by-play available; cannot run the gate")
    games = nflverse.games()
    joined = panel_mod.with_results(frame, games)
    return ratings_mod.week_index(joined)


def walk_forward(
    panel: pd.DataFrame,
    *,
    first: int,
    half_life: float,
    ridge: float,
    sim: DriveSim | None = None,
) -> pd.DataFrame:
    """Rate, price and grade every game from ``first`` on, prior weeks only."""
    weeks = (
        panel[["season", "week", "week_index"]]
        .drop_duplicates()
        .sort_values("week_index")
    )
    rows: list[Row] = []
    for entry in weeks.itertuples():
        if entry.season < first:
            continue
        history = panel[panel.week_index < entry.week_index]
        book = ratings_mod.fit(
            history, asof=entry.week_index, half_life=half_life, ridge=ridge
        )
        if not book.is_usable():
            continue
        slate = panel[
            (panel.week_index == entry.week_index) & panel.is_home
        ].drop_duplicates("game_id")
        for game in slate.itertuples():
            if pd.isna(game.spread_line) or pd.isna(game.total_line):
                continue
            situation = Situation(
                roof=None if pd.isna(game.roof) else str(game.roof),
                wind_mph=None if pd.isna(game.wind) else float(game.wind),
                temp_f=None if pd.isna(game.temp) else float(game.temp),
                home_rest=None if pd.isna(game.home_rest) else int(game.home_rest),
                away_rest=None if pd.isna(game.away_rest) else int(game.away_rest),
                div_game=bool(game.div_game) if not pd.isna(game.div_game) else False,
            )
            # market_weight=0 so the gate scores the rating itself, not the line.
            view = forecast(
                book,
                str(game.home_team),
                str(game.away_team),
                situation=situation,
                market_margin=float(game.spread_line),
                market_total=float(game.total_line),
                market_weight=0.0,
            )
            prob: float | None = None
            if sim is not None:
                prob = sim.simulate(view.expected_game()).moneyline(home=True).conditional
            home_rating = book.rating(str(game.home_team))
            away_rating = book.rating(str(game.away_team))
            rows.append(
                Row(
                    season=int(game.season),
                    week=int(game.week),
                    rating_margin=view.margin(),
                    rating_total=view.total(),
                    d_off_epa=home_rating.off_epa - away_rating.off_epa,
                    d_def_epa=home_rating.def_epa - away_rating.def_epa,
                    s_off_epa=home_rating.off_epa + away_rating.off_epa,
                    s_def_epa=home_rating.def_epa + away_rating.def_epa,
                    market_margin=float(game.spread_line),
                    market_total=float(game.total_line),
                    margin=float(game.home_score - game.away_score),
                    total=float(game.home_score + game.away_score),
                    home_win_prob=prob,
                    home_won=float(game.home_score > game.away_score),
                )
            )
    return pd.DataFrame([row.__dict__ for row in rows])


def tstat(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    var = (resid @ resid / (len(y) - 2)) * np.linalg.inv(design.T @ design)[1, 1]
    return float(beta[1]), float(beta[1] / np.sqrt(var))


def _record(edge: np.ndarray, resid: np.ndarray, label: str) -> None:
    """Win rate by size of edge, pushes dropped rather than counted as wins."""
    for threshold in (0.0, 1.0, 2.0, 3.0):
        pick = (np.abs(edge) > threshold) & (resid != 0)
        if pick.sum() < 100:
            continue
        wins = float((np.sign(edge[pick]) == np.sign(resid[pick])).mean())
        roi = wins * (100 / 110) - (1 - wins)
        verdict = "clears" if wins > BREAK_EVEN else "fails"
        print(
            f"    {label} edge>{threshold:.0f}  n={int(pick.sum()):5d}"
            f"  win {wins*100:.2f}%  ROI {roi*100:+.2f}%  {verdict} break-even"
        )


def report(graded: pd.DataFrame) -> None:
    margin = graded.margin.to_numpy(dtype=float)
    total = graded.total.to_numpy(dtype=float)
    r_margin = graded.rating_margin.to_numpy(dtype=float)
    r_total = graded.rating_total.to_numpy(dtype=float)
    m_margin = graded.market_margin.to_numpy(dtype=float)
    m_total = graded.total.to_numpy(dtype=float) * 0 + graded.market_total.to_numpy(dtype=float)
    print(f"n={len(graded)}  seasons {graded.season.min()}-{graded.season.max()}")
    print("  accuracy (points of error)")
    print(
        f"    margin  rating MAE {np.abs(margin - r_margin).mean():.3f}"
        f"   market MAE {np.abs(margin - m_margin).mean():.3f}"
    )
    print(
        f"    total   rating MAE {np.abs(total - r_total).mean():.3f}"
        f"   market MAE {np.abs(total - m_total).mean():.3f}"
    )
    print("  blends of the two (this is what chooses MARKET_WEIGHT)")
    for weight in (0.0, 0.25, 0.55, 0.75, 0.9, 1.0):
        blend = weight * m_margin + (1.0 - weight) * r_margin
        print(
            f"    w_market={weight:.2f}  margin MAE {np.abs(margin - blend).mean():.3f}"
        )

    print("  the residual test: does the rating explain the line's own error?")
    for label, edge, resid in (
        ("margin", r_margin - m_margin, margin - m_margin),
        ("total ", r_total - m_total, total - m_total),
    ):
        slope, t = tstat(edge, resid)
        print(f"    {label}  slope {slope:+.3f}  t {t:+.2f}")

    print("  record against the price")
    _record(r_margin - m_margin, margin - m_margin, "ATS  ")
    _record(r_total - m_total, total - m_total, "total")

    print("  by week bucket (is the market thinner in September?)")
    for lo, hi, name in ((1, 4, "weeks 1-4"), (5, 9, "weeks 5-9"), (10, 23, "weeks 10+")):
        sub = graded[(graded.week >= lo) & (graded.week <= hi)]
        if sub.empty:
            continue
        edge = sub.rating_margin.to_numpy(dtype=float) - sub.market_margin.to_numpy(dtype=float)
        resid = sub.margin.to_numpy(dtype=float) - sub.market_margin.to_numpy(dtype=float)
        slope, t = tstat(edge, resid)
        live = resid != 0
        ats = float((np.sign(edge[live]) == np.sign(resid[live])).mean())
        print(
            f"    {name:9s} n={len(sub):5d}  slope {slope:+.3f}  t {t:+.2f}"
            f"  ATS {ats*100:.2f}%"
        )

    if graded.home_win_prob.notna().any():
        _report_probability(graded.dropna(subset=["home_win_prob"]))


def _report_probability(graded: pd.DataFrame) -> None:
    prob = graded.home_win_prob.to_numpy(dtype=float)
    won = graded.home_won.to_numpy(dtype=float)
    clipped = np.clip(prob, 1e-6, 1 - 1e-6)
    brier = float(np.mean((prob - won) ** 2))
    logloss = float(-np.mean(won * np.log(clipped) + (1 - won) * np.log(1 - clipped)))
    base = float(won.mean())
    print("  moneyline probability, from the possession simulator")
    print(
        f"    Brier {brier:.4f}  log loss {logloss:.4f}"
        f"   (always predicting the base rate {base:.3f}: Brier {base*(1-base):.4f})"
    )
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, 9)
    print("    calibration")
    for b in range(10):
        pick = idx == b
        if pick.sum() < 30:
            continue
        print(
            f"      {bins[b]:.1f}-{bins[b+1]:.1f}  n={int(pick.sum()):5d}"
            f"  predicted {prob[pick].mean():.3f}  actual {won[pick].mean():.3f}"
        )


def grid(panel: pd.DataFrame, first: int) -> None:
    print("half-life / ridge grid, scored on out-of-sample margin MAE")
    for half_life in (4.0, 6.0, 8.0, 12.0):
        for ridge in (600.0, 900.0, 1300.0):
            graded = walk_forward(panel, first=first, half_life=half_life, ridge=ridge)
            if graded.empty:
                continue
            margin = graded.margin.to_numpy(dtype=float)
            edge = graded.rating_margin.to_numpy(dtype=float) - graded.market_margin.to_numpy(dtype=float)
            resid = margin - graded.market_margin.to_numpy(dtype=float)
            _, t = tstat(edge, resid)
            mae = float(np.abs(margin - graded.rating_margin.to_numpy(dtype=float)).mean())
            print(
                f"  half_life={half_life:5.1f} ridge={ridge:6.1f}"
                f"  n={len(graded):5d}  MAE {mae:.3f}  residual t {t:+.2f}"
            )


def fit_points(graded: pd.DataFrame) -> None:
    """Refit the ratings-to-points coefficients in nfl_engine.models.expectation."""
    for label, target, cols in (
        ("margin", "margin", ("d_off_epa", "d_def_epa")),
        ("total", "total", ("s_off_epa", "s_def_epa")),
    ):
        design = np.column_stack(
            [np.ones(len(graded))] + [graded[c].to_numpy(dtype=float) for c in cols]
        )
        y = graded[target].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        var = (resid @ resid / (len(y) - design.shape[1])) * np.linalg.inv(
            design.T @ design
        )
        se = np.sqrt(np.diag(var))
        print(f"  {label}: n={len(graded)} resid sd {resid.std(ddof=design.shape[1]):.3f}")
        for name, value, err in zip(("intercept",) + cols, beta, se, strict=True):
            print(f"    {name:12s} {value:+9.2f}  t {value / err:+6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=2013, help="first graded season")
    parser.add_argument("--last", type=int, default=2025)
    parser.add_argument("--grid", action="store_true", help="refit half-life and ridge")
    parser.add_argument(
        "--fit", action="store_true", help="refit the ratings-to-points coefficients"
    )
    parser.add_argument(
        "--no-sim",
        action="store_true",
        help="skip the possession simulator (much faster, no calibration table)",
    )
    parser.add_argument("--sims", type=int, default=8000)
    args = parser.parse_args()

    panel = build_panel(PANEL_FIRST_SEASON, args.last)
    print(f"panel: {len(panel)} team-games, {panel.game_id.nunique()} games")
    if args.grid:
        grid(panel, args.first)
        return
    sim = None if (args.no_sim or args.fit) else DriveSim(n_sims=args.sims)
    graded = walk_forward(
        panel,
        first=args.first,
        half_life=ratings_mod.HALF_LIFE_WEEKS,
        ridge=ratings_mod.RIDGE,
        sim=sim,
    )
    if graded.empty:
        raise SystemExit("nothing graded")
    if args.fit:
        fit_points(graded)
        return
    report(graded)


if __name__ == "__main__":
    main()
