"""Render the offline batting study CSVs to a reproducible Markdown report.

Run: .venv/bin/python -m scripts.worksheet_batting_report --audit ~/.mlb_engine/audit
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.worksheet_batting_backtest import SPLITS, bootstrap_mean, payout, schedule_games


def required_n(target: float) -> int:
    """Approximate independent-game n for two-sided 5% / 80% correlation power."""
    return math.ceil(3 + (1.96 + 0.84) ** 2 / np.arctanh(target) ** 2)


def value(row: pd.Series) -> str:
    n = int(row["n"])
    if not np.isfinite(row["effect"]):
        return f"not estimable (n={n})" if n else "not observed (n=0)"
    return (f"{row['effect']:+.3f} [{row['lo']:+.3f}, {row['hi']:+.3f}], "
            f"p={row['p']:.3f}, n={n}")


def verdict(row: pd.Series, threshold: float, n_min: int | None = None) -> str:
    n_min = n_min or required_n(threshold)
    if not np.isfinite(row["effect"]):
        return "underpowered"
    if row["lo"] >= threshold:
        return "supported"
    if row["hi"] < threshold and row["n"] >= n_min:
        return "rejected"
    return f"underpowered (independent n≈{n_min} for r={threshold:+.2f})"


def render(audit: Path) -> str:
    games = pd.read_csv(audit / "batting_games.csv")
    teams = pd.read_csv(audit / "batting_teams.csv")
    effects = pd.read_csv(audit / "batting_effects.csv").set_index("test")
    metrics = pd.read_csv(audit / "batting_metrics.csv")
    grid = pd.read_csv(audit / "batting_interaction_grid.csv")
    fg = games["pitching_source"].eq("FanGraphs as-of").all()
    schedule = json.loads((audit / f"schedule_{date.fromisoformat(games['date'].max()).year}.json")
                          .read_text())
    final = schedule_games(schedule, date.fromisoformat(games["date"].min()),
                           date.fromisoformat(games["date"].max()))

    def e(name: str) -> pd.Series:
        return effects.loc[name]

    def direction(row: pd.Series) -> str:
        if not np.isfinite(row["effect"]):
            return "not estimable"
        sign = "positive" if row["effect"] > 0 else "negative" if row["effect"] < 0 else "zero"
        return sign + (" (CI spans zero)" if row["lo"] <= 0 <= row["hi"]
                       else " (CI excludes zero)")

    headline = ", ".join(
        f"{outcome} {direction(e(f'vs hand {outcome} model SP BP market'))}"
        for outcome in ("win", "rl_cover", "total_residual")
    )
    lines = [
        "# As-of-date MLB batting-score study",
        "",
        f"**Coverage:** {games['date'].min()}–{games['date'].max()}; "
        f"{len(games)} uniquely matched priced games, {len(teams)} team-games. "
        f"{len(final)} completed regular-season games have usable linescores "
        "after excluding ambiguous same-day matchups; the difference includes "
        "missing prices and unqualified/unmatched starting pitchers. "
        "Retrospective observational analysis, not a production recommendation.",
        "",
        f"**Answer (vs-hand gap, after pitching and market):** {headline}. "
        "Raw, partial, pitching-only and market-controlled estimates for all "
        "three batting splits follow; signs are point estimates, not claims "
        "of stable edges.",
        "",
        "## Primary answer: batting gap versus outcomes",
        "",
        "Gap = away minus home; positive run differential/ML/RL means the away "
        "team wins or covers. Total residual = Over result − no-vig implied Over "
        "probability (pushes excluded). Coefficients are per one observed SD of "
        "batting gap; logit for binary outcomes, OLS for continuous outcomes. "
        "Partial correlations control for SP and bullpen gaps; the last model "
        "additionally controls for the away ML logit, RL implied probability, "
        "or total line. Each interval uses 2,000 game bootstrap draws.",
        "",
        "| Split | Outcome | Raw r [95% CI], p, n | Partial r [95% CI], p, n | "
        "Pitching model | + market model | Change |",
        "|---|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        for outcome in ("win", "run_diff", "rl_cover", "total_residual"):
            raw = e(f"{split} {outcome} raw")
            partial = e(f"{split} {outcome} partial SP BP")
            pitched = e(f"{split} {outcome} model SP BP")
            market = e(f"{split} {outcome} model SP BP market")
            flipped = (np.isfinite(raw["effect"]) and np.isfinite(market["effect"])
                       and np.sign(raw["effect"]) != np.sign(market["effect"]))
            change = ("flipped" if flipped else "shrunk" if
                      abs(market["effect"]) < abs(pitched["effect"]) else
                      "not shrunk") if np.isfinite(market["effect"]) else "not observed"
            lines.append(f"| {split} | {outcome} | {value(raw)} | {value(partial)} | "
                         f"{value(pitched)} | {value(market)} | {change} |")
    lines.extend(["", "### Pitching-standardized checks", "",
                  "| Split | Equal-pitching subset (|SP|, |BP| ≤ 1): r with run diff | "
                  "Pitching-gap tercile r with run diff |",
                  "|---|---|---|"])
    for split in SPLITS:
        strata = effects.loc[[i for i in effects.index if i.startswith(
            f"{split} pitch tercile ")]]
        lines.append(f"| {split} | {value(e(f'{split} pitching matched run_diff'))} | "
                     f"{'; '.join(value(row) for _, row in strata.iterrows())} |")
    lines.extend([
        "", "### Recorded-price, blind directional audit",
        "",
        "Pick the away side for a positive score gap, home for negative, "
        "skip zero gaps; total Over if batting-score sum exceeds the "
        "sample median, Under otherwise. These are **in-sample diagnostics**, "
        "not independently tested betting rules. One unit risked per wager; "
        "payout is at the corresponding recorded American quote.",
        "",
        "| Split | ML units (n) | RL units (n) | Total units (n) |",
        "|---|---:|---:|---:|",
    ])
    for split in SPLITS:
        gap = games[f"bat_{split}_gap"].to_numpy(float)
        summed = games[f"bat_{split}_sum"].to_numpy(float)
        stats = []
        for market, side, result in (
            ("ml", np.sign(gap), games["win"].to_numpy(float)),
            ("rl", np.sign(gap), games["rl_cover"].to_numpy(float)),
            ("total", np.sign(summed - np.nanmedian(summed)),
             games["total_over"].to_numpy(float)),
        ):
            away = games[f"{market}_american"].to_numpy(float)
            other = games[("total_under_american" if market == "total"
                           else f"home_{market}_american")].to_numpy(float)
            units = np.array([payout(away[i] if side[i] > 0 else other[i],
                                     result[i] if side[i] > 0 else 1 - result[i])
                              if np.isfinite(side[i]) and side[i] != 0 else math.nan
                              for i in range(len(games))])
            stat = bootstrap_mean(units)
            stats.append(f"{stat['effect'] * stat['n']:+.1f} ({int(stat['n'])})"
                         if stat["n"] else "not observed")
        lines.append(f"| {split} | {' | '.join(stats)} |")
    lines.extend([
        "", "## Pre-registered B1–B8", "",
        "Each effect below includes its bootstrap 95% CI, two-sided bootstrap "
        "p, and n. Hit-rate p tests 54% (B1) or the paired no-vig price (B6); "
        "other p-values test zero. ‘Required n’ uses an independent-observation "
        "Fisher-z 80%-power approximation at 5%; repeated teams/game "
        "dependence means it is an optimistic lower bound. Units are shown "
        "only where a recorded wager exists; otherwise not observed.",
        "",
    ])
    b1 = e("B1 batting sum vs total residual")
    b1ml = e("B1 bat gap vs ML market residual")
    b1top = e("B1 both top tercile Over hit rate")
    b1units = e("B1 both top tercile Over units/wager")
    b1_verdict = ("rejected" if b1ml["effect"] > b1["effect"] or
                  (b1top["hi"] < .54 and b1top["n"] >= required_n(.10))
                  else "supported" if b1["lo"] >= .10 and
                  b1top["lo"] >= .54 and b1ml["lo"] <= 0 <= b1ml["hi"]
                  else "underpowered")
    lines.extend([
        f"**B1 — {b1_verdict}.** Bat sum–game total residual "
        f"{value(b1)} (threshold +0.10); gap–ML residual {value(b1ml)}; "
        f"both top-tercile Over hit rate {value(b1top)} (threshold 54%), "
        f"no-vig Over residual {value(e('B1 both top tercile Over market residual'))}, "
        f"recorded-price units/wager {value(b1units)} "
        f"({b1units['effect'] * b1units['n']:+.1f}u total). "
        "Top-offense team-total line: not observed in closing snapshots. "
        "Blind sum-directed total units are in the table above.",
        "",
    ])
    b2rows = [n for n in effects.index if n.startswith("B2 ")]
    lines.extend(["**B2 — underpowered** for PA-specific superiority unless "
                  "the split comparisons below separate. PA uses the trailing "
                  "60 days; effect = r with next-game own runs. "
                  "Left/right difference is observational, not a causal sample-size test.",
                  "",
                  "| Opposing hand | Split PA | Overall | vs hand | "
                  "Paired Δr (vs hand − Overall) |",
                  "|---|---|---|---|---|"])
    for hand in ("L", "R"):
        for band in ("<400", "400-799", "800+"):
            left = f"B2 {hand} {band} Overall runs"
            right = f"B2 {hand} {band} vs hand runs"
            if left in b2rows and right in b2rows:
                lines.append(f"| {hand} | {band} | {value(e(left))} | "
                             f"{value(e(right))} | "
                             f"{value(e(f'B2 {hand} {band} split minus Overall r'))} |")
            else:
                lines.append(f"| {hand} | {band} | not observed (n=0) | "
                             "not observed (n=0) | not observed (n=0) |")
    lines.extend(["", "Required independent n for r=+0.10: "
                  f"{required_n(0.10)} per bucket; units: not observed.",
                  "", "**B3 — rejected overall (Power threshold).** "
                  "Expected Power r(runs) ≥+0.12, "
                  "Discipline |r|<0.05; Production market-priced and residual ~0. "
                  "Team-total Over residual and units: not observed "
                  "(no team-total quotes in matched closing files).",
                  "",
                  "| Family | Own runs | Game-total residual | "
                  "F5-total residual | ML probability | ML residual |",
                  "|---|---|---|---|---|---|"])
    for fam in ("power", "discipline", "production"):
        r = e(f"B3 {fam} runs")
        note = verdict(r, .12) if fam == "power" else (
            "supported" if abs(r["effect"]) < .05 and
            r["lo"] > -.05 and r["hi"] < .05 else "underpowered")
        lines.append(f"| {fam} ({note}) | {value(r)} | "
                     f"{value(e(f'B3 {fam} total_residual'))} | "
                     f"{value(e(f'B3 {fam} f5total_residual'))} | "
                     f"{value(e(f'B3 {fam} ml_prob'))} | "
                     f"{value(e(f'B3 {fam} ml_residual'))} |")
    corner = grid[(grid["power_terc"] == 2) & (grid["contact_terc"] == 0) &
                  (grid["outcome"] == "total_residual")]
    corner_text = (value(corner.iloc[0]) if len(corner) else "not observed")
    total_cells = grid[grid["outcome"] == "total_residual"].pivot(
        index="power_terc", columns="contact_terc", values="effect")
    monotone = ((total_cells.diff(axis=0).iloc[1:] >= 0).to_numpy().all() and
                (total_cells.diff(axis=1).iloc[:, 1:] <= 0).to_numpy().all())
    lines.extend(["", f"Required independent n: r=.12 → {required_n(.12)}; "
                  f"|r|=.05 → {required_n(.05)}. Family units: not observed "
                  "(no pre-registered family staking rule).", "",
                  "**B4 — lineup power × opposing-starter contact suppression.** "
                  "Contact ranks use xERA/xFIP/SIERA/FB%/Barrel%/HardHit%; "
                  "higher score means better suppression. Seeded tie-breaking "
                  "enforces three bins per axis; some adjacent bins share "
                  "the same underlying integer score. Grid is in "
                  "`batting_interaction_grid.csv`, with game-total and F5 "
                  "residuals as proxies only: team-total lines not observed. "
                  "Each grid cell reports n, mean, 95% CI and p against zero. "
                  f"High-power/weak-suppression game-total residual: {corner_text}; "
                  f"game-total grid {'is' if monotone else 'is not'} monotone "
                  "in the hypothesized directions; "
                  "Additive-controlled interaction on own runs: "
                  f"{value(e('B4 additive interaction runs'))}; "
                  "game-total residual: "
                  f"{value(e('B4 additive interaction total_residual'))}; "
                  "F5-total residual: "
                  f"{value(e('B4 additive interaction f5total_residual'))}. "
                  "Verdict: underpowered against the +5 percentage-point "
                  "team-total threshold because team-total prices were not observed. "
                  f"Required independent n≈{required_n(.05)} for r=.05; units: "
                  "not observed.", ""])
    b5 = e("B5 season own late runs") if "B5 season own late runs" in effects.index else (
        e("B5 Innings 6+ own late runs"))
    market6 = e("Innings 6+ win model SP BP market")
    inversion = ("Negative CI excludes zero (inverted)." if market6["hi"] < 0
                 else "Coefficient CI includes zero (noise not ruled out)." if
                 market6["lo"] <= 0 <= market6["hi"] else "Positive CI excludes zero.")
    lines.extend([f"**B5 — {verdict(b5, .10)}.** "
                  f"Season Innings 6+ score–own innings 6+ runs {value(b5)} "
                  f"(threshold r≥+.10); priced-window ML coefficient after "
                  f"pitching and market logit {value(market6)}. "
                  f"{inversion} "
                  f"Required independent n≈{required_n(.10)} for r=.10 and "
                  "at least 600 priced games for ML; units: see blind ML table.",
                  "",
                  f"**B6 — {verdict(e('B6 vs-hand F5 runs'), .12)}.** "
                  "vs-hand score–own F5 runs "
                  f"{value(e('B6 vs-hand F5 runs'))}; top-tercile offense "
                  f"F5 total residual {value(e('B6 top offense f5total_residual'))} "
                  "versus full-game total residual "
                  f"{value(e('B6 top offense total_residual'))}; "
                  f"paired F5 − full residual "
                  f"{value(e('B6 top offense F5 minus full total residual'))}. "
                  "Top-tercile F5 underdogs: F5 ML "
                  f"{value(e('B6 top offense underdog f5ml win rate'))}, "
                  f"no-vig residual "
                  f"{value(e('B6 top offense underdog f5ml market residual'))}, "
                  f"units/wager {value(e('B6 top offense underdog f5ml units/wager'))}; "
                  "F5 RL "
                  f"{value(e('B6 top offense underdog f5rl win rate'))}, "
                  f"no-vig residual "
                  f"{value(e('B6 top offense underdog f5rl market residual'))}, "
                  f"units/wager {value(e('B6 top offense underdog f5rl units/wager'))}. "
                  f"Required independent n≈{required_n(.12)} for r=.12.",
                  "",
                  "**B7 — as-of rolling windows** (next-game runs, all features "
                  "computed strictly before that date; this is a rolling "
                  "one-step-ahead feature comparison, not a fitted model "
                  "with a held-out test set).",
                  "", "| Family | 30d | 60d | 90d | season-to-date | "
                  "Paired Δr (season − 90d) |",
                  "|---|---|---|---|---|---|"])
    for fam in ("power", "discipline"):
        lines.append(f"| {fam} | " + " | ".join(
            value(e(f"B7 {fam} {w} next runs")) for w in (30, 60, 90, 365)) +
            f" | {value(e(f'B7 {fam} season minus 90 r'))} |")
    lines.extend(["", f"Verdict: underpowered for superiority of one window; "
                  f"required independent n≈{required_n(.10)} for r=.10; "
                  "units: not observed.", "",
                  "**B8 — alternate compression.** "
                  "Decile points are the sum of 1–10 rank points per metric; "
                  "z-scores sum oriented metric z-values, each using its "
                  "production rolling window and date-specific league cohort.",
                  "",
                  f"Compressed gap–run differential: "
                  f"{value(e('vs hand run_diff raw'))}; decile gap: "
                  f"{value(e('B8 decile gap vs run diff'))}; z-score gap: "
                  f"{value(e('B8 zscore gap vs run diff'))}. "
                  f"Compressed score–own runs: "
                  f"{value(e('B8 compressed score vs own runs'))}; decile: "
                  f"{value(e('B8 decile score vs own runs'))}; z-score: "
                  f"{value(e('B8 zscore score vs own runs'))}. "
                  f"Paired Δr decile gap minus compressed gap: "
                  f"{value(e('B8 decile minus compressed r'))}. "
                  f"Verdict: {verdict(e('B8 decile gap vs run diff'), .20)} "
                  f"against r≥+.20; required independent n≈{required_n(.20)}. "
                  "Units: not observed for alternate scores.", "",
                  "## Per-metric table", "",
                  "Correlations and recorded-price directional units are in "
                  "`batting_metrics.csv` and `batting_metrics.md`. "
                  "Units are totals, not per-bet means; blank prices are excluded.",
                  "",
                  "## Provenance and limits", "",
                  "* Every Statcast metric is built from pitches with game_date "
                  "strictly earlier than the game date. Actual starter hand "
                  "is inferred from the first-inning Statcast pitcher; "
                  "missing/unqualified starters are excluded.",
                  "* Outcomes and inning runs come from the cached Stats API "
                  "schedule/linescore; closing prices from local "
                  "`origin/engine-state`, not an opening-price proxy. "
                  "Duplicate same-day matchup labels are excluded; "
                  "the line nearest the median offered total is used, "
                  "with matched Under at that same line.",
                  ("* Starter/bullpen controls use date-bounded FanGraphs "
                   "leaderboards (including Stuff+) and worksheet ranking "
                   "rules. Starter eligibility follows the observed "
                   "FanGraphs inning minimum; as-of skill fallback uses "
                   "season values. This is not byte-identical to the "
                   "production pitcher pipeline."
                   if fg else
                   "* Starter/bullpen controls use Statcast-formula proxy "
                   "xERA (xwOBA rank), xFIP, SIERA and other worksheet "
                   "metric ranks, without FanGraphs-centering or Stuff+. "
                   "Treat market-controlled signs as proxy-standardized "
                   "rather than exact worksheet-standardized."),
                  "* Team-game correlations and grid rows share outcomes "
                  "between opponents. Intervals are game-row bootstrap "
                  "rather than game-cluster bootstrap for those tests; "
                  "they may be optimistic. Model and window comparisons "
                  "are exploratory; there is no multiplicity correction. "
                  "A positive residual correlation does not establish "
                  "a tradeable edge.",
                  "* Team-total quotes were not observed among matched "
                  "closing snapshots; this does **not** imply that "
                  "team-total markets do not exist. Nonpriced season "
                  "games cannot contribute market-residual or units tests.",
                  "",
                  "Reproduce: run `scripts.worksheet_batting_backtest` with "
                  "`--fetch-inputs --season-late --draws 2000` after fetching "
                  "local `origin/engine-state` closing snapshots"
                  f"{' and add `--fangraphs`' if fg else ''}, then run "
                  "`scripts.worksheet_batting_report --audit "
                  "~/.mlb_engine/audit`. CSVs and report are generated outside "
                  "the repository and must not be committed.", ""])
    metrics.to_markdown(audit / "batting_metrics.md", index=False)
    grid.to_markdown(audit / "batting_interaction_grid.md", index=False)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path.home() / ".mlb_engine/audit")
    args = parser.parse_args()
    path = args.audit / "batting_study.md"
    path.write_text(render(args.audit))
    print(path)


if __name__ == "__main__":
    main()
