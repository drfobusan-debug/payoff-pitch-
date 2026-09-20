# live-edge: in-play edge detector (NFL / CFB)

Watch-only. Rolls a pregame prior forward through the live score and clock,
compares the rolled win / cover / over probabilities with devigged in-play
prices from The Odds API, and writes flags to a ledger that is graded at the
final. Nothing is staked; the ledger exists to find out whether the rule has an
edge before anything is.

```
live-edge --sport nfl tick                 # one pass, prints new flags
live-edge --sport cfb run --interval 60    # poll all day; stops when nothing is live
live-edge report                           # graded ledger by market and quarter
```

Needs `THE_ODDS_API_KEY` (or `ODDS_API_KEY`). Each tick that has a live or
imminent game costs one board call (3 markets, ~15-20 credits for the NFL).
State lives in `LIVE_EDGE_DIR` (default `~/.live_edge`):

| file | contents |
| --- | --- |
| `priors.json` | pregame consensus (home margin, total) per ESPN event; captured from the board before kickoff, or from ESPN's pickcenter closing line for games already under way |
| `streaks.json` | consecutive qualifying ticks per (event, market, side, line, book) |
| `tape/{sport}_{day}.jsonl` | one line per live game per tick: score, clock, possession, rolled distribution, ESPN's own win probability, every candidate |
| `flags.csv` | every flag raised, with `result` / `pnl` filled in once the game is final |

## Model

Let `f` be the fraction of regulation remaining, `μ`/`T` the pregame margin and
total, `m`/`s` the current margin and points. The remaining margin is

```
R ~ N( a_q·f·μ + b_q·f·(m − (1−f)·μ) + c_q·EP ,  σ(f)² )
```

with per-quarter `(a, b, c)` and the residual curve `σ(f)` fitted on 2024 FBS
play-by-play (43k game-minute snapshots) and validated out of sample on 2025
and 2026 wk1-3 (Brier 0.137 vs 0.139 for a naive roll). The total uses
`T_rem = 1.059·f·T + 0.088·f·pace`. The NFL reuses the FBS coefficients with
the residual scaled by the ratio of pregame dispersions (13.2 / 16.0) until an
NFL fit exists. Coefficients are in `live_edge/roller.py`.

Two fitted facts drive the behaviour:

* `b_1 ≈ +0.07`: first-quarter points say almost nothing about team strength,
  so an underdog up 10 early is rolled as *prior + score*, not as the new
  favourite. (In the backtest, pregame dogs of 7+ that led by 10+ after Q1 won
  37%, versus 52% for a score-only roll.)
* `b_4 ≈ −0.54`: fourth-quarter leaders bleed margin, so late covers by big
  leaders are priced down.

## Flag rule (defaults in `Thresholds`)

A flag needs, on two consecutive ticks:

* model minus the **median devigged price across ≥2 books** ≥ 6 pts (ML) or 5 pts
  (spread / total), with positive EV at the best available price;
* quotes ≤ 90 s old, implied price between 12% and 88%;
* ≥ 15% of regulation left.

Measuring against the median rather than the best price stops a single slow
book from raising a flag on its own; the best price among those books is what
gets recorded. Flags are deduplicated per (event, market, side, line, book).

## What the backtest does and does not show

The historical test used CFBD play-by-play and closing lines only; there were
no historical in-play prices, so the "market" there was a naive roll of the
closing line. Simulated ROI against that proxy is an upper bound on nothing in
particular. The forward tape is the evidence: keep `run` going through the
weekends, grade with `report`, and do not treat any market as proven until it
has a few hundred graded flags with the by-quarter breakdown holding up.
