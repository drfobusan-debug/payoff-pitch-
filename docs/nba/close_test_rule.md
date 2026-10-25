# NBA historical close test — preregistered decision rule

Written before the pull, so the outcome cannot be argued with afterwards (the NHL
test of 2026-10 is the template: `docs/nhl` close test, `nhl-engine closetest`).

## What is tested

The team-rating model (`nba_engine.models.ratings`, fitted by `nba-engine fit-ratings`)
against the **closing** spread and total, on games the fit never saw.

* Archive: Odds API historical event odds at the `close` anchor (T−5), game markets only
  (`h2h,spreads,totals`), seasons 2023-24 and 2024-25; ESPN finals + boxes via
  `nba-engine boxes` (free).
* Fit: `--train 2023 --holdout 2024`. The first season is burn-in and is never graded.
* Grade: per market, `slope` of `actual − close` on `model − close`, bootstrap clustered by
  slate date (`rating_fit.vs_close`), plus MAE of model and close on the same games.

## Decision rule (per market: spread, total)

| Holdout result | Action |
|---|---|
| slope's 95% interval includes 0 | rating is a **prior for the sim only**; market weight `w = 0` for that market; the market stays `no_model` → market-only, graded-only |
| interval excludes 0, slope ≤ 0.5 | anchor weight `w = slope` (model carries `slope` of its own disagreement), **probation-gated**: live buys at that weight grade as their own cut; §11 bar (CLV lower bound > 0, Brier ≤ market, 100 graded buys) before "Buy" |
| interval excludes 0, slope > 0.5 | same as above with `w` capped at 0.5 until a second holdout season repeats it |
| model MAE worse than the close by > 0.5 pt on the holdout | no weight regardless of slope |

Moneylines are not graded here: the ML is read off the fitted margin, so it inherits the
spread's verdict. 1H and props wait for their own fits (`fit-minutes`, `fit-sim`).

## What cannot change this

* No parameter is re-tuned after seeing the holdout grade. If the holdout fails, the fix is
  a new feature, then a new holdout (2025-26), not a re-fit on 2024.
* The pull size is approved before it runs; the rule is not edited between approval and
  report.

## Size

~1,230 regular-season games a season. Game markets only at one anchor cost
`10 credits × 3 markets = 30` per event: about **37,000 credits a season, 74,000 for two**,
plus one listing call a day. `nba-engine coverage` is free and says what is already held.
