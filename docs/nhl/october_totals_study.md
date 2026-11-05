# Is the NHL totals signal weaker in October?

Research only. Nothing here changes pricing; `NHLE_TOTAL_ANCHOR` stays 0.60.

## Preregistered (fixed 2026-11-04, before results)

Data: `nhl-engine closetest` obs for the 2023-24 and 2024-25 regular seasons, archived historical closes,
official starters, unanchored sim. Split: October = game date in October; rest = Nov-Apr. Totals = Over at
the main line, pushes dropped.

Measures per split: slope b (result - p_close on p_sim - p_close, OLS without intercept) +- SE;
mean(p_sim - p_close) +- SE; Brier sim vs close.

Decision:

1. "October is different" only if b_oct <= 1 SE above zero (or negative) AND
   z = (b_rest - b_oct) / sqrt(se_oct^2 + se_rest^2) > 1 -> recommend a 0.90 totals anchor for October games,
   0.60 the rest of the season.
2. If b_oct > 1 SE -> keep 0.60 year-round.
3. Otherwise -> not answerable at this n; keep 0.60.

Level check: October mean(p_sim - p_close) < 0 by > 2 SE in both seasons -> "recurring October level gap".

## Result (2026-11-05)

`python scripts/nhl/october_totals_split.py ct_2324.csv ct_2425.csv`

| split | n | slope +- SE | sim - close (pts) | Over hit | Brier sim / close | goals / sim / line |
|---|---|---|---|---|---|---|
| October | 234 | +0.02 +- 0.62 | -1.92 (se 0.32) | 49.6% | 0.2514 / 0.2488 | 6.35 / 6.28 / 6.26 |
| Nov-Apr | 1,973 | +0.46 +- 0.23 | -2.21 (se 0.10) | 48.0% | 0.2502 / 0.2499 | 6.14 / 6.15 / 6.10 |
| all | 2,207 | +0.40 +- 0.22 | -2.18 (se 0.09) | 48.2% | 0.2503 / 0.2498 | 6.16 / 6.17 / 6.12 |
| 2023-24 Oct | 101 | -0.46 +- 1.03 | -1.30 (se 0.46) | 43.6% | 0.2503 / 0.2459 | 6.12 / 6.40 / 6.36 |
| 2023-24 rest | 1,025 | +0.71 +- 0.32 | -3.16 (se 0.12) | 48.6% | 0.2484 / 0.2494 | 6.22 / 6.19 / 6.23 |
| 2024-25 Oct | 133 | +0.30 +- 0.79 | -2.40 (se 0.44) | 54.1% | 0.2523 / 0.2510 | 6.51 / 6.19 / 6.19 |
| 2024-25 rest | 948 | +0.17 +- 0.34 | -1.19 (se 0.15) | 47.5% | 0.2521 / 0.2506 | 6.04 / 6.11 / 5.96 |

z (rest - oct) = +0.65 -> rule 3: **not answerable at this n; keep 0.60 year-round.**

The level check passes (October is 2.8 and 5.5 SE under the close), but the rest of the season is just as far
under, so it shows a year-round lean, not an October one. The check should have been October minus rest.
Over the full seasons that lean was right (Overs hit ~48% vs ~50% implied), and the two Octobers went
opposite ways (43.6% vs 54.1%).

Context: `scripts/nhl/live_vs_rebuild_study.py` on the 2026 ledger (240 games, 09-29..11-03) put the live
Over -3.9 pts (se 0.3) and the rebuild -2.9 (se 0.3) under the consensus. Live matched the rebuild from 10-09
on, and the live card's named goalies moved the rebuild only +0.1 pts, so the 2026 gap is the model's own
opinion, not a live-only input.
