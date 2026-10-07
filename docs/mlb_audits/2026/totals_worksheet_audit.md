# MLB 2026: totals sheet and daily worksheet audit

Through 2026-10-06, from `origin/engine-state`. Regenerate with `python -m scripts.season_audit --season 2026 --engine-state`; the graded ledgers beside this file are the rows it counted.

## Totals sheet

23 days, 243 graded games on bands `centred-2026.09`; 1 with no line on record, never graded. Break-even at -110 is 52.4%; a slice is an edge only when its whole 95% range clears it on 30+ decided games.

| Slice | Record | 95% range | Verdict |
|---|---|---|---|
| SUM sign as the call | 118-102-11 (54%) | 47-60% | unproven |
| Always the Over, same games | 133-99-11 (57%) | 51-64% | unproven |
| SUM says Over | 81-55-6 (60%) | 51-67% | unproven |
| SUM says Under | 37-47-5 (44%) | 34-55% | unproven |
| Top-3 Overs each day | 26-21-2 (55%) | 41-69% | unproven |
| Bottom-3 Unders each day | 21-31-2 (40%) | 28-54% | unproven |
| \|SUM\| 1..4 | 44-36-6 (55%) | 44-65% | unproven |
| \|SUM\| 5..9 | 48-33-3 (59%) | 48-69% | unproven |
| \|SUM\| 10..14 | 17-23-1 (42%) | 29-58% | unproven |
| \|SUM\| >= 15 | 9-10-1 (47%) | 27-68% | unproven (n<30) |
| Watch Over (SUM +5..14, total <= 8.5) | 35-24-1 (59%) | 47-71% | unproven |
| Watch Under (SUM -14..-5, total <= 7.5) | 16-19-1 (46%) | 30-62% | unproven |
| Engine's side of the sheet's line | 88-84-7 (51%) | 44-59% | unproven |
| Engine's side vs the market | 44-41 (52%) | 41-62% | unproven |
| Sheet call, engine agrees | 59-43-4 (58%) | 48-67% | unproven |
| Sheet call, engine disagrees | 34-25-3 (58%) | 45-69% | unproven |
| Engine call, engine disagrees | 25-34-3 (42%) | 31-55% | unproven |

Over rate by SUM (the slate's own base rate inside each band):

| SUM | Overs-Unders-Pushes | Over rate |
|---|---|---|
| <= -5 | 28-25-2 (53%) | 40-66% |
| -4..-1 | 19-12-3 (61%) | 44-76% |
| 0 | 5-7 (42%) | 19-68% |
| 1..9 | 64-34-4 (65%) | 55-74% |
| 10..19 | 16-19-2 (46%) | 30-62% |
| >= 20 | 1-2 (33%) | 6-79% |

By month:

| Month | Games | SUM sign | Over rate |
|---|---|---|---|
| 2026-09 | 234 | 113-99-10 (53%) | 130-94-10 (58%) |
| 2026-10 | 9 | 5-3-1 (62%) | 3-5-1 (38%) |

## Daily worksheet

Weights `w1-1.5-1-dec-spz` (2026-09-25 to 2026-10-06): 47 graded games with both starters scored; 110 graded rows on older weights, not counted. The call is the side with the higher wTOTAL; priced games are judged against the no-vig probability the market gave that side when the worksheet was written.

| Gap band | Fav W-L | Win % (95%) | Priced fav W-L | Priced win % (95%) | Market | Flat 1u ML | Fav run line | Verdict |
|---|---|---|---|---|---|---|---|---|
| 0-9 | 12-6 | 66.7% (44-84%) | 7-2 | 77.8% (45-94%) | 53.6% | +3.79u on 9 | 4-5 | unproven (n<30) |
| 10-19 | 10-5 | 66.7% (42-85%) | 6-5 | 54.5% (28-79%) | 56.8% | -0.74u on 11 | 5-6 | unproven (n<30) |
| 20-34 | 6-2 | 75.0% (41-93%) | 3-1 | 75.0% (30-95%) | 54.7% | +1.60u on 4 | 2-2 | unproven (n<30) |
| 35+ | 4-2 | 66.7% (30-90%) | 1-0 | 100.0% (21-100%) | 67.9% | +0.44u on 1 | 1-0 | unproven (n<30) |
| **All** | 32-15 | 68.1% (54-80%) | 17-8 | 68.0% (48-83%) | 55.7% | +5.09u on 25 | 12-13 | unproven (n<30) |

Priced favourites won 17 against 13.9 expected at the market's prices (z = +1.25). Confirming the current gap at 95% with 80% power takes about 129 priced games.

By month:

| Month | Fav W-L | Priced W-L vs market |
|---|---|---|
| 2026-09 | 27-13 | 12-6 vs 54.7% |
| 2026-10 | 5-2 | 5-2 vs 58.4% |
