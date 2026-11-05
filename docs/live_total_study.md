# Live-total study: pregame total + the start, four sports

Regenerate: `python -m scripts.live_total_study pull` then `python -m scripts.live_total_study report --out docs/live_total_study.md` (ESPN, free; cache in `~/.live_edge/total_study`, or `LIVE_STUDY_DIR`).

Research only. ESPN finals and ESPN-kept pregame totals; no live price is used. Totals graded as books grade them (overtime in; NHL shootout = 1 goal). Break-even uses −110 (52.4%).

## NFL

1,207 regular-season finals with a pregame total, 2022-09-08 to 2026-11-02; final − pregame averages +0.77 points.

**1. Slope, pooled (share fit on earlier days only)**

| checkpoint | games | share | b (all) | verdict | b slow starts | b fast starts | slow vs fast | miss frozen / fitted |
|---|---|---|---|---|---|---|---|---|
| Q1 | 1177 | 19.7% | 1.04 ± 0.06 | underpowered (needs ~1848 games) | 1.13 ± 0.13 (n=629) | 0.99 ± 0.11 (n=548) | underpowered (needs ~5140 games) | 11.60 / 11.60 |
| half | 1177 | 51.3% | 1.01 ± 0.03 | holds: start stays in, rest plays to the pregame total | 0.97 ± 0.07 (n=606) | 1.03 ± 0.06 (n=571) | underpowered (needs ~1609 games) | 9.63 / 9.63 |
| Q3 | 1177 | 72.2% | 0.99 ± 0.02 | holds: start stays in, rest plays to the pregame total | 0.93 ± 0.05 (n=602) | 1.04 ± 0.04 (n=575) | same both ways | 7.74 / 7.74 |

**By season** (b ± SE; n)

| checkpoint | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Q1 | 1.03 ± 0.13 (241) | 1.22 ± 0.14 (272) | 0.91 ± 0.13 (271) | 0.97 ± 0.12 (272) | 1.20 ± 0.20 (121) |
| half | 1.05 ± 0.07 (241) | 1.10 ± 0.08 (272) | 0.92 ± 0.07 (271) | 0.96 ± 0.07 (272) | 1.10 ± 0.11 (121) |
| Q3 | 0.98 ± 0.05 (241) | 1.10 ± 0.04 (272) | 0.95 ± 0.05 (271) | 0.96 ± 0.05 (272) | 1.01 ± 0.06 (121) |

**2. Same games off ESPN BET's open vs its close vs the book median** (b ± SE; residual SD). A noisy pregame number pushes b above 1.

| checkpoint | games | ESPN BET open | ESPN BET close | book median |
|---|---|---|---|---|
| Q1 | 643 | 1.01 ± 0.09; 11.78 | 0.99 ± 0.09; 11.70 | 0.99 ± 0.09; 11.70 |
| half | 643 | 0.99 ± 0.05; 9.85 | 0.97 ± 0.05; 9.80 | 0.98 ± 0.05; 9.80 |
| Q3 | 643 | 0.99 ± 0.03; 7.92 | 0.99 ± 0.03; 7.90 | 0.99 ± 0.03; 7.90 |

**3. Miss of the fitted rule by lead at the checkpoint** (final − (T + a + b·gap), mean ± SE points; games)

| checkpoint | lead 0-3 | lead 4-7 | lead 8-14 | lead 15+ |
|---|---|---|---|---|
| Q1 | -0.27 ± 0.48 (520) | +0.71 ± 0.54 (492) | -1.96 ± 0.97 (149) | – (16) |
| half | +0.56 ± 0.53 (320) | +0.47 ± 0.52 (364) | -0.10 ± 0.53 (323) | -1.86 ± 0.69 (170) |
| Q3 | +1.39 ± 0.49 (275) | +0.39 ± 0.42 (305) | -0.28 ± 0.42 (321) | -1.48 ± 0.47 (276) |

**4. What a gap is worth.** Break-even = the gap between the fair number and the live line a −110 ticket needs (Normal miss). Typical continuation = |b − 1| × SD of the surprise. Ceiling: walk-forward (each season's a, b and lead bands fit on the seasons before it) against a hypothetical live line of exactly T + surprise, rounded to the half point, betting only where the rule is at least the break-even gap off it; hit rate with pushes out. No live price is used. NHL lines move by juice, not half goals, so its ceiling is rough.

| checkpoint | miss SD | break-even gap | typical continuation | its win rate | ceiling: slope (edge ≥ break-even) | ceiling: slope + lead band (edge ≥ break-even) |
|---|---|---|---|---|---|---|
| Q1 | 11.61 | 0.69 | 0.25 | 50.8% | 51.9% ± 3.2% (n=239) | 53.9% ± 2.3% (n=473) |
| half | 9.63 | 0.58 | 0.05 | 50.2% | 51.8% ± 2.3% (n=485) | 55.2% ± 2.0% (n=638) |
| Q3 | 7.75 | 0.46 | 0.08 | 50.4% | 50.7% ± 2.5% (n=414) | 56.6% ± 1.9% (n=655) |

## NBA

5,045 regular-season finals with a pregame total, 2022-10-18 to 2026-11-04; final − pregame averages +0.55 points.

**1. Slope, pooled (share fit on earlier days only)**

| checkpoint | games | share | b (all) | verdict | b slow starts | b fast starts | slow vs fast | miss frozen / fitted |
|---|---|---|---|---|---|---|---|---|
| Q1 | 5009 | 25.2% | 1.03 ± 0.03 | holds: start stays in, rest plays to the pregame total | 1.01 ± 0.05 (n=2534) | 1.04 ± 0.05 (n=2475) | same both ways | 15.94 / 15.94 |
| half | 5009 | 50.3% | 0.98 ± 0.02 | holds: start stays in, rest plays to the pregame total | 0.98 ± 0.03 (n=2536) | 0.99 ± 0.03 (n=2473) | same both ways | 13.65 / 13.65 |
| Q3 | 5009 | 75.6% | 1.01 ± 0.01 | holds: start stays in, rest plays to the pregame total | 1.00 ± 0.02 (n=2577) | 1.02 ± 0.02 (n=2432) | same both ways | 10.30 / 10.30 |

**By season** (b ± SE; n)

| checkpoint | 2022-23 | 2023-24 | 2024-25 | 2025-26 | 2026-27 |
|---|---|---|---|---|---|
| Q1 | 1.04 ± 0.06 (1195) | 1.08 ± 0.05 (1231) | 0.92 ± 0.06 (1231) | 1.09 ± 0.05 (1231) | 0.92 ± 0.17 (121) |
| half | 1.01 ± 0.04 (1195) | 0.98 ± 0.03 (1231) | 0.96 ± 0.03 (1231) | 1.00 ± 0.03 (1231) | 0.92 ± 0.13 (121) |
| Q3 | 1.01 ± 0.02 (1195) | 0.99 ± 0.02 (1231) | 1.02 ± 0.02 (1231) | 1.02 ± 0.02 (1231) | 1.07 ± 0.06 (121) |

**2. Same games off ESPN BET's open vs its close vs the book median** (b ± SE; residual SD). A noisy pregame number pushes b above 1.

| checkpoint | games | ESPN BET open | ESPN BET close | book median |
|---|---|---|---|---|
| Q1 | 2754 | 1.05 ± 0.04; 16.16 | 0.99 ± 0.04; 15.84 | 0.99 ± 0.04; 15.83 |
| half | 2754 | 1.01 ± 0.02; 13.68 | 0.97 ± 0.02; 13.50 | 0.97 ± 0.02; 13.50 |
| Q3 | 2754 | 1.02 ± 0.01; 10.36 | 1.00 ± 0.01; 10.31 | 1.00 ± 0.01; 10.31 |

**3. Miss of the fitted rule by lead at the checkpoint** (final − (T + a + b·gap), mean ± SE points; games)

| checkpoint | lead 0-5 | lead 6-10 | lead 11-15 | lead 16-20 | lead 21+ |
|---|---|---|---|---|---|
| Q1 | -0.15 ± 0.33 (2391) | +0.07 ± 0.41 (1494) | +0.52 ± 0.57 (768) | -0.73 ± 0.92 (272) | +0.55 ± 1.37 (84) |
| half | -0.09 ± 0.33 (1790) | +0.17 ± 0.38 (1356) | -0.40 ± 0.43 (924) | +0.52 ± 0.60 (537) | +0.07 ± 0.65 (402) |
| Q3 | +0.50 ± 0.29 (1546) | +0.14 ± 0.31 (1176) | -0.10 ± 0.33 (895) | -0.33 ± 0.36 (619) | -0.83 ± 0.32 (773) |

**4. What a gap is worth.** Break-even = the gap between the fair number and the live line a −110 ticket needs (Normal miss). Typical continuation = |b − 1| × SD of the surprise. Ceiling: walk-forward (each season's a, b and lead bands fit on the seasons before it) against a hypothetical live line of exactly T + surprise, rounded to the half point, betting only where the rule is at least the break-even gap off it; hit rate with pushes out. No live price is used. NHL lines move by juice, not half goals, so its ceiling is rough.

| checkpoint | miss SD | break-even gap | typical continuation | its win rate | ceiling: slope (edge ≥ break-even) | ceiling: slope + lead band (edge ≥ break-even) |
|---|---|---|---|---|---|---|
| Q1 | 15.94 | 0.95 | 0.22 | 50.6% | 51.9% ± 1.2% (n=1752) | 50.0% ± 1.0% (n=2293) |
| half | 13.65 | 0.82 | 0.20 | 50.6% | 51.5% ± 0.8% (n=3662) | 50.8% ± 0.9% (n=3089) |
| Q3 | 10.30 | 0.62 | 0.14 | 50.6% | 50.0% ± 0.8% (n=3523) | 50.0% ± 1.0% (n=2415) |

## CBB

17,065 regular-season finals with a pregame total, 2023-11-06 to 2026-11-04; final − pregame averages +0.69 points.

**1. Slope, pooled (share fit on earlier days only)**

| checkpoint | games | share | b (all) | verdict | b slow starts | b fast starts | slow vs fast | miss frozen / fitted |
|---|---|---|---|---|---|---|---|---|
| half | 16951 | 47.1% | 1.12 ± 0.01 | carry-over: +12% of the gap added | 1.09 ± 0.02 (n=8757) | 1.15 ± 0.02 (n=8194) | same both ways | 12.32 / 12.26 |

**By season** (b ± SE; n)

| checkpoint | 2023-24 | 2024-25 | 2025-26 | 2026-27 |
|---|---|---|---|---|
| half | 1.14 ± 0.02 (5520) | 1.10 ± 0.02 (5639) | 1.13 ± 0.02 (5644) | 1.19 ± 0.09 (148) |

**2. Same games off ESPN BET's open vs its close vs the book median** (b ± SE; residual SD). A noisy pregame number pushes b above 1.

| checkpoint | games | ESPN BET open | ESPN BET close | book median |
|---|---|---|---|---|
| half | 11830 | 1.13 ± 0.01; 12.23 | 1.12 ± 0.01; 12.20 | 1.12 ± 0.01; 12.20 |

**3. Miss of the fitted rule by lead at the checkpoint** (final − (T + a + b·gap), mean ± SE points; games)

| checkpoint | lead 0-5 | lead 6-10 | lead 11-15 | lead 16+ |
|---|---|---|---|---|
| half | +0.51 ± 0.15 (6772) | -0.05 ± 0.18 (4698) | -0.45 ± 0.22 (2930) | -0.75 ± 0.22 (2551) |

**4. What a gap is worth.** Break-even = the gap between the fair number and the live line a −110 ticket needs (Normal miss). Typical continuation = |b − 1| × SD of the surprise. Ceiling: walk-forward (each season's a, b and lead bands fit on the seasons before it) against a hypothetical live line of exactly T + surprise, rounded to the half point, betting only where the rule is at least the break-even gap off it; hit rate with pushes out. No live price is used. NHL lines move by juice, not half goals, so its ceiling is rough.

| checkpoint | miss SD | break-even gap | typical continuation | its win rate | ceiling: slope (edge ≥ break-even) | ceiling: slope + lead band (edge ≥ break-even) |
|---|---|---|---|---|---|---|
| half | 12.26 | 0.73 | 1.23 | 54.0% | 52.5% ± 0.6% (n=7868) | 52.9% ± 0.6% (n=7900) |

## NHL

5,461 regular-season finals with a pregame total, 2022-10-07 to 2026-11-04; final − pregame averages +0.13 goals.

**1. Slope, pooled (share fit on earlier days only)**

| checkpoint | games | share | b (all) | verdict | b slow starts | b fast starts | slow vs fast | miss frozen / fitted |
|---|---|---|---|---|---|---|---|---|
| P1 | 5423 | 29.0% | 0.96 ± 0.02 | fade: -4% of the gap given back | 0.99 ± 0.05 (n=2511) | 0.94 ± 0.03 (n=2912) | same both ways | 1.95 / 1.94 |
| P2 | 5423 | 63.3% | 0.98 ± 0.01 | holds: start stays in, rest plays to the pregame total | 0.99 ± 0.02 (n=3031) | 0.98 ± 0.02 (n=2392) | same both ways | 1.39 / 1.39 |

**By season** (b ± SE; n)

| checkpoint | 2022-23 | 2023-24 | 2024-25 | 2025-26 | 2026-27 |
|---|---|---|---|---|---|
| P1 | 0.93 ± 0.04 (1274) | 0.91 ± 0.04 (1314) | 1.06 ± 0.04 (1319) | 0.94 ± 0.04 (1310) | 1.03 ± 0.10 (206) |
| P2 | 0.97 ± 0.02 (1274) | 0.99 ± 0.02 (1314) | 0.99 ± 0.02 (1319) | 0.98 ± 0.02 (1310) | 1.00 ± 0.05 (206) |

**2. Same games off ESPN BET's open vs its close vs the book median** (b ± SE; residual SD). A noisy pregame number pushes b above 1.

| checkpoint | games | ESPN BET open | ESPN BET close | book median |
|---|---|---|---|---|
| P1 | 3010 | 0.98 ± 0.03; 1.96 | 0.98 ± 0.03; 1.96 | 0.98 ± 0.03; 1.96 |
| P2 | 3010 | 0.99 ± 0.01; 1.41 | 0.99 ± 0.01; 1.41 | 0.99 ± 0.01; 1.40 |

**3. Miss of the fitted rule by lead at the checkpoint** (final − (T + a + b·gap), mean ± SE goals; games)

| checkpoint | lead 0 | lead 1 | lead 2 | lead 3+ |
|---|---|---|---|---|
| P1 | +0.05 ± 0.04 (1847) | +0.02 ± 0.04 (2324) | -0.03 ± 0.06 (948) | -0.41 ± 0.11 (304) |
| P2 | +0.06 ± 0.04 (1180) | +0.09 ± 0.03 (1901) | +0.05 ± 0.04 (1320) | -0.30 ± 0.04 (1022) |

**4. What a gap is worth.** Break-even = the gap between the fair number and the live line a −110 ticket needs (Normal miss). Typical continuation = |b − 1| × SD of the surprise. Ceiling: walk-forward (each season's a, b and lead bands fit on the seasons before it) against a hypothetical live line of exactly T + surprise, rounded to the half point, betting only where the rule is at least the break-even gap off it; hit rate with pushes out. No live price is used. NHL lines move by juice, not half goals, so its ceiling is rough.

| checkpoint | miss SD | break-even gap | typical continuation | its win rate | ceiling: slope (edge ≥ break-even) | ceiling: slope + lead band (edge ≥ break-even) |
|---|---|---|---|---|---|---|
| P1 | 1.94 | 0.12 | 0.05 | 51.1% | 47.9% ± 1.0% (n=2455) | 49.1% ± 1.0% (n=2501) |
| P2 | 1.39 | 0.08 | 0.03 | 50.9% | 59.3% ± 1.3% (n=1471) | 54.8% ± 0.9% (n=2779) |
