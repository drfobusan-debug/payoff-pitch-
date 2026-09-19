# NHL Betting Engine — Master Plan

Fourth engine in `payoff-pitch-`, alongside `mlb_engine`, `cfb_engine`, `nfl_engine`.
Markets in scope: moneyline, puck line (NHL's "run line", ±1.5), game totals,
1st/2nd/3rd period markets (period ML incl. tie, period puck line ±0.5/±1.5, period
totals), and player props (shots on goal, points, goals, assists, saves, blocked
shots, PP points).

Two reading notes on the request: "run lines" in hockey are **puck lines** (±1.5,
with ±2.5 alternates); "runtiness" I have read as the **period markets**
(scoring/margin per 20-minute period). And I have taken "NIL engine" to mean the
**NFL engine** — the repo has `nfl_engine`, `mlb_engine`, `cfb_engine`, `lineshop`
and `cardscout`; there is no NIL module. Say so if you meant something else.

---

## 1. What 383 PRs taught us (the rules the NHL engine inherits on day one)

These are the lessons that cost the most in MLB/CFB/NFL and are now structural.
Each one maps to a design decision below.

| # | Lesson (where it was learned) | NHL consequence |
|---|---|---|
| L1 | **The market is the better forecaster in every market we bet** (MLB Brier .2347 vs .2408 over 27 slates; CFB efficiency layer adds *nothing* after the closing spread, partial r −0.001). | The NHL model's job is to *earn departures* from the market, not to replace it. Ship with `NHLE_MARKET_BLEND` on the means (CFB uses 0.35) and a `max_edge` cap that reads big disagreements as model error. |
| L2 | **Judge on CLV, not ROI.** Nine slates of ROI still spanned [−16%, +6%]; CLV decides in dozens of bets. | `close` snapshots per game (puck drop is staggered 7–10:30pm ET → multiple passes, pregame-only, like CFB PR #351/#377). CLV is the primary scoreboard from slate one. |
| L3 | **Rank on edge, not EV.** `EV = dec_odds × edge` made the Strong tier a plus-money-dog bucket that inverted (39.9% vs Moderate 46.9%). | Copy `cfb_engine/market/tiers.py` verbatim: `min_ev=0`, `min_edge`, `strong_edge_gap`, `max_edge`. |
| L4 | **Plus-money buys leak** (28.5% for −15.5% ROI, n=933) → per-market price ceilings (`+109` on run lines). | Puck-line favourites at −1.5 are typically +150 to +220. Ship a per-market `MAX_BUY_ODDS` on `game_pl`; expect the −1.5 side to be mostly refused and the +1.5 dog to be the bettable side, until the ledger says otherwise. |
| L5 | **A 50-bet cell is not evidence** → `probation.py` (n≥100, >1 s.e., both halves of the window agree) grades every market, live screen and *candidate* screen. | Port `cfb_engine/audit/probation.py` unchanged. NHL has ~1,300 games/season and ~6–8 games/night — volume arrives faster than CFB, so the 100 bar is a few weeks. |
| L6 | **Screens that delete winners cost as much as markets that lose.** Every gate stamps `pass_gate`; refusals are graded as the counterfactual. | Every NHL veto (goalie-unconfirmed, price band, drift, sharp money, B2B) is a named `pass_gate`, priced and graded anyway. |
| L7 | **Results repeat far less than skill/process metrics** (pitcher xwOBA 0.31 vs velocity 0.95; bullpen line used at 5× its worth). Shrink to measured reliability, not to a flat prior. | Hockey's version: **PDO/shooting% and goalie Sv% are noise-heavy; shot-quality (xG) and shot-volume rates are the signal.** Fit per-metric empirical-Bayes shrinkage on 2015–2025 before any metric is priced (Phase 1 study). Goalie GSAx must be shrunk hard. |
| L8 | **Announced/confirmed lineup is the highest-value pre-game fact** (MLB lineup lock, probable pitcher; CFB QB absence worth −2.2 pts *before* the line moves and 0 after). | The **starting goalie** is the NHL analogue. Never buy a side/total without a confirmed or highly probable starter; log the time we learned it against the line at that moment (`availability.py` pattern) so the "did we beat the move" question is answerable. |
| L9 | **Props: no historical prop prices exist anywhere**, so a prop model can only be judged against the model's own pseudo-lines (friendliest possible test) until a forward archive exists. NFL ships props `research_only` and archives every quote since preseason. | Start the **NHL prop capture archive** in Phase 0, before a line of model code, and stamp every prop row `research_only` until the archive holds graded closes. |
| L10 | **Batter overs lost −169 units; every prop market except doubles lost.** Low-probability, plus-money props (HR) need a *price band*, not a probability floor. | Anytime-goal-scorer and "2+ points" are the HR of hockey. Ship them `NO_BUY` or in a tight price band from the start; SOG (high-frequency count, like receptions — the only NFL prop that beat its base rate) is the prop to model first. |
| L11 | **One buy per prop, one side per prop, no correlated legs** (a QB's passing yards and his WR's receiving yards are one opinion sold twice). | One row per player; one per team-direction; a team total over and its goalie's saves under are one opinion — dedupe. |
| L12 | **Sharp money as a moneyline gate** (MLB EV AUC on ML 0.33 — inverted; handle−tickets AUC 0.80). | Port `vsin_splits.py` + `mlsharp.py` (VSiN posts NHL splits). Gate ML only; weakest form (divergence ≥ 0) until NHL rows exist. |
| L13 | **Price the ML on the SD the market uses**, not the sim's (CFB PR #382: flat 16 handed every dog value, 7–52). | Hockey margins are discrete and Poisson-ish, not normal — but the same trap exists: an over-dispersed sim gives every dog value. Calibrate goal-distribution dispersion (Dixon-Coles ρ, OT rate) to the *market's* ML↔puck-line implied relationship on live boards before trusting the sim's dog probabilities. |
| L14 | **Ops that survive disposable machines**: orphan `engine-state` branch, write-once pregame predictions, first-seen board snapshot, pregame-only closes, idempotent installers, `/etc/engine.env`. | Reuse `mlb_engine.state` plumbing under an `nhl/` prefix exactly as `cfb_engine/state.py` does. |
| L15 | **Measured-null features ship OFF with the study in the docstring** (CFB wind, returning production at 51.96% ATS, efficiency blend 0). | Every situational term (B2B, travel, rest, altitude) ships at the fitted coefficient *only* if it survives the closing line; otherwise 0 with the number written down. |
| L16 | **Open-side board only** — "refuse a buy on a one-way quote", devig the same book's two sides, never invent a partner. | Period and prop markets are thin and often one-sided at a given book; enforce pairing on ingest. |

---

## 2. Hockey-specific modelling facts that shape the design

- **Scoring is low and discrete** (~6.0–6.3 goals/game in recent seasons). Team goals ≈ Poisson with slight under-dispersion at 5v5 and correlation between the two sides. The standard spine is a **bivariate Poisson / Dixon-Coles** goal model, *not* a normal-margin Monte Carlo like CFB. Use the per-period simulator (below) so full-game, puck-line, totals and period markets come from **one** simulation and stay internally consistent.
- **Overtime and shootout** define the markets: 2-way ML pays on OT/SO; puck line −1.5 effectively needs a regulation win by 2 or a late empty-netter; totals include OT/SO goals (max +1). ~23% of games go to OT; ~half of those to SO. The sim must model regulation tie → OT (3v3, ~55–60% ends in OT) → SO (coin-flip-ish, slight home/goalie edge) explicitly.
- **The 3rd period is not the other two.** Empty-net goals (~0.3/game, almost all in the last 2 minutes) inflate 3rd-period totals and 2-goal margins; trailing teams push (score effects raise shot rates for the trailer). Period markets therefore need a **score-state-aware** period model, not "1/3 of the game total". Period 1 has the lowest scoring and the most ties (a 3-way period ML with a tie leg near +200 is standard).
- **Goalies are the single largest lineup variable** and their skill signal is weak: season Sv% has year-to-year r ≈ 0.1–0.3; GSAx/60 regressed to ~half. Starters are confirmed the morning of or at warm-ups (~5pm ET). A backup vs starter swing on the ML is often 3–6 probability points — larger than any edge we will ever claim.
- **Special teams** (PP ~ 20–25% conversion, ~3 PPO per team-game) supply ~20% of goals; penalties drawn/taken rates are somewhat repeatable. 5v5 xG rates (score- and venue-adjusted) repeat best; PDO repeats least.
- **Schedule effects** are real and *mostly priced*: back-to-back (second-night team −1 to −2 win-prob pts), 3-in-4, long road trips, time zones, Denver altitude. Measure against the close before pricing (L15).
- **Player props**: SOG is the liquid, count-based, usage-driven market (TOI × individual shot rate × opponent shot suppression × score effects); points/goals/assists are rare-event Poissons dominated by linemates and PP unit; saves = opponent shot volume × (1−Sv%), conditional on the starter. TOI is the "usage" of hockey (NFL L: usage is projectable, edge is not).

---

## 3. Data sources (free-first; verify each before Phase 1 — none is a stated fact until it returns data)

| Need | Primary (free) | Notes / fallback |
|---|---|---|
| Schedule, boxscores, play-by-play, shifts, final scores per period, OT/SO flag | **NHL API** (`api-web.nhle.com/v1/…`: schedule, gamecenter landing/boxscore/play-by-play, shift charts via `api.nhle.com/stats/rest`) | Public, no key. Landing endpoint carries starting-goalie flags near game time. Back to 2010s. |
| Shot-level xG, team/skater/goalie rates 5v5/PP/PK/4v5/3v3, score-adjusted | **MoneyPuck** public CSVs (shots 2007–present; teams/skaters/goalies per season) | Bulk download, cache to `~/.nhl_engine/cache`. Fallback: Natural Stat Trick (scrape, rate-limited; treat like TeamRankings — cache, never hammer). **Arena tracking bias**: off-ice tracking is rink-biased (shot distances, hits, giveaways/takeaways). The engine uses MoneyPuck *xG*, not raw coordinates, and MoneyPuck is believed to rink-adjust its model (verify in Phase 0). Phase 1 `scripts/nhl/rink_bias_study.py`: per-arena home-vs-away shot-distance and shot-count residuals vs league; if material, a rolling multi-season arena correction is applied to coordinates *before* any RAPM/isolated-impact fit. **Hits, giveaways, takeaways and blocked-shot counts are barred as model inputs** (card context only). |
| Confirmed starting goalies / projected lines | **RotoWire NHL lineups page** (`rotowire.com/hockey/nhl-lineups.php` — goalie status Confirmed/Expected/Unconfirmed + skater lines; same page-shape as `mlb_engine/data/rotowire.py`) | Optional second vote: DailyFaceoff (off the critical path). **Final lineup: NHL API gamecenter roster once warmups end (~20–30 min before drop)**, read by the per-game pre-drop pass; boxscore starter is grading truth. Manual CSV drop-in (`~/.nhl_engine/goalies_<date>.csv`) overrides both. |
| Lines, pairs, PP units (actual, for grading & matchup section) | NHL API shift charts (`api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=…`) + play-by-play, clustered by TOI-together into L1–L4 / D1–D3 / PP1–PP2 | No scraping needed; RotoWire/DailyFaceoff projected lines override when they differ (coach change). |
| Injuries | **RotoWire hockey injury table** (`rotowire.com/hockey/tables/injury-report.php`) for designation/player/team + RotoWire RSS (`rss/news.php?sport=NHL`) for posting time — the exact pair `nfl_engine/data/injuries.py` and `cfb_engine/data/injuries.py` use | Cross-check/confirm: NHL API roster + boxscore scratches (official, late). Every capture logs `(source, posted_at, seen_at, status)` to the availability log so the latency study runs from day one. Role tags (1C/1D/PP1/G1) come from our lines clustering, not the feed. |
| Season aggregates by strength (5v5/PP/PK) | NHL stats API (`api.nhle.com/stats/rest/en/` team/skater/goalie tables) | Second xG model for cross-checks: Natural Stat Trick (scrape, cache, polite). Evolving-Hockey GAR/RAPM optional/paid. |
| Arena lat/long/tz/altitude (rest & travel context) | Static table in `data/arenas.py` (like `mlb_engine/data/parks.py`) | Indoor sport: no weather feed. |
| Live odds: ML / PL / totals / periods / props | **The Odds API**, sport `icehockey_nhl`; featured markets `h2h, spreads, totals`; period markets (`h2h_p1`, `spreads_p1`, `totals_p1`, …p2/p3); player props (`player_points`, `player_shots_on_goal`, `player_goals`, `player_assists`, `player_total_saves`, `player_blocked_shots`, `player_power_play_points`, `player_goal_scorer_anytime/first/last`) | **Verify the exact market keys against the current Odds API docs before writing the client.** Featured markets = 1 bulk call; period/prop markets are **per-event** calls — budget below. Reuse `cfb_engine/data/oddsapi.py` caching/credit accounting. |
| Historical closing ML/PL/total prices for the backtest | Free archives are partial (sportsbookreview-style archives circa 2007–2023, MoneyPuck game files do not carry odds). Odds API historical endpoint is paid (10× credits). | **Decision needed in Phase 1**: buy one season of historical closes vs rely on a free archive. Without it, the backtest is *accuracy* only (like the MLB 2024 backtest), not edge. |
| Public splits (sharp money) | VSiN NHL splits page (same scraper family as `cfb_engine/data/vsin_splits.py`) | Existing subscriber-cookie handling from PR #378. |
| Outside benchmark | TeamRankings NHL predictions / MoneyPuck pregame win probabilities (public) | MoneyPuck's own pregame probabilities are a **free Opta-style benchmark** — capture daily like `mlb-engine opta`. |

**Odds API credit budget (rough):** featured board 1 call/refresh = 3 credits (3 markets × 1 region). Period markets: 9 keys/event; props: ~8 keys/event; at 8 games/night ≈ 16 event calls × credits-per-market. Prop/period capture must be cadence-limited (idempotent snapshots on *change*, as in `nfl_engine/data/capture.py`) and reuse the `MLBE_ODDS_CACHE_TTL` pattern.

---

## 4. Architecture — `nhl_engine/` mirrors `cfb_engine/`

```
nhl_engine/
  cli.py            run | card | close | audit | report | calibrate | probation |
                    capture (props/periods archive) | backtest | goalies (confirm/override)
  config.py         NHLE_* env knobs; Credentials (THE_ODDS_API_KEY, GMAIL_*, VSIN_*)
  schemas.py        Game/Slate (+ start time, venue, OT/SO result fields)
  pipeline.py       slate+board -> team strength -> goalie -> situational adj ->
                    period sim -> price every market -> gates -> Recommendation
  recommendations.py  Recommendation (reuse CFB fields + period, ot_rule, goalie_status)
  state.py          engine-state sync under nhl/ (copy of cfb_engine/state.py)
  data/
    nhlapi.py       schedule, boxscore, pbp, shifts, per-period scores, goalie flags
    moneypuck.py    xG/shot/skater/goalie CSVs, cached, as-of-date slicing for backtests
    preseason.py    builds/refreshes preseason_prior_<season>.json: prior-season rates
                    regressed to league (fitted), futures-implied strength, roster
                    carry-over; the Bayesian prior every in-season rate shrinks toward
    preseason_prior_<season>.json  derived asset (not synced state), rebuilt on demand
    oddsapi.py      icehockey_nhl board: featured + period + prop markets, pairing rule
    goalies.py      starter book: API flags + CSV drop-in + heuristics (B2B, last start)
    capture.py      forward archive of period & prop quotes (from nfl_engine/data/capture.py)
    vsin_splits.py  NHL splits (subclass/parametrise the CFB one)
    espn.py, injuries.py, teamnames.py (alias table: NHL API tricode <-> Odds API names)
  features/
    strength.py     team offense/defense rates: 5v5 xGF/60, xGA/60, shot rates,
                    PP/PK, finishing (shrunk), all EB-shrunk to fitted reliabilities
    goalie.py       per-goalie Sv% / GSAx shrunk; starter vs backup delta
    context.py      rest, B2B, 3-in-4, travel, tz, altitude, home/away splits
    adjustments.py  point (goal) deltas with reasons, all default to measured value or 0
  models/
    goals.py        bivariate Poisson / Dixon-Coles team goal rates per game state
    periods.py      per-period, score-state-aware Poisson simulation incl. EN goals,
                    OT (3v3) and SO; returns joint draws -> every market from one sim
    props.py        skater SOG (neg-binom on TOI×rate), points/goals/assists (Poisson),
                    goalie saves (opp shot volume × (1−Sv%)) — research_only stamp
  market/           ev.py, tiers.py, priceband.py, drift.py, mlsharp.py, ordering.py,
                    keys.py  (copied from cfb_engine; keys gain period + ot suffixes)
  audit/            ledger.py (grade every market incl. periods off per-period scores),
                    clv.py, probation.py, scorecard.py, snapshot.py, availability.py
                    (goalie-confirmation latency log), priced.py
  output/           excel.py, card.py, brief.py (goalie matchup, lines, PP units, B2B),
                    audit_report.py
scripts/nhl/        studies (reliability, dispersion fit, B2B vs close, period shape),
                    macos/linux installers + schedule (card 11:00, goalie re-run 17:00,
                    close per puck-drop pass, audit 03:00), engine.env.example
tests/nhl/
```

**Shared code rule (decided).** Engines stay separate. A new `engine_common/` package holds only *pure, dependency-free primitives* that can be shared without altering any existing engine: odds conversion (`american↔decimal↔prob` — MLB and CFB `market/odds.py` are already byte-identical), devig, EV math, the probation *statistics* (half-window mean / s.e. test with no `LedgerEntry` dependency), price-bucket binning, and the WeasyPrint/edge-tts render helpers. It is added **additively**: existing engines are only re-pointed at it where the module is byte-identical (odds.py), and only as proof the import path works. `tiers`, `probation`, `priceband`, `drift`, `ledger`, `clv` are **not** lifted — a diff shows MLB and CFB share ~10% of `probation.py`/`tiers.py`, NFL has no `priceband`/`drift`, and reconciling them is a design decision for a separate post-season PR. NHL builds its own versions on the `engine_common` primitives, using the CFB modules (newest) as the template. Also reused as-is via import: `mlb_engine.state` git plumbing, `lineshop` (`--sport nhl`), the Odds API HTTP/caching base.

### Market keys (ledger `market` column)

`game_ml`, `game_pl` (±1.5), `game_total`, `p1_ml3` (3-way), `p1_ml` (2-way where quoted), `p1_pl` (±0.5/±1.5), `p1_total`, same for `p2_*`, `p3_*` (3rd-period markets grade on regulation-3rd only — confirm the book rule; most exclude OT), `team_total_reg` / `team_total_inc_ot` (books differ on whether the SO winner's credited goal counts toward a team total — a 3–3 game won in the SO settles 3 or 4 depending on the house; the two are separate markets), props: `sk_sog`, `sk_pts`, `sk_g`, `sk_a`, `sk_blk`, `sk_ppp`, `g_saves`, `ags` (anytime goal; shootout goals never count as player goals). Every row carries `ot_rule` (`incl_ot_so` | `reg_only` | `incl_ot_only`) so grading cannot mis-settle. **Settlement rules are verified per book in Phase 0** for PL, totals, team totals and 3rd-period markets and stored in `data/book_rules.py`; a quote whose book rule is unknown is refused by a `settlement_unverified` gate (stamped as `pass_gate`, graded like every other screen) rather than guessed. **Rules drift**, so each entry carries `verified_on` and **expires after 60 days** — the gate flips back to `settlement_unverified` until re-verified (the CFB price-band expiry pattern); re-verification is a season-start and post-provider-change runbook item. The audit grades every OT/SO-affected row under *both* `reg` and `inc_ot` and flags any book where the two differ and the recorded rule is stale, so a mismatch surfaces the first night it could matter. Reconciliation against a book's actual payout cannot be automated from the Odds API and stays manual.

---

## 5. Model design

### 5.1 Team strength (Phase 1)

**Season start — the preseason anchor.** "Season-to-date" is undefined on opening night
and a one-game sample on day two, so no rate is ever computed from the current season
alone. Every team/goalie/skater rate is an **empirical-Bayes posterior**:

    rate = (k · prior + n · observed) / (k + n)

where `n` is the current-season exposure (5v5 minutes, PP opportunities, shots faced)
and `k` is the *fitted* exposure at which the season sample deserves equal weight —
the reliability study's output, per metric (the same EB machinery the MLB pen study
and §5.2 goalie layer use). This replaces a hand-set "20-game linear ramp": a linear
ramp is a fixed schedule that does not know a PP% needs far more games to stabilise
than a shot-attempt rate. `k` gives each metric its own ramp, and the prior never
fully disappears — it just becomes negligible once `n ≫ k` (for shot rates that is
~15–25 games; for shooting % and PK % it is most of a season, which is the point).

The prior for team *T*, metric *m* comes from `data/preseason.py`, which writes
`preseason_prior_<season>.json` once before opening night (rebuilt on demand, not
synced state — the `ros_prior.py` pattern: derived, refreshable, loud fallback):

1. **Prior-season rate regressed to league** at the fitted regression weight (split-
   season r from the reliability study: last season's 5v5 xGF/60 explains perhaps
   30–50% of this season's; shooting % near 0).
2. **Roster carry-over**: the prior-season rate rebuilt from the players expected on
   the opening roster (§5.7 lineup rebuild, run on last season's on-ice numbers), so a
   team that lost its 1C or changed goalies starts where its new roster says, not
   where last year's did. Goalies carry their own multi-season priors (§5.2) and are
   not part of the team prior at all.
3. **Futures-implied strength** (Odds API `outrights`: Stanley Cup, points totals /
   division odds, devigged): converted to a season goal-differential scale by a
   mapping fitted on 2015–25 preseason futures vs realised goal differential.
   Blended with (1)+(2) at a weight the same backtest fits — the market's preseason
   view is the CFB `preseason.py`/Makinen-ensemble idea, and like there the weight is
   measured, not assumed. If futures are unavailable the prior is (1)+(2) and the
   card's methodology footer says so.

The prior is refreshed weekly through October (roster moves, opening goalie
decisions) and frozen thereafter; the file version travels on every Recommendation
so the ledger can cut "first 20 games" performance against "prior version". Opening-
week buys are a probation *candidate* from day one ("October only"): the hypothesis
that early-season markets are softer is tested in the ledger, not asserted.

**In-season update.** Per team, as-of the slate date, `observed` in the formula above
is the current-season rate with exponential decay (half-life fitted, expect 20–30
games) and `n` its effective exposure:
- 5v5 xGF/60, xGA/60 (score- & venue-adjusted, MoneyPuck), shot attempts (CF/60, CA/60), high-danger share.
- PP: xGF/60 on PP, PPO drawn/60; PK: xGA/60 on PK, PIM taken/60.
- Finishing: goals − xG per shot, shrunk heavily (skater finishing has some repeatability, team finishing very little).
- **Reliability study first** (`scripts/nhl/reliability_study.py`): split-half and block-to-block r for each rate; fit EB `k` so a season-sized sample keeps exactly the measured reliability (the exact method the MLB pen study used). Nothing is priced above its measured repeatability.
- Output: expected goals for/against per 60 at 5v5, PP, PK for the matchup → λ_home, λ_away by game state.

### 5.2 Goalie layer (Phase 1)
- Starter identification: NHL API flag → CSV drop-in → heuristic (didn't start yesterday; B2B second night → backup with p≈0.8). Status enum `confirmed | probable | projected | unknown`.
- Skill: **GSAx/60 is the skill input; Sv% is display-only.** GSAx (goals saved above expected, xG per shot faced) already nets out shot quality, which is how a goalie who played behind an elite defense stops looking like a god when he moves behind a bad one — raw Sv% cannot do that and never enters the sim. Two seasons + current, shrunk to league at fitted k (expect thousands of shots). Backups get a wider prior.
- **Team-change residual**: xG models miss screens, pre-shot movement and rebound control, so some team-defense contamination survives in GSAx. Study (`scripts/nhl/goalie_team_change_study.py`, 2015–25): GSAx residual in season N+1 for goalies who changed teams, against the defensive gap (xGA/60 and HDCA/60) between old and new team. If the residual scales with the gap, the carry-over prior is shrunk further by a fitted factor of that gap; if not, the knob ships at 0 with the study in the docstring.
- **Age curve**: goalie decline is not linear — fit an age term (piecewise, from MoneyPuck 2010+ goalie seasons) and apply it to the prior mean, not the observed rate; expect a knee in the mid-30s but *fit it*, do not assume it.
- **Cold starts (TSLS)**: time-since-last-NHL-start as a feature. Study (`scripts/nhl/goalie_cold_start_study.py`): Sv%/GSAx residual vs the closing ML for starters with TSLS > 7/14/21 days. If the penalty is real, ship it as a whole-game Sv% step-down first; only period-shape it ("first 30 minutes") if the PBP shows the damage is early-game concentrated.
- **Call-ups (`MIN_GAMES_PRIOR`)**: a goalie under a minutes floor (start at 180 NHL minutes; fit) is anchored to the **empirical distribution of call-up starts** (fit from history: mean, spread), not league average and not a guessed percentile. This is the single largest expected leak in the goalie layer — a `.905` prior on an AHL call-up overvalues his team every time — so the anchor ships in Phase 1, before the first live card.
- **Backup rust**: a backup who has not started in >14 days gets the TSLS treatment above; the B2B heuristic that predicts a backup start also predicts the rust.
- Gate: `goalie_unconfirmed` veto on ML/PL/team totals/saves until status ≥ probable; `run` at 11:00 prices with projected starters and the 17:00 re-run reprices with confirmations (both boards saved; the card that is emailed at 17:00 is the graded one — MLB's four-pass lesson, PR #327).
- Latency log: the moment a starter is known vs the ML at that moment → after a season, decide if "we hear first" is ever true (CFB's answer for QBs was no).

### 5.3 Game & period simulation (Phase 2) — the core
- Rates: 5v5 λ per minute by score state (leading/tied/trailing: score effects ~±10%), PP/PK λ with a stochastic penalty process (~3 PPO/team, 2 min each, 5v4 conversion rates).
- **Shorthanded scoring is team-specific, not a league constant.** In a PP block the shorthanded side's λ = its own 4v5 xGF/60 × the PP side's 5v4 xGA/60 (both MoneyPuck strength-state rates, EB-shrunk with their own fitted `k` — SH samples are small). Aggressive PP structures leak high-danger rushes and this is where alt puck lines and total tails pick it up. Giveaway/takeaway counts are **not** used as the conditioning input (see arena bias in §3). Sub-state samples are thin, so the reliability study reports `k` for 4v5/5v4 explicitly (expect most of a season); if a sub-state's split-half r at a season's exposure is < 0.2 the team factor ships at exactly 1.0 (league rate) with the study in the docstring. Nightly audit diagnostic: shorthanded goals predicted vs realised.
- **Empty-net logic is dynamic, not a 2:00 rule.** Pull time is a function of (deficit, time remaining, man-advantage state) **and coach**, with coach pull-time profiles *measured* from play-by-play goalie-off events (the NHL PBP carries them) and refit monthly — never a hand-typed tendency table. Modern pulls run 3:00–4:30 down 2 and earlier on a late PP; a static 2:00 trigger mis-prices alt puck lines (±2.5), 3rd-period totals and game totals. While pulled, both λs scale (EN-for ~0.17 goals per pull, pulled-goalie scoring ~0.04; refit). The sim's realised EN rate vs actual is a nightly diagnostic in the audit report.
- **PK fatigue (registered candidate, not shipped)**: hypothesis — heavy early penalty time (>6 PK minutes in 40) raises opponent 5v5 λ later in the game. Unsourced as a coefficient; backtest on PBP first (`scripts/nhl/pk_fatigue_study.py`), and if real, ship as an `adjustments.py` multiplier at the measured value. Until then it is graded as a probation *candidate* on the buys it would have refused, like every other screen.
- Regulation tie → **OT as its own block, not a 5v5 multiple.** 3v3 λ per side = league 3v3 base rate (~55–60% of OTs end with a goal within 5 min) × team 3v3 xGF/xGA factors **heavily shrunk** toward 1.0 (a team plays ~20–25 OT minutes a season; unshrunk 3v3 rates are noise and would bleed on dogs faster than the crude multiplier) × goalie GSAx factor, with the link from 5v5 strength to OT outcome *fitted* on 2015–25 (it is empirically weak, which is what keeps 2-way ML dogs honest). Goalie pulls in OT are not modelled unless the PBP shows goalie-off events in OT at a non-trivial rate (unverified claim; the study will say). → SO (p_home ≈ .50–.52, goalie SO record shrunk to ~0).
- One draw yields: per-period goals for each side, regulation result, OT/SO result → every market's probability from the same joint distribution: ML 2-way and 3-way, PL ±1.5/±2.5, totals at each rung, team totals, period 3-way MLs, period PL ±0.5/1.5, period totals, "both teams score", 1st goal.
- **Dispersion calibration** (L13): fit Dixon-Coles ρ / a shared latent factor so the sim's implied ML↔PL↔total relationships match the *market's* on 60+ live boards (RMSE of sim-vs-market implied probabilities). This is the NHL version of `CFBE_ML_SD_BASE/KNEE/SLOPE`. Do it before any dog is allowed to read as value.
- Calibration: isotonic per market from the backtest (`nhl_engine/data/calibration_<season>.json`), confidence shrink past a pivot (copy `cfb_engine/calibration.py`), then `calibrate` refits from the ledger with a holdout.

### 5.4 Situational adjustments (Phase 2, ship at measured value or 0)
B2B second night (home and away separately), 3-in-4, rest days ≥3, travel miles / tz shift, altitude (COL/UTA), post-long-homestand first road game, "schedule loss" flag. Each measured against the **closing ML** on 2015–2025 (`scripts/nhl/schedule_study.py`); if the closing line already carries it (the CFB result for nearly everything), the knob ships at 0 and the study lives in the docstring.

### 5.5 Player props (Phase 3, `research_only` until the archive grades)
- **SOG (first)**: mean = projected TOI (EV+PP, shrunk to player's recent 10/25 games and season) × individual shots/60 by strength × opponent shots-against factor, **conditioned on the game sim's score-state matrix**: expected SOG = Σ over sim paths of (minutes trailing × trailing shot rate + minutes tied × tied rate + minutes leading × leading rate), with the skater's per-state rates shrunk toward his team's and the league's score-effect curve (individual splits are thin). This is what stops the model buying Overs on skaters whose team builds an early lead and coasts — the leading-state rate and the compressed 3rd-period TOI both fall out of the same draws that price the game markets. Distribution: negative binomial with variance fitted linearly in the mean (the NFL `player.py` approach). Pseudo-line Brier vs base rate is the Phase-3 gate; a positive result is *necessary*, not sufficient (L9).
- Points/goals/assists: Poisson on individual G/60, A1+A2/60 and PP unit slot; linemate-adjusted only if the lines CSV is present. Anytime goal-scorer is priced from the goals Poisson; ships `NO_BUY` or in a `+150..+300` band pending the ledger (L10).
- Saves: opponent shot volume from the sim × (1 − shrunk Sv%), conditional on confirmed starter; correlated with the team-total under of the same game — dedupe (L11).
- Blocked shots, PP points: price and archive, do not bet.
- Capture archive from Phase 0: every prop and period quote, both sides of the same book, timestamped, idempotent on change.

### 5.7 Injuries & adjustments — how an absence reaches (or does not reach) a price

The inherited rule (CFB QB study, NFL `availability.py`, MLB #154): **an absence is a
line on the card and nothing else until a study shows the market mis-times it.**
Two tracks, kept apart in code and in the audit.

**Track 1 — record everything (day one, never prices).** `data/injuries.py` captures
every designation with `(source, posted_at, seen_at, status)`, the ML/total at the
last archived board *before* the posting and the first *after* it → the availability
log. The card prints each absence with a role tag from our line clustering
(`1C/1D/PP1/G1`), the replacement, and one of `[already priced in]` /
`[reported, not scored]` / `−x.x` (the sim charged it). After a season the latency
study answers "do we ever hear first" per role; the CFB answer for QBs was no.

**Track 2 — what moves λ, and only where measured.**
- **Goalies** (the one injury hockey prices explicitly): no adjustment knob. The sim
  runs with whoever is projected/confirmed, using that goalie's shrunk Sv%/GSAx with
  the age/TSLS/call-up anchors of §5.2; `goalie_unconfirmed` vetoes ML/PL/team
  totals/saves until status ≥ probable. The AM-vs-PM CLV cut (B.4) measures what
  pricing before confirmation costs.
- **Skaters — lineup-driven strength rebuild, not hand-typed deltas.** Team 5v5 and
  PP/PK rates are rebuilt from the players expected to dress:
  `λ_team = Σ_players (projected TOI share × on-ice xGF/60 or xGA/60, shrunk)`,
  then rescaled to the team's season rate so a full lineup reproduces §5.1 exactly.
  A missing 1C or PP1 quarterback lowers λ by what *his* shrunk numbers say
  relative to his replacement's, nothing more.
  **The player rate must be isolated, not raw on-ice.** A winger's on-ice xGF/60 is
  contaminated by the centre he plays with; rebuilding from raw on-ice rates after
  removing the centre keeps the centre's value alive in his wingers' numbers and
  understates the loss. So the per-player input is an **isolated impact** — a
  ridge/RAPM-style regression of shift-level xG on the ten skaters on the ice (fit
  from MoneyPuck shot xG + NHL shift charts; Evolving-Hockey RAPM as an external
  cross-check), or at minimum a WOWY adjustment — so each player's contribution is
  net of linemates. The centre's absence then shows up correctly twice: his own
  isolated value leaves, and the wingers are credited only their own. Isolated
  estimates are noisier than on-ice rates, so their shrink `k` is larger (fitted in
  the reliability study).
  **The lineup rebuild is the primary λ; the team rate is only its prior.** The
  per-player sum is *not* rescaled to the team's season rate — that rescale would
  drag minutes played by since-traded players back into tonight's number. Instead
  the team-level EB rate of §5.1 is the shrink target and `n` is counted on **player-
  minutes of the currently dressed roster**, so a team that sells three isolated-
  impact skaters at the deadline reprices the same night: the departed are simply
  absent from the sum, and no exponential decay has to "notice" it over 25 games.
  No discontinuity trigger or threshold is needed; a *diagnostic* — share of the
  team's isolated xG turned over in the last 7/30 days — is printed on the card
  and cut in the audit ledger so the mechanism can be seen working (and probation
  can grade "post-deadline buys" as a candidate). Unit test: a healthy, unchanged
  roster reproduces the §5.1 team rate within the shrink tolerance. Shrinkage `k` for individual on-ice
  rates comes from the Phase 1 reliability study (skater on-ice xG is noisy; expect
  heavy shrink, so a star's absence moves λ a few percent, not a goal). The
  replacement's rate is his own if he has ≥ the minutes floor, else the team's
  depth-line average. Fallbacks, in order: confirmed lines (17:00) → RotoWire
  projected lines (11:00) → last game's shift-chart lines minus newly-out players →
  season team rate with every absence tagged `[reported, not scored]`. The card
  always says which level it used.
- **Situational** (B2B, 3-in-4, rest ≥3, travel/tz, altitude, coach EN tendencies,
  PK fatigue): one knob each in `features/adjustments.py`, default = the value the
  closing-line study (`scripts/nhl/schedule_study.py`, 2015–2025) measures, or **0**
  when the close already carries it. The study result is quoted in the docstring
  beside the default, as CFB does for `CFBE_INJURY_QB_PTS`.
- **Every applied adjustment is a stamped `reason` with its size** on the
  Recommendation, so the ledger can be cut by "buys where injury/lineup rebuild moved
  λ by > x" and by adjustment name; probation grades each cut and can retire a knob
  that loses. No adjustment ever bypasses the gates or the market anchor.

Phase 1 deliverables for this section: `features/lineup.py` (rebuild + fallbacks),
reliability `k` for skater on-ice rates, a unit test that a full healthy lineup
reproduces the team rate to 1e-9, and a backtest cut: ML Brier with vs without the
lineup rebuild on 2023–25 games where a top-6 forward or top-pair D was out.

### 5.6 Gates & selection (copied, then measured)
`min_edge 0.02`, `strong_edge_gap 0.02`, `max_edge` (start 0.06 — NHL markets are efficient), `MAX_BUY_ODDS` per market (`game_pl` ≤ +109 to start, mirroring run lines), drift gate (first-seen board write-once), price band per market, `ml_no_sharp_money` (ML only), `goalie_unconfirmed`, `one_way_quote`, `research_only` (props). All stamped as `pass_gate`, all graded by probation.

---

## 6. Phased plan (sessions = Devin work sessions; calendar waits called out)

**Phase 0 — Capture first (1 session).** `nhl_engine/data/oddsapi.py` + `capture.py` + `state.py` + `nhlapi.py` schedule/scores; `nhl-engine capture` on a schedule (every 30 min on game days, idempotent). Verify Odds API NHL market keys and per-event credit cost with one real call. Archive opens *before* the model exists. **Start this before the season's first slate** — the archive is the only thing that cannot be back-filled. **Mac desktop shortcuts ship in Phase 0** (user request): `scripts/nhl/macos/` with `run_predictions.command`, `run_audit.command`, `nhl_capture.command`, `open_ledger.command`, `install_shortcuts.command` and `install_schedule.command` (launchd), mirroring `scripts/cfb/macos/` and sourcing `scripts/macos/_repo.sh`, so every pipeline step can be run manually by double-click.

**Phase 1 — Data + studies (2 sessions).** MoneyPuck/NHL API clients with as-of slicing; team-strength and goalie reliability studies; dispersion/market-relationship fit; schedule-effect study vs the close; decide on historical odds purchase. Deliverable: numbers in docstrings and a `config.py` whose defaults quote them.

**Phase 2 — Game engine & card (2–3 sessions).** `models/goals.py`, `models/periods.py`, pipeline for ML/PL/totals/period markets, isotonic calibration from the accuracy backtest, Excel + card + brief (goalie matchup, PP/PK, rest), email, installers, schedule (11:00 card, 17:00 goalie re-run, **per-game pre-drop pass** every 15 min from 60 to 30 min out and every 5 min inside T-30, reading the NHL API post-warmup roster; each write stamped `(seen_at, api_last_updated, minutes_to_drop)`. It is a **lead-time log, not truth**: the bet record is write-once at 17:00 and CLV grades against the last pregame *price*, so a stale roster cannot touch either; the roster of record for every audit cut is the next-morning **boxscore**, and pre-drop snapshots that disagree with it are marked `stale` and excluded — per-puck-drop close passes, 03:00 audit). Ledger grades every market off per-period scores with the correct `ot_rule`.

**Phase 3 — Props (1–2 sessions).** `models/props.py`, prop pricing off the archive, `research_only` stamp, pseudo-line Brier study, `props_grade` against boxscores. No prop is bettable in this phase by construction.

**Phase 4 — Live probation (calendar: first 4–6 weeks of slates, ~1 session of tuning).** Run daily; CLV, probation, price buckets, Needs %; turn on nothing new without the three-test verdict. Candidate screens (goalie-confirmed-only, B2B fade, period-1 under band) registered as candidates, not shipped.

**Phase 5 — Extensions (later, evidence-gated).** Alternate puck lines/totals via lineshop, team totals, live-period repricing at intermissions (period markets are quoted in-play; the drift gate + pregame-only rule need an "intermission close" variant), MoneyPuck/TeamRankings outside benchmark AUC table on the card.

Estimated total build: **~7–9 sessions** plus the calendar time the probation bar demands.

---

## 7. Success criteria (what "working" means, in this repo's language)

1. Every buy has a CLV column filled from a pregame close; the market-level CLV sheet is positive on the markets we keep after 100+ buys and both halves agree.
2. The engine's Brier/log-loss vs the devigged close is reported per market on every audit; markets where the market wins by a wide margin get anchored or shut, by probation verdict, not by hand.
3. The forward prop/period archive grows daily with zero duplicate snapshots and both sides per book.
4. No component ships with an unmeasured coefficient; every default in `config.py` quotes the study that set it, including the ones set to 0.
5. `ruff`, `mypy`, `pytest -q` stay green; `nhl-engine run` is free with `NHLE_DATA_DIR` pointed at scratch, `NHLE_STATE_SYNC=0` and a long odds cache TTL (same safety rails as MLB/NFL).

---

## 8. Open decisions for you

1. **Historical closing prices**: buy a season via the Odds API historical endpoint (paid credits) or accept an accuracy-only backtest + forward CLV like NFL props? Recommendation: forward CLV first; buy only if Phase 1 shows a strength model that is not already the market.
2. ~~Shared `engine_common/` refactor~~ **Decided**: additive `engine_common/` of pure primitives only; engines stay separate (see §4).
3. **Books**: which accounts do you hold for NHL? Sets `DEFAULT_BOOKS` in lineshop and the price-band evidence.
4. **Period-market scope**: pregame only (fits the current pregame-close discipline) or also intermission repricing (needs new close/grade rules)?
5. Confirm "NIL" meant the NFL engine, and "runtiness" meant the period markets.
6. ~~RotoWire/DailyFaceoff scraping~~ **Settled**: RotoWire (already scraped by three engines) is the morning soft projection and the 17:00 confirmation source; the **NHL API gamecenter roster after warmups** (~20–30 min before drop) is the final-lineup source, read by a per-game pre-drop pass (§6 schedule). DailyFaceoff is optional and off the critical path. Note: the pipeline is **cron-scheduled, not event-driven** — the pre-drop pass is a staggered cron like the `close` passes, not a webhook.

---

## 9. Reviewer additions folded in (2026-09-18)

| Item | Status in plan | Where |
|---|---|---|
| Goalie age curve, TSLS cold-start penalty, call-up `MIN_GAMES_PRIOR` anchored to empirical call-up distribution | **Ships Phase 1**, coefficients fitted not assumed | §5.2 |
| Score-state-conditioned SOG (trailing/leading rates × simulated minutes) | **Phase 3**, on the same sim draws as the game markets | §5.5 |
| Dynamic empty-net pull (deficit × time × coach), coach profiles measured from PBP | **Ships Phase 2**, EN diagnostic in audit | §5.3 |
| PK-fatigue multiplier | **Registered candidate**, backtest first, probation-graded | §5.3 |
| Injury/lineup/goalie sourcing (RotoWire table+RSS, RotoWire lineups, DailyFaceoff vote, NHL API truth, shift-chart line clustering) | Added | §3 |
| `engine_common/` scope and separation rule | Decided | §4 |
| Goalie new-team mirage: GSAx is the only skill input (Sv% display-only) + team-change residual study; linemate ripple: per-player isolated (RAPM/WOWY) impact instead of raw on-ice rates in the lineup rebuild | Added | §5.2, §5.7 |
| Deadline/roster turnover: lineup rebuild is primary λ with `n` on dressed player-minutes (no rescale to team season rate), turnover diagnostic + probation candidate instead of a threshold trigger | Added | §5.7 |
| Team-total OT/SO settlement split (`team_total_reg` / `team_total_inc_ot`), SO goals never player goals, per-book rules in `book_rules.py`, `settlement_unverified` gate | Added | §4 market keys |
| Final lineups from NHL API post-warmup roster via per-game pre-drop cron pass; RotoWire as soft projection; DailyFaceoff optional; plan is cron, not event-driven | Added / Decision 6 settled | §3, §6, §8 |
| Team-specific shorthanded λ in PP blocks (4v5 xGF/60 × 5v4 xGA/60, shrunk) | Added | §5.3 |
| OT as its own block: league 3v3 base × heavily-shrunk team 3v3 factors × goalie, fitted 5v5→OT link; OT goalie pulls unverified/not modelled | Added (replaces 5× multiplier) | §5.3 |
| Arena tracking bias: xG not coordinates, rink-bias study, coordinate correction before RAPM, tracking-count stats barred as inputs | Added | §3 |
| Execution risks: pre-drop pass is a stamped lead-time log with boxscore as roster of record; SH sub-state `k` reported with r<0.2 → factor 1.0 and SHG diagnostic; `book_rules.py` entries expire after 60 days with dual-rule grading flag | Added | §6, §5.3, §4 |
| Season-start anchor: `preseason_prior_<season>.json` (prior season regressed + roster carry-over + futures-implied), EB posterior with fitted `k` per metric instead of a fixed 20-game ramp; opening-week buys as probation candidate | Added | §4, §5.1 |
| Injuries & adjustments: record-everything track, lineup-driven skater strength rebuild with fallbacks, situational knobs at measured value or 0, stamped reasons audited by probation | Added, Phase 1 deliverables listed | §5.7 |
