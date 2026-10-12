# College Basketball (NCAAM) Payoff Engine — Master Plan

Written 2026-10-01 from a read of `mlb_engine/`, `cfb_engine/`, `nfl_engine/`, `nhl_engine/`,
`nba_engine/`, their audit modules, the MLB totals-sheet/worksheet audit
(`docs/mlb_audits/2026/`) and the NHL ledger audits of 2026-09-29 → 2026-10-06.

Labels used throughout: **[VERIFIED]** = probed from this VM today, **[MEASURED]** = a number
from a graded ledger or study in this repo, **[HYPOTHESIS]** = a design assumption to be fitted
in Phase 1 before it prices anything. Nothing unlabelled is a claim.

---

## 0. The short version

Five engines, one verdict: **the closing line beats our ratings on every main market** (MLB
Brier .2198 vs .2149 on 140,630 rows; CFB .2151 vs .2063; NFL ridge MAE 10.28 vs 9.91; NHL buys
28/29 negative CLV, p<0.001 [MEASURED]). Where money was made (CFB +20u on 192 buys) the gates
selected, the ratings did not. So the CBB engine is **market-anchored like NBA**:
`p = w·p_model + (1−w)·p_fair_close`, `w` fitted per market, edge cap, and every filter earns
its coefficient against the close or ships at 0.

What makes college basketball different, and worth an engine at all:

* **~364 D-I teams, ~5,900 games/season** (ESPN scoreboard 2026-03-01: 23 D-I games on one
  Sunday; a weekday in January has 60–100) [VERIFIED volume]. Volume reaches the 100-buy
  probation bar in weeks, not months.
* **The market is uneven.** Power-conference games are NBA-liquid; mid-major weeknight games are
  priced off KenPom-style ratings with low limits and few books. The market's own quality is a
  variable we measure (books quoting, line dispersion, hold) and the engine's job is to find
  the segment where the close is beatable — the NHL audit says a model that beats nothing
  everywhere beats nothing.
* **Tempo-free efficiency is the established language** (points per 100 possessions, four
  factors). Free, public, daily, with game logs back to 2008: Bart Torvik's T-Rank CSVs
  [VERIFIED].
* **Roster churn is extreme** (transfer portal + one-and-dones + 5th years): the early-season
  prior is the hard problem and the one with the most published structure (returning
  minutes, recruiting, transfers).
* **No official injury report.** Availability comes from RotoWire's CBB injury page [VERIFIED],
  ESPN notes and box scores; for mid-majors often only the box score. The engine must price
  "we don't know" honestly.
* **Weather is not a factor** (indoor); it is omitted by design, not forgotten. Travel,
  altitude, rest and neutral sites take its place.

---

## 1. Databases — what each one feeds (probed 2026-10-01)

| # | Source | Seasons | Feeds | Status |
|---|---|---|---|---|
| D1 | **The Odds API `basketball_ncaab`** live | from Phase 0 | ML / spread / total, `spreads_h1`, `totals_h1`, `h2h_h1`, team totals, player props (where quoted); every pass archived; close = last pre-tip snapshot | key is live; sport listed, `active=false` in the off-season (season starts early Nov) [VERIFIED]. Books on this key for US basketball today: ballybet, betmgm, betparx, betrivers, bovada, draftkings, espnbet, fanduel, hardrockbet, kalshi, mybookieag, **novig** [VERIFIED on `basketball_nba`]; **no Circa, no Pinnacle in the US regions** — `novig` (no-vig exchange) is the candidate sharp reference, Pinnacle via `regions=eu` to be verified in Phase 0 |
| D2 | The Odds API **historical** | 2020-06 → (10-min snapshots; 5-min from 2022-09) | **2024-25 game-line closes bought in Phase 1** (decided §10.1); 1H deferred | paid credits; **the key in this VM's env shows 497 credits remaining [VERIFIED 2026-10-01]**, not the 5M the NBA plan recorded — confirm which key holds the balance before the pull; cost ≈ 60–100 events/day × 2 snapshots × 1 call ⇒ ~40–60k credits for a season of game lines |
| D2b | **Futures / win-total fallback** (`data/futures_scraper.py`) | pre-season | Devigged regular-season win totals and conference futures for the market-implied prior (§3.2), captured in October while `basketball_ncaab` is inactive (`basketball_ncaab_championship_winner` *is* active [VERIFIED]); sources: Odds API championship futures + a scraped consensus board (verify in Phase 0) | required before the first slate |
| D3 | **Bart Torvik T-Rank** `barttorvik.com/<year>_team_results.csv` | 2008 → 2026 | Team AdjOE / AdjDE / AdjT / Barthag, SOS, WAB, conference splits — 45 columns, daily | [VERIFIED] 200, 213 KB |
| D4 | **Torvik game logs** `getgamestats.php?year=YYYY&csv=1` | 2008 → | Per team-game four factors, tempo, OE/DE, box line; ~12,000 rows/season | [VERIFIED] 200, 4.9 MB |
| D5 | **Torvik super schedule** `<year>_super_sked.csv` | 2008 → | Every game with Torvik's own pre-game projected spread/total/win% and the final — a free *forecast baseline* to beat before buying the market's close | [VERIFIED] 200, 6,293 rows × 55 cols for 2025; column map undocumented → Phase 1 mapping study |
| D6 | **ESPN site API** (`mens-college-basketball/scoreboard`, `summary`) | live + history | Schedule, tips, venue, `neutralSite`, `conferenceCompetition`, finals, **half scores**, box scores, PBP; grading feed | [VERIFIED] 434 KB for one date, `neutralSite` present, **no odds in the payload** |
| D7 | **RotoWire CBB injury report** | live | Out/doubtful/questionable with notes | [VERIFIED] 200; parser pattern from `nhl_engine/data/rotowire.py` |
| D8 | **VSiN betting splits (CBB)** | live | Handle / tickets per side (DK, Circa) for the ML sharp-money gate and the sheet's book column | [VERIFIED] redirects to `/betting-splits/?source=DK&sport=CBB`; port `cfb_engine/data/vsin_splits.py` |
| D9 | Sports-Reference CBB | all | Cross-check of ratings, rosters, preseason AP poll, returning players | [VERIFIED] but rate-limited (429 on second hit) — nightly, cached, Mac only |
| D10 | **KenPom** | 2002 → | Offline cross-check only (preseason AdjEM, luck) | **paid, 403 unauthenticated** [VERIFIED]; **never in the live pipeline** — manual CSV drop-in for offline verification, the card does not depend on it (decided §10.2) |
| D11 | Torvik preseason projections / returning-minutes pages | yearly | Early-season prior inputs: returning minutes %, transfers in/out, recruiting | page exists; `trank.php` sits behind a JS browser check [VERIFIED] → fetch via the Mac Chrome/Playwright route or the per-season CSV mirrors; mark **unverified until it returns data** |
| D12 | hoopR / sportsdataverse MBB data (PBP, box, lineups) | 2002 → | Possession-level PBP, player box logs for the minutes model and props | GitHub API rate-limited from the VM today — **unverified**; Phase 0 verifies on the Mac |
| D13 | Arena table (lat/lon, altitude, tz) | static | Travel miles, time zones, altitude (Wyoming, Air Force, Colorado State, Utah State, BYU, Denver, New Mexico…) | built once; ESPN venue ids |
| D14 | Official results = ESPN finals + half scores | — | Grading of game, 1H, team totals; props from the ESPN box | same as NBA |

**Fail-safe rule.** Torvik (D3/D4) is the rating spine; if the fetch errors or returns a stale
file (date older than yesterday), the card **halts with an error email** rather than pricing off
unadjusted historical priors. ESPN schedule failure halts the same way. The capture job never
halts (prices are the one thing that cannot be re-fetched).

Every row carries `source`, `captured_at`, `season`, Odds API `event_id` + ESPN `game_id`
(matchup text is display only — MLB doubleheader lesson, PR #423) and a `team_alias` table
(Torvik names ≠ ESPN names ≠ book names; ~364 teams, built once, tested).

---

## 2. Markets and outputs

| Market key | Model output | Settlement | Phase |
|---|---|---|---|
| `game_ml` | P(home win) incl. OT | ESPN final | 2 |
| `game_ats` | P(cover) at every quoted line (alt lines via lineshop) | incl. OT | 2 |
| `game_total` | P(over) at every line | incl. OT | 2 |
| `team_total_home` / `team_total_away` | P(over) team points | incl. OT | 2 |
| `h1_ml` / `h1_ats` / `h1_total` | first-half distribution (fitted, not a half-share — NBA measured +0.77 pts/team 1H vs 2H; CBB has its own number to measure, and the under-10-minute "foul game" inflates 2H) | ESPN half scores | 2 |
| `h1_team_total` | where quoted | half scores | 3 |
| `h2_*` | **not pregame** (quoted at half) → Phase 5 only, needs an "intermission close" | — | 5 |
| `pl_pts`, `pl_reb`, `pl_ast`, `pl_3pm`, `pl_pra` | `research_only` from day one, like NFL/NHL props: no historical CBB prop prices exist, archive first | box, `dnp_void` | 3 |

Every row carries `pass_gate` (which named gate refused it) and `tier`; refusals are priced and
graded as the counterfactual (L6/NBA-L3). Per-market `MAX_BUY_ODDS` (ML dogs > +200 refused by
default, the CFB priceband inheritance; CBB mid-major dogs at +600 are exactly the tail the MLB
Strong tier drowned in).

**Measured on every market, every slate, kept apart** (the NBA §2 table, unchanged): model
accuracy (Brier/log-loss/MAE) · same metric for the no-vig close and open on the same rows ·
buys vs model-favoured vs naive baselines (favourite, home, always-Over, **Torvik's own pick** —
D5 gives us a free public forecaster to beat) · CLV in prob-points and EV · W-L-P/ROI by market,
tier, price band · PPV/NPV vs base rate · calibration · each gate's refused-row table ·
integrity (one-way quotes, closes after tip = 0, stale, no-board).

**Extra CBB cuts the audit must always show**: power vs mid-major (by `market_quality` bucket:
books quoting, line dispersion), home / away / **neutral**, conference vs non-conference,
November (prior-heavy) / Dec–Feb / March (conference tournaments + NCAAT), and games with a
`starter_unknown` flag.

---

## 3. Model

### 3.1 Team strength — tempo-free, opponent-adjusted, sequential

Same spine as `nba_engine/models/ratings.py` (Kalman mean+variance per team, sequential
no-look-ahead replay, league level and HCA as running means), on **per-100-possession**
observations:

```
home pts/100 = league + hca·(1 − neutral) + off(home) + def(away)
away pts/100 = league − hca·(1 − neutral) + off(away) + def(home)
possessions  = league pace + pace(home) + pace(away)
```

* Fitted on D4 2008–2025 by walk-forward replay; graded against Torvik's projection (D5) and,
  where D2 is bought, the close. Report MAE on margin and total, and **the partial correlation
  of our rating with the final after the close** — CFB's efficiency layer scored −0.001 there;
  if CBB's does the same, `w` goes to ~0 and the engine is a selection/timing engine, said
  plainly.
* `neutral` from ESPN's flag; HCA is **fitted per venue tier** (Cameron ≠ a 2,000-seat gym)
  [HYPOTHESIS: per-team HCA repeats; test split-half].
* Four-factor sub-ratings (eFG%, TO%, OR%, FT rate, 3PA rate, 3P% for and against) are kept as
  **features**, each shrunk at its own stabilization window (§3.3), and feed the distribution
  shape (variance, 3-point dependence) rather than the mean.
* Garbage-time filter: possessions after the game is decided (Torvik/KenPom convention — lead
  > some fitted function of time) are down-weighted in the fit [HYPOTHESIS, fit the cutoff].

### 3.2 Early-season prior (the hard part; where the preseason market is beatable *if* anywhere)

The first ~8 games are ~25% of the pre-conference schedule and the thinnest-priced. Each team's
opening rating is a weighted blend, every weight **fitted on 2010–2025 preseason-vs-March
outcomes**, not assumed:

1. **Last season's final rating** regressed at the fitted year-to-year r (expected low: ~0.5–0.6
   for AdjEM [HYPOTHESIS]) — *conditional on roster continuity*.
2. **Returning production**: % of minutes, points, possessions used, and OR%/BLK% returning
   (CFB measured returning production at 51.96% ATS — null *against the close*. Here it enters
   the **prior**, not a bet; the test is whether the prior's residual on the close is zero.)
3. **Transfer portal**: incoming players' prior-level production, discounted by the
   level-of-competition jump (fitted: mid-major → high-major production carries at ~x%)
   [HYPOTHESIS] — the Torvik player pages give per-player BPR/OE with team context.
4. **Recruiting**: 247/On3 composite for freshmen, mapped to expected minutes × impact (small,
   noisy; fitted or 0).
5. **Coach change** flag (new scheme → wider prior variance, not a mean shift).
6. **Market-implied**: devigged season win totals / conference futures and the **first slate's
   closes** — the market's prior, which L1 says is the best single number. Opening `w` for
   the market in November is expected to be *higher* than in February and is fitted per month.

The six inputs are combined by `features/prior.py` into two numbers per team — an opening
**mean** and an opening **variance** — and nothing else touches the Kalman state:

* **Roster Continuity Index** (returning % of minutes, possessions used, and defensive
  production — box plus-minus / block / steal rates — from Torvik player pages). High continuity
  → last season's rating carries at the fitted r and the opening variance is small; low
  continuity → the mean leans on inputs 3–6 and the variance opens.
* **Transfer carry-over by level jump**: an incoming player's prior-level production is mapped
  with a **fitted coefficient per (from-tier → to-tier)** — the hypothesis to test is ~65% for
  mid-major → high-major and >100% for the reverse [HYPOTHESIS]; fitted on 2018–2025 portal
  movers' before/after Torvik player ratings, never set by hand.
* **NIL as a market proxy**: dollar figures are opaque and unverified, so NIL talent aggregation
  enters only through **preseason futures** (devigged win totals, title odds, D2b) and the
  **opening-night closes**, which already price it. This is the "market-implied" input and it is
  expected to carry the largest November weight.
* **Coaching change = variance, not mean**: a new coach does not predict better or worse, it
  predicts *schematic change*. The engine (a) multiplies the opening variance by a fitted
  `coach_change_var_mult` so early games update that team faster, and (b) **overrides the pace
  prior with the coach's career pace at previous schools** (Torvik has every coach-season), so
  a slow→fast change does not poison early totals. Both effects fitted on 2010–2025 coach
  changes.
* Recruiting and last-season regression as above; one source per quantity, so an absence,
  transfer or coach cannot be counted twice.

**Hand-off protocol.** The prior must not linger into winter. Hypothesis to fit: games 1–5 the
prior carries ~75% of the team rating, games 6–10 it inverts, **from ~game 12 it is retired
(prior weight → 0)** and the opponent-adjusted efficiency stands alone. In the Kalman spine this
is the opening variance plus a per-month `drift`, so the schedule is an *output* of the fit on
2010–2025, not a constant — if the fit says game 9 or game 15, that is what ships, and the
number is printed in the docstring. The automated audit tracks the **hand-off decay curve** every
week in November–December (prior-only vs rating-only vs market Brier by game number, §4.4) and
flags a prior that is still winning past the fitted retirement point, or losing before it. The
market anchor `w` (§3.7) is a separate, later blend and is *not* the prior: retiring the prior
does not retire the market.

### 3.3 Player layer, regression, minutes

* **Player posteriors** (EB-shrunk per stat, `k` fitted by split-half reliability on D12/D4
  player logs): usage, eFG%, 3P%, FT%, OR/DR%, AST%, TO%, BLK/STL%, fouls/40, minutes.
  Memory rule: *each metric at its own stabilization window* — 3P% is the slowest (many
  hundreds of attempts), FT rate / OR% / usage the fastest.
* **Regression filter** (luck vs skill, the MLB regression report in basketball terms): for
  every team and rotation player, the gap between actual and expected — 3P% vs the shooter's
  shrunk career/season posterior, opponents' 3P% vs league (defensive 3P% is mostly noise — the
  single biggest "luck" lever in CBB [HYPOTHESIS to confirm on D4]), FT% vs posterior, close-game
  record vs point differential (KenPom "luck"), turnover luck. Output: `regression_gap` with
  reliability, printed on the card and the sheet; it enters the **model** only through the
  posteriors (never a second time as an adjustment).
* **Minutes model** = `nba_engine/models/minutes.py` with CBB parameters: 200 team minutes, no
  OT in regulation history, **foul-out rule (5 fouls)** → foul-rate term matters more than in
  the NBA, blowout pulls more common (starters sit the last 8 minutes in 25-pt games — fit the
  curve), freshmen minute growth over the season (fitted trend). Next-man-up from box history
  where a regular has sat before; league-average positional profile otherwise.
* **Team strength for the game** = Σ(projected minutes × player impact) + shrunk team residual
  (NBA §4.3). Injuries move the number **only** through this rebuild. For teams with no player
  data depth (low-majors), the team rating stands alone and the row is flagged
  `lineup_unmodelled` so the audit can show whether those games deserve a different `w`.

### 3.4 Injuries, availability, late news

| Signal | Source | Use |
|---|---|---|
| Out / doubtful / questionable / GTD with note | D7 RotoWire, ESPN game notes | `P(plays)` by status, fitted from status→box outcome on the forward archive (no history exists; the NBA table — Probable 92%, Questionable 74% — is the prior, labelled as such) |
| Starter confirmed (starting lineups ~T-30 on ESPN/box) | D6 | pre-tip pass flips `starter_unknown` off |
| Suspension / eligibility / transfer-waiver / "personal reasons" | D7, D9 | out until seen in a box |
| Mid-season transfer / redshirt decision | D9 | roster edit |

**Mid-major information gap.** For teams outside the top ~8 conferences injury news rarely reaches
RotoWire; it surfaces on beat-writer X accounts, local-paper RSS, or only at warm-ups. So
`alarm.py` also ingests a **curated beat-writer / local RSS list** (`data/beat_feeds.yaml`, one
entry per team, grown over the season) and the pre-tip starting-lineup read (T-30) is the
final word. `starter_unknown` is **strictly enforced** below the high-major tier: a game whose
projected top-7 minutes diverge from the pre-tip lineup, or whose status is unresolved at write
time, is passed (gate stamped, graded as the counterfactual like every gate).

Questionable is priced as a mixture (`P(plays)·p_with + (1−P(plays))·p_without`), both
lineups rebuilt. **News alarm** ported from `nba_engine/alarm.py`: every watch tick compares the
board and the injury feed; an alerted game is re-captured and stays `pending` until the line
settles; the card may not recommend a pending game. Where the first news is the line itself
(mid-majors), the alarm is the only source — and the audit grades what the alarm caught vs what
the box later showed.

### 3.5 Travel, fatigue, schedule (no weather)

Each term is **measured against the closing line on D4/D5 2008–2025 first** and enters at the
fitted value or 0 with the number in the docstring (L15):

* Rest days / rest differential; **3 games in 6 days** (conference week + Saturday); multi-team
  events (MTEs: Maui, Battle 4 Atlantis — 3 games in 3 days on a neutral floor; `mte` flag);
  conference-tournament day-2/day-3 games (the "fatigue vs momentum" question is answerable
  here: thousands of rows); NCAAT first-weekend turnaround (Thu→Sat).
* Travel miles since last game and time zones crossed, from D13; **altitude** (the Mountain
  West has four 4,500 ft+ venues; measure separately home-team-at-altitude vs visitor-from-sea-
  level).
* Exam-week / semester-break layoffs (10–20 days idle in December) — rust vs rest, fitted.
* Road-trip game number (mid-majors' November "buy games": 3–4 straight away at high-majors).
* Neutral-site "home-ish" venue (a Big East team at MSG; an SEC team in Nashville) — distance of
  each team to the venue as a continuous HCA term.

NBA measured that B2Bs move *availability*, not minutes; CBB has no B2Bs outside MTEs and
tournaments, so the hypothesis set is different and all of it is fitted, none assumed.

### 3.6 Game distribution and the 1H model

`possessions × efficiencies → bivariate (home, away) points`; **dispersion and correlation are
fitted to the market's own ML↔spread↔total relation** (CFB #382 lesson: a flat SD handed every
dog value). CBB specifics the fit must carry: higher variance for 3PA-heavy teams, lower for
slow/FT-heavy teams; late-game foul/free-throw inflation that fattens the Over tail in
close games and the favourite's cover probability at −7..−9; OT ~5% of games.

**The under-2-minute foul game is modelled explicitly, not as pace variance.** In a close game
the trailing team fouls intentionally; the last two minutes add free throws (and clock stoppage)
that have nothing to do with the first 38 minutes' pace. The distribution is therefore a
**mixture**: `points = P(first 38 min | pace, efficiencies) + P(end-game | margin at 38:00,
FT rates of the leading team, trailing team's foul rate)`, with the end-game component fitted
on D4/PBP by margin bucket at the 2:00 mark (expected: +3 to +6 points in games within 6, ~0
in blowouts [HYPOTHESIS]). Without it the model's totals lean Under in close games and Over in
blowouts — the same kind of artificial lean the NHL audit caught (Under in 35/41 games). The
audit's totals-lean table is cut by projected closeness so a residual shows up on its own.

**1H** from a separate fitted half distribution: 1H pace and efficiency shift vs 2H (CBB 2H has
more fouls/FTs and timeouts; expected 2H > 1H by a few points — measure on D6 half scores
2015–2025), plus 1H minutes from the rotation model. 1H ATS/ML/total come off that distribution
so they are consistent with the full game.

### 3.7 Market anchor, tiers, gates

* Final probability `p = w·p_model + (1−w)·p_fair_close`, `w` fitted **per market × month ×
  market-quality bucket** (November mid-major ≠ February Big Ten). Power de-vig by default (NFL
  study: proportional de-vig creates the favourite–longshot slope).
* Tiers copied from `cfb_engine/market/tiers.py` (rank on edge, `min_ev=0`, `max_edge` reads
  big disagreement as model error). Per-market price bands. One buy per game-direction; a team
  total Over and the game Over are one opinion — dedupe.
* Gates, every one a named `pass_gate`: `starter_unknown` (ML/ATS/1H when a rotation player's
  status is unresolved at write time), `one_way_quote`, `settlement_unverified` (team totals
  and 1H push rules per book, `book_rules.py` with 60-day expiry — NHL pattern), `ml_sharp`
  (VSiN handle ≥ tickets, ML only, neutral when absent — `mlsharp.py`), `drift` (measured not
  vetoing, CFB stance), `pending` (news alarm), `thin_market` (fewer than N books quoting — a
  gate we expect to *retire* if its refused rows beat its admitted rows; that's the point of
  grading refusals).

---

## 4. Audit function

Ported from NHL (`nhl_engine/audit/ledger_report.py`, PR #468) and CFB probation, with the CBB
cuts of §2. One command, `cbb-engine audit`, run in the 03:00 pass and again before the card:

1. **Grade yesterday** from ESPN finals + half scores + boxes (DNP → void for props), write once
   to `cbb/ledger/graded_<date>.csv.gz` on engine-state; each priced pass is its own immutable
   `priced_<stamp>_<tag>.csv.gz`.
2. **Stamp closes**: last snapshot before tip with ESPN `status=pre`; a close after tip is
   dropped and counted (CFB #377 found 66 in-play closes).
3. **Rebuild the whole-ledger report** (md + PDF, attached to the card email like NHL): slate
   W-L-P/units lead → ledger-to-date buys with bootstrap CI (ROI "not callable" < 100 buys) →
   exact sign test on CLV → one-side-per-game model-vs-market Brier per market → totals
   directional lean (the NHL Under lean, 27/32 + 8/9, was caught this way — CBB's version is
   the pace prior being wrong in November) → availability-read accuracy (named starters vs box,
   by status) → **gate false-negative table** (admitted vs refused record/ROI/CLV per gate) →
   §2 cuts (power/mid, home/away/neutral, month, market-quality) → probation status with
   required-n remaining → integrity block → calibration curve.
4. **Prior hand-off audit** (CBB-specific): Brier of prior-only vs market by game number 1…15,
   printed each week in November/December.
5. **Baselines on the same games**: favourite, home, always-Over, Torvik's pick, the sheet's
   call (§6) — every lean list carries its naive baseline (MLB totals sheet lesson: SUM 118-102
   vs always-Over 133-99).
6. **Probation** (`probation.py` verbatim): n ≥ 100, > 1 s.e., both halves agree; markets and
   screens symmetric; monthly review retires/reopens, not a hand edit.
7. Weekly audit email (Monday) with the regression report and the sheet ledger audit (§6).

Everything is CSV/gzipped JSON on `engine-state` under `cbb/`, no database, same as the others.

---

## 5. Outputs, delivery, schedule

**Workbook `PayoffPitch_CBB_<date>.xlsx`** (tabs as NBA §8): Buys · Games (every game × 9
markets: model line, market line, model %, fair %, edge, gates) · Props (research_only) ·
Regression · Injuries & Schedule (status + timestamp, minutes impact, rest, trip game #, miles,
tz, altitude, MTE/tournament flag) · Line Moves · Fades (refused rows with their gate) · Audit ·
**Worksheet** and **Totals Sheet** (§6) · Legend.

**Daily slate PDF** (`report.py`, the NHL PDF pattern): front page = slate summary, buys by
tier, top gates fired, prior-vs-market status in November; per game (tip order, grouped by
window) = `AWAY @ HOME — tip ET, venue (neutral?), TV`, market line under the title (spread,
total, ML, 1H), records (overall, home/road/neutral, ATS, O/U, last 5 / last 5 home / last 5
away, streak), **tempo-free ranks** (AdjOE, AdjDE, AdjT, four factors for/against, 3PA rate),
key players with projected minutes and regression gap, injuries with source + timestamp, travel
/ rest line, model shape (margin, total, 1H), the game's rows table with gates, line movement
since open, Torvik's number beside ours (labelled), pod/outside read box reserved for Phase 5.
Big slates (80+ games) get a **one-line-per-game board** first and full blocks only for games
with a buy or a `watch` flag — otherwise the PDF is 150 pages.

**Audio**: MP3 narration of the card with edge-tts (MLB/NHL pattern, no AI key), ~4–6 minutes:
buys, the three biggest injury/line stories, audit lead. Research-only props are not narrated.

**Email**: one message at the card pass with PDF + xlsx + txt + md + MP3 + the newest ledger
audit md/PDF (NHL `email.send_files` pattern), via the existing SMTP config.

**Schedule (Mac launchd, `scripts/cbb/macos/`, Desktop/engines shortcuts, idempotent installer
like NHL):**

| Time (ET) | Job |
|---|---|
| every 30 min 09:00–23:30 on game days | `capture` (board + 1H + team totals + props), idempotent |
| every 5 min 11:00–23:55 | `watch` news alarm (board + RotoWire + ESPN notes) |
| 03:00 | `audit` (grade, closes, ledger report) |
| 11:15 | preview card (no buys) — weekday noon tips exist (MTEs, conference tournaments) |
| **11:30** | `card --email` **on any day with a tip before 16:00 ET** (MTEs, holiday and weekend blocks, conference tournaments) — the ESPN schedule decides, not a calendar; every game on the slate is priced in this card |
| **16:30** | `audit` + `card --email` on evening-only slates; on early-tip days a **re-price of the evening games only** (write-once ledger: a game already written at 11:30 is not re-written, it gets a lead-time log entry). **The default times are measured in Phase 1 by replaying the card at T-6h/T-3h/T-90/T-60/T-30 against archived closes** (NBA §10) and moved if CLV says so |
| per game, T-60 → T-5 | `close` passes (pre-tip only) + starting-lineup read at T-30 |

Tournament weeks (conference tourneys, NCAAT: 16 games/day from noon) are the early-tip case
above: preview at 09:00, card at 11:30, automatically from the ESPN schedule.

---

## 6. The CBB worksheet and totals sheet (the MLB hand sheets, translated)

Both MLB sheets are *records beside the card, not prices* — "nothing here touches a price" — and
both proved it: through 2026-10-06 the totals sheet's SUM sign went 118-102-11 (54%, 95% 47–60%)
while always-Over on the same games went 133-99 (57%); the daily worksheet's favoured side went
32-15 (68%) against a 55.7% market rate on 47 games, unproven at n<30 per band [MEASURED,
`docs/mlb_audits/2026/`]. The CBB versions keep every structural rule (per-metric +2/−2/+1 or
decile scoring at each metric's own window, a weighted total, a gap ranking, a ledger with the
best board price and VSiN splits at write time, an Audit tab by band vs the market's implied
rate, a Legend tab, naive baselines on the same games, "incomplete" rows never graded) and swap
the baseball metrics for their basketball analogues.

### 6.1 Daily worksheet (`scripts/cbb_worksheet.py` → `worksheet_<date>.xlsx`)

MLB scores **offense vs the starter's hand × 1 + bullpen × 1.5 + starter × 1**. Basketball has no
starter/bullpen split; the analogous three blocks are **offense**, **defense**, **rotation /
availability**, each a ranking table rebuilt every morning from Torvik (D3/D4) and the player
layer:

| Block | Metrics (each at its own window; decile among D-I, rescaled to the ±2 range like `rescale_batting`) | MLB analogue |
|---|---|---|
| **Offense** (vs the opponent's defensive profile — the "vs hand" split becomes **vs man / vs zone-heavy** and **vs pressing / non-pressing** where PBP tags exist, else vs opponent AdjDE tier) | AdjOE, eFG%, 3PA rate × 3P% (shrunk), OR%, TO%, FT rate, FT%, assist rate, transition share, bench scoring share | bat-vs-hand (ISO, Barrel%, xwOBA…) |
| **Defense** | AdjDE, opp eFG%, opp 3PA rate (what the D *allows*, which repeats; opp 3P% does not), DR%, forced TO%, block%, FT rate allowed, fouls/40 | bullpen |
| **Rotation / availability** | Projected minutes of the top-7 vs season norm (missing starter = −2), foul-trouble risk (fouls/40 of key bigs), experience (minutes-weighted class year), bench depth (minutes share of 8–10), days rest, travel/altitude flag, home/neutral | starter PTS |

**Style mismatch outranks raw tiers.** In baseball the platoon split carries the variance; in
basketball the analogous lever is the **stylistic clash**, so the sheet has a fourth block,
**Matchup**, scored only from pairwise features: pace clash (fast offence vs slow defence and
who dictates — fitted from D4 game tempo vs both teams' season AdjT), 3PA-heavy offence vs
defence that allows/denies 3PA rate, press/turnover-forcing defence vs ball-handling (TO%),
zone-heavy defence vs offence's zone efficiency where PBP/Synergy-style tags exist (else opp
AdjDE tier), offensive rebounding vs defensive rebounding rate, FT-rate offence vs foul-prone
defence, size (minutes-weighted height) mismatch. Its weight `w_match` is fitted alongside the
others and the Audit tab reports the matchup block's **own** record so its claimed primacy is
checked, not assumed.

Weighted total per team = off × 1 + def × 1 + rotation × **w_rot** + matchup × **w_match**, with
the weights fitted on the 2008–2025 game logs (MLB's 1 / 1.5 / 1 came from the pitcher's share of a game; the CBB share
is measured, versioned as `WEIGHTS_VERSION`, and the Audit tab only pools rows on the current
weights, exactly as MLB does). `gap = away − home`; right-hand block ranks the slate by gap.

Ledger (`cbb/audit/worksheet_ledger.csv`): date, game, event_id/game_id, the three block
scores per side, weighted totals, gap, fav, `complete` (false when a key player's status is
unresolved → never graded), best ML / spread price on the board, market implied %, VSiN handle
/ tickets at DK and Circa **read only when the page's own date is the sheet's date** (PR #431's
guard), Torvik's spread. Next morning: final, fav W/L, ATS. Audit tab: fav record by gap band
vs market implied rate, flat 1u ML, fav ATS, **required n to confirm at 95%/80% power** printed
("129 priced games" is the MLB line).

### 6.2 Totals sheet (`scripts/cbb_totals_sheet.py` → `totals_sheet_<date>.xlsx`)

MLB: every column a signed score (+ leans Over, − Under), the row sum is the read; bands live in
`mlb_engine/output/totals_sheet.py`; the audit grades SUM sign, |SUM| bands, top-3/bottom-3,
engine-agrees/disagrees slices against always-Over. CBB columns:

| Column | Score (bands fitted on D4 2008–2025 and versioned, e.g. `cbb-centred-2026.11`) | MLB analogue |
|---|---|---|
| **Pace** — both teams' AdjT vs league, and the *pace mismatch* (fast vs slow games land near the slow team — fitted) | ± by tercile | BaseRuns / park |
| **Shooting profile** — combined 3PA rate (variance) and shrunk 3P% | ± | barrel rate / ISO |
| **Free-throw rate** for and against (fouls/40 both teams) | + for high | — (new) |
| **Turnover rate** both offences | − for high (TOs kill possessions' points) | K-BB% / CSW% |
| **Offensive rebounding** both | + | — |
| **Defensive efficiency** both AdjDE | − for two top-50 defences | SIERA / xERA of both arms |
| **Rest / travel / altitude** (altitude +, 3-in-6 −, long layoff −) | ± | weather + roof (−1) |
| **Officials** — crew foul calls per game vs league, where the assignment is known pre-tip (rare) | ± | home-plate umpire runs/game |
| **Book splits** — DK & Circa total handle − tickets, signed by side | ± | same |
| **Market tier** — number of books / line dispersion (thin = louder sheet, we expect) | informational | — |
| **Engine side** — the model's P(over) at the sheet's line and Torvik's total | not summed; the "engine agrees" slice | engine_p_over |

Ledger (`cbb/audit/totals_ledger.csv`) and `cbb_totals_audit`: the exact MLB slice table (SUM
sign, always-Over same games, SUM says Over/Under, top-3/bottom-3, |SUM| bands, watch bands,
engine's side vs sheet, engine agrees/disagrees), each with a 95% range and a verdict that is
"unproven" until the whole range clears 52.4% on 30+ decided games. The ledger needs *the line
on record at write time* — a row with no line is never graded (the MLB rule).

Both sheets are emailed in the morning package and sit as tabs in the workbook; neither feeds a
tier, gate or stake. If the Audit tab ever clears probation, the finding becomes a **candidate
screen** in the engine (registered, graded, then promoted) — the sheet itself never becomes a
bet.

---

## 7. Architecture — `cbb_engine/` mirrors `nba_engine/` + `nhl_engine/`

```
cbb_engine/
  config.py        CBBE_* env (data dir, state sync, odds TTL, books DK/BetMGM, thresholds)
  schemas.py       Game, TeamRating, PlayerPosterior, Quote, Recommendation, LedgerRow
  state.py         engine-state plumbing under cbb/ (mlb_engine.state reuse, as CFB/NHL)
  data/            oddsapi.py (basketball_ncaab, 1H/team-total/props keys), capture.py,
                   futures_scraper.py (October win totals / conference futures fallback),
                   beat_feeds.py (+ beat_feeds.yaml: per-team beat-writer / local RSS),
                   espn.py (schedule/finals/halves/box/neutral), torvik.py (D3/D4/D5 + aliases),
                   rotowire.py, vsin_splits.py, sref.py (cached, Mac), arenas.py, teamnames.py,
                   book_rules.py
  features/        strength.py (Kalman tempo-free), prior.py (early-season blend),
                   players.py (EB posteriors), minutes.py, regression.py, context.py (rest/travel/
                   altitude/MTE), availability.py (status log + P(plays))
  models/          distribution.py (bivariate points fitted to market), half.py, props.py
  market/          ev.py, tiers.py, priceband.py, drift.py, mlsharp.py, keys.py, blend.py (w)
  alarm.py         news alarm (nba_engine.alarm port)
  audit/           ledger.py, settle.py, clv.py, probation.py, scorecard.py, ledger_report.py,
                   worksheet_ledger.py, totals_ledger.py
  output/          excel.py, report.py (PDF), card.py (txt/md), audio.py (edge-tts), email.py,
                   worksheet.py, totals_sheet.py
  pipeline.py, cli.py (capture | watch | close | card | audit | worksheet | totals | replay)
scripts/cbb/       studies (prior fit, reliability windows, schedule-vs-close, half shape,
                   dispersion fit, send-time replay), macos/ installers + launchd
tests/cbb/
```

Shared code rule unchanged (NHL plan §4, decided): `engine_common/` pure primitives only
(odds, devig, EV, probation statistics, render helpers); nothing lifted out of a live engine;
CBB builds its own `tiers/probation/ledger` on the primitives from the newest templates
(NBA/NHL). `lineshop` gains `--sport cbb` (`basketball_ncaab`).

---

## 8. Build order (Devin sessions; season tips ~Nov 3)

| Phase | Work | Sessions |
|---|---|---|
| **0 — capture first** | scaffold, config, team alias table (364 teams × Torvik/ESPN/books), Odds API capture (game, 1H, team totals, props), **futures fallback capture in October**, ESPN schedule/finals/halves, RotoWire feed + beat-feed list seed, VSiN CBB, engine-state `cbb/`, Mac shortcuts + launchd, `book_rules` for 1H/team-total pushes, sharp-reference book check (novig / Pinnacle-eu). **Must be live before the first slate** — the archive cannot be back-filled. Verify D11/D12 on the Mac. | 1 |
| **1 — data + studies** | **Buy 2024-25 game-line closes first** (D2, ~40–60k credits; 1H deferred) — the prior fit needs the close curve as its benchmark; Torvik 2008–2025 loaders with as-of slicing; D5 column mapping; reliability windows (`k` table); Kalman fit + walk-forward vs Torvik and vs the 2024-25 closes; **early-season prior fit** by game number; schedule/travel/altitude/MTE terms vs the close; half-shape study; dispersion fit incl. the end-game foul component; send-time replay; style-mismatch block fit. Deliverable: numbers in docstrings, `config.py` defaults quoting them, `cbb/params/*.json` versioned fits. | 2–3 |
| **2 — engine + card** | players/minutes/lineup rebuild, distribution + 1H, blend `w`, tiers/gates/alarm, ledger, PDF/xlsx/txt/md/MP3, email, schedule. Walk-forward backtest vs Torvik (and vs closes if bought). | 2–3 |
| **3 — sheets + props + audit report** | worksheet + totals sheet + their ledgers/audits (§6); props research_only; `ledger_report` port with CBB cuts; weekly audit email. | 1–2 |
| **4 — live probation** | Nov–Dec paper/live; monthly probation review; prior hand-off audit weekly; nothing new turns on without the three-test verdict. | ongoing |
| **5 — extensions** | 2H/intermission markets, podcast/outside read box, alt lines via lineshop, tournament-mode card. | later |

~7–9 sessions plus the calendar the probation bar demands — the same shape as NHL.

---

## 9. Success criteria

1. Every buy has a pre-tip close and a CLV; the markets we keep after 100+ buys have positive
   CLV with both halves agreeing.
2. Model-vs-close Brier per market and per §2 cut on every audit; markets the close wins by a
   wide margin are anchored (`w→0`) or shut by probation, not by hand.
3. The prior hand-off is visible: by week 5 the team's own games dominate and prior-vs-market
   Brier converges.
4. Worksheet and totals-sheet ledgers graded daily with naive baselines on the same games.
5. No unmeasured coefficient ships; `ruff`, `mypy`, `pytest -q` green; `cbb-engine card` runs
   free with `CBBE_DATA_DIR` scratch, `CBBE_STATE_SYNC=0`, long odds TTL.

---

## 10. Decisions (taken 2026-10-01 after review; two modified)

1. **Historical closes — decided: buy 2024-25 game-line closes in Phase 1, defer 1H.** The prior
   fit without a close curve would be volatility with no benchmark. Blocker: the key in this
   VM shows 497 credits; the Mac key / NBA balance must be confirmed first.
2. **KenPom — decided: offline verification only, never in the live pipeline.** Torvik is the
   spine; a Torvik error halts the card (§1 fail-safe).
3. **Coverage — decided: all 364 D-I teams ingested and modelled**, segmented by
   `market_quality`; the `thin_market` and `starter_unknown` gates protect the ledger and are
   graded like every gate.
4. **Books — modified.** Buys and CLV against the best of DK/BetMGM (the accounts held). A
   sharp reference is added for a *second* CLV column **only from a book the Odds API actually
   carries**: today that is `novig` (no-vig exchange) in the US regions; Circa and Bookmaker are
   not in the feed [VERIFIED], and Pinnacle (`regions=eu`) is verified in Phase 0. Account
   limits are real but are a staking question, not a model one; the ledger records the book of
   each quote so a per-book CLV table exists from day one.
5. **Card time — modified.** 11:30 ET card on any day with a tip before 16:00 ET (MTEs,
   holidays, weekends, tournaments — decided by the ESPN schedule), 16:30 ET on evening-only
   slates with an evening-only re-price on early days; the Phase 1 send-time replay can move
   either. A fixed 11:30 every day would price Big Monday 21:00 tips nine hours out with no
   lineup information, which the NHL and NBA ledgers say is the expensive end.
6. **Mac daemons** for Sports-Reference and the JS-guarded Torvik pages — as `nba_api`.

Still needed from you: confirmation of which Odds API key holds the credit balance, and the
go-ahead to spend ~40–60k credits on the 2024-25 closes.
