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
the consensus devigged median (a **consensus** that excludes the books we bet) anchors the
location, two hierarchical weights `w_m`, `w_t` shift the margin/total means toward a rating
mean, and one calibrated simulation prices every market (§3.6–3.7); a **market-only channel is
judged first**, and every filter earns its coefficient against the close or ships at 0 (§3.7).
**Year-1 scope (ships / shadow / deferred) is the one-page sheet `year1_scope.md`; this
document is the reasoning behind it.**

> **Source of truth.** Where an older sentence in this document disagrees with §3.7, §4.6, §8 or
> §10.7–10.20, the newer section wins. The 2026-10-12 consistency pass removed the known
> contradictions (`p_fair_close`, per-market `w`, the 100-buy ROI bar, Torvik halt, limits as
> "staking only", daily-graded sheets); report any survivor as a doc bug.

What makes college basketball different, and worth an engine at all:

* **~364 D-I teams, ~5,900 games/season** (ESPN scoreboard 2026-03-01: 23 D-I games on one
  Sunday; a weekday in January has 60–100) [VERIFIED volume]. Volume reaches n ≥ 100 per
  pre-registered hypothesis within weeks **if the gate funnel lets it** (simulated in Phase 0,
  §8); the bar is consensus-CLV, never ROI (§4.6).
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
| D9 | Sports-Reference CBB | all | Cross-check of ratings, rosters, preseason AP poll, returning players | [VERIFIED] but rate-limited (429 on second hit) — nightly, cached, Mac only, polite rate (≥3 s/req, one pass); **never in the live path, no proxy rotation** (ToS; a ban costs the cross-check, a block on the live path would cost the card) |
| D10 | **KenPom** | 2002 → | Offline cross-check only (preseason AdjEM, luck) | **paid, 403 unauthenticated** [VERIFIED]; **never in the live pipeline** — manual CSV drop-in for offline verification, the card does not depend on it (decided §10.2) |
| D11 | Torvik preseason projections / returning-minutes pages | yearly | Early-season prior inputs: returning minutes %, transfers in/out, recruiting | page exists; `trank.php` sits behind a JS browser check [VERIFIED] → fetch via the Mac Chrome/Playwright route or the per-season CSV mirrors; mark **unverified until it returns data** |
| D12 | hoopR / sportsdataverse MBB data (PBP, box, lineups) | 2002 → | Possession-level PBP, player box logs for the minutes model and props | GitHub API rate-limited from the VM today — **unverified**; Phase 0 verifies on the Mac |
| D13 | Arena table (lat/lon, altitude, tz) | static | Travel miles, time zones, altitude (Wyoming, Air Force, Colorado State, Utah State, BYU, Denver, New Mexico…) | built once; ESPN venue ids |
| D14 | Official results = ESPN finals + half scores | — | Grading of game, 1H, team totals; props from the ESPN box | same as NBA |

**Fail-safe rule.** Torvik (D3/D4) is the rating spine of the **model channel only**: a stale (>1 day) or partial
Torvik file **stops the model channel** (no blend, no shape rungs that read it — the card
continues on the consensus median and the market-only channel, flagged `torvik_stale`); it does
not halt capture or the card. The old whole-card halt is superseded. The card **halts with an error email**
if the fetch errors, if the newest game row is **more than 1 day older than the last completed
D-I slate** (in-season feed drop), or if the game count for yesterday is below 60% of ESPN's
final count (partial file). ESPN schedule failure halts the same way. The capture job never
halts (prices are the one thing that cannot be re-fetched).

**Capture redundancy (Mac is the runner).** Every engine lives on the Mac and the ledger must be
auditable there, so it stays the runner; the risk of a sleeping/rebooting Mac is handled the
way the other engines do it plus two additions: (1) launchd job with `StartInterval` + a
`caffeinate`/wake-for-network setting so capture runs through sleep; (2) every capture pass is
written as a small immutable **Parquet shard to encrypted off-site object storage** (rclone crypt) with a daily manifest + hash committed to `engine-state`, so a second machine can replay from the shards (git carries ledger/params/manifests only, §4.d); (3) the audit's
**capture-gap table** counts missing passes **against the expected interval of each `pass_kind`**
(60/30/15 board by time-to-tip, 10-min featured inside T-3h, close passes) per slate day and marks any game whose
close came from a gap as `close_quality=degraded` (excluded from CLV, counted). A cloud capture
node (small VPS, cron → engine-state push, Mac pulls and does the heavy builds) is **not** a
prerequisite but has a hard trigger: if the November gap table shows **> 2% of expected
passes missing** on any week, the VPS goes up before December. Until then the Mac is a $0
pipeline whose every pass lands in an off-site shard with its manifest hash in git.

Every row carries `source`, `captured_at`, `season`, Odds API `event_id` + ESPN `game_id`
(matchup text is display only — MLB doubleheader lesson, PR #423) and a `team_alias` table
(Torvik names ≠ ESPN names ≠ book names; ~364 teams, built once, tested).

---

## 2. Markets and outputs

Every row carries **`status` ∈ {`live`, `shadow`, `research`}**; only `live` rows can be
bought and only `live` `bet_of_record` rows enter the promotion test. Year-1 status:

| Market key | Model output | Settlement | Phase | Year-1 status |
|---|---|---|---|---|
| `game_ml` | P(home win) incl. OT | ESPN final | 2 | research |
| `game_ats` | P(cover) at every quoted line (alt lines via lineshop) | incl. OT | 2 | research (main) / shadow (alts) |
| `game_total` | P(over) at every line | incl. OT | 2 | **live** |
| `team_total_home` / `team_total_away` | P(over) team points | incl. OT | 2 | shadow |
| `h1_ml` / `h1_ats` / `h1_total` | first-half distribution (fitted, not a half-share — NBA measured +0.77 pts/team 1H vs 2H; CBB has its own number to measure, and the under-10-minute "foul game" inflates 2H) | ESPN half scores | 2 | shadow |
| `h1_team_total` | where quoted | half scores | 3 | shadow |
| `h2_*` | **not pregame** (quoted at half) → Phase 5 only, needs an "intermission close" | — | 5 | — |
| `pl_pts`, `pl_reb`, `pl_ast`, `pl_3pm`, `pl_pra` | `research_only` from day one, like NFL/NHL props: no historical CBB prop prices exist, archive first | box, `dnp_void` | 3 |

Every row carries `pass_gate` (which named gate refused it) and `tier`; refusals are priced and
graded as the counterfactual (L6/NBA-L3). `MAX_BUY_ODDS` and the EV floor are set **per market × market-quality bucket**. Year 1 buys
game totals only, so the **live band is a totals band around the juice actually posted**
(e.g. −125…+105 per bucket, fitted on 2024-25 closes); a dog-odds cap is a research-row
setting (sides are `research`), where the CFB +200 default and the mid-major +300…+450
hypothesis live until a side market earns promotion.

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

### 3.1 Team strength — tempo-free, opponent-adjusted, two estimators raced

Spine as `nba_engine/models/ratings.py` (mean + variance per team, sequential no-look-ahead
replay, league level and HCA as running means), on **per-100-possession** observations:

```
home pts/100 = league + hca·(1 − neutral) + off(home) + def(away)      # additive baseline rung
away pts/100 = league − hca·(1 − neutral) + off(away) + def(home)
possessions  ~ Poss(mismatch, expected closeness, pace(home), pace(away))  # a distribution, §3.6
```

* **Observations are lineup-adjusted before they update the rating.** The raw team efficiency
  of a game played without a star is converted to a full-strength-equivalent efficiency through
  the player layer (§3.3, high-majors first; identity where the player layer is `unmodelled`),
  so the rating is not pulled down by the absence *and* penalised again by the lineup mixture
  at pricing time. Observations are **weighted by possession count** (a 58-possession game is
  noisier than a 78).
* **Two estimators, raced, not stacked.** (a) the sequential Kalman; (b) a **joint
  time-decayed ridge** refit on the season-to-date each morning (the Torvik/KenPom family — it
  re-solves every opponent adjustment at once, which the filter cannot). Both are rungs in the
  ladder (§3.7): whichever wins OOS against the close on MAE / partial correlation ships, or a
  fitted blend if it beats both. The filter is kept regardless for its variance, which must be
  shown to **widen the predictive distribution** in November and under lineup uncertainty
  (PIT check, §3.6), or the variance is re-fitted.
* **Functional-form tests, pre-registered in the ladder:** additive vs **multiplicative**
  efficiency (elite offense × 340th defence), **capped per-game margin credit** (starters sit —
  diminishing returns past a fitted margin), and garbage-time down-weighting (Torvik/KenPom
  convention, cutoff fitted). Each is a rung; none ships on plausibility.
* Fitted on D4 2008–2025 (recency-weighted, §3.2) by walk-forward replay; graded against
  Torvik's projection (D5) and, where D2 is bought, the close. Report MAE on margin and total,
  and **the partial correlation of our rating with the final after the close** — CFB's
  efficiency layer scored −0.001 there; if CBB's does the same, `w` goes to ~0 on the main line
  and the model owns only the shape (§3.6), said plainly.
* `neutral` from ESPN's flag; HCA is **fitted per venue tier** with team-specific deviations
  shrunk to the tier [HYPOTHESIS: per-team HCA repeats; test split-half].
* Four-factor sub-ratings (eFG%, TO%, OR%, FT rate, 3PA rate, 3P% for and against) are kept as
  **features**, each shrunk at its own stabilization window (§3.3) and **hierarchically toward
  the conference / tier mean** (per-team samples are small), and feed the distribution shape
  (variance, 3-point dependence, foul tendency) rather than the mean.

### 3.2 Early-season prior (the hard part; where the preseason market is beatable *if* anywhere)

The first ~8 games are ~25% of the pre-conference schedule and the thinnest-priced. Each team's
opening rating is a weighted blend, every weight **fitted recency-weighted on 2010–2025 (inputs 1–5; input 6 is HAND-SET in year 1, §3.7) preseason-vs-March
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
* **NIL as a market proxy, with a public-hype term**: dollar figures are opaque and unverified, so NIL talent aggregation
  enters only through **preseason futures** (devigged win totals, title odds, D2b) and the
  **opening-night closes**, which already price it. CBB futures are thin and
  public-skewed (blue bloods overbought), so the futures weight inside the prior is **allowed to
  shrink with a fitted public-interest proxy** (preseason AP votes, futures handle share where
  VSiN shows it, brand tier) — a fitted interaction, not a hand-cap, tested on whether
  blue-blood opening lines were in fact biased 2010–2025 [HYPOTHESIS]; low-profile teams keep
  the full market weight. This is the "market-implied" input and it is
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

**Recency weighting of every Phase 1 fit.** The portal (2021 one-time transfer, 2024 unlimited),
NIL and the 2023+ officiating points of emphasis changed turnover, carry-over and scoring
distributions, so the 2008–2025 window is not weighted flat: fits (continuity index, transfer
carry-over, hand-off decay, foul-game component, HCA by month) use a **time-decay loss weight**
whose half-life is **chosen by holdout** (candidates 3/5/8 seasons; 2022+ ≈ 3× 2010–2015 is
the starting hypothesis, not a constant), and each fit is **re-estimated on 2023–25 alone** as a
sanity split — if the modern-only estimate falls outside the weighted fit's CI, the weighted fit
is rejected and the modern one ships with its wider CI stated.

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
  `lineup_unmodelled`. Those rows get their **own blend weight**: `w_unmodelled = w ×
  m_unmodelled`, `m` fitted per market-quality bucket (starting hypothesis 0.65 — i.e. the
  team-only rating is trusted less and the market more, because low-major November ratings
  carry the highest turnover and no player layer to catch it [HYPOTHESIS]); the audit reports
  the unmodelled rows separately so a false-edge pattern cannot hide inside the aggregate.

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
entry per team, grown over the season). Beat-feed items are run through a **keyword rule set**
("street clothes", "walking boot", "not in warmups", "will not play", "out indefinitely",
"cleared"/"available") that writes an **availability override at T-90** (`projected_minutes
= 0` or status upgrade, source + posted + seen stamped), so the game can be priced on the
lineup before the T-30 ESPN lineup posts move the line — the whole point of the mid-major edge
is pricing before that move. No X/Twitter (no API access) and no LLM in the critical path; an
optional LLM pass may summarise the item for the card, as the NHL podcast does. The pre-tip
starting-lineup read (T-30) remains the final word and grades the keyword overrides (hit /
false alarm per phrase). Overrides are **tri-state**, never a binary zero on ambiguous text:
`CONFIRMED_OUT` ("will not play", "out indefinitely", "ruled out") ⇒ `projected_minutes = 0`;
`AMBIGUOUS` ("walking boot", "game-time decision", "limited", "expected to try") ⇒ the §3.4
`P(plays)` mixture with a phrase-specific probability (starting prior 0.50, fitted from
phrase → box outcome as the archive grows) **and** a tighter price band on that game until the
T-30 lineup resolves it; `CONFIRMED_AVAILABLE` ("cleared", "available", "will start") ⇒ baseline
projection restored. A later item overrides an earlier one; conflicting items within 30 minutes
⇒ `AMBIGUOUS`. `starter_unknown` is **strictly enforced** below the high-major tier: a game whose
projected top-7 minutes diverge from the pre-tip lineup, or whose status is unresolved at write
time, is passed (gate stamped, graded as the counterfactual like every gate).

Questionable in year 1 is a **variance multiplier keyed to P(plays)** on the points-lost
impact (§3.4b); the full both-lineups-rebuilt mixture is Phase 1b, and the multiplier is
checked against the mixture's kurtosis when both exist. **News alarm** ported from `nba_engine/alarm.py`: every watch tick reads
**ESPN and RotoWire only** (no board, no credits); a hit pulls the next featured pass forward,
and the alerted game stays `pending` until the line
settles — **defined, not felt: no consensus move > ½ pt (spread/total) or > 5¢ (ML) across two
consecutive **featured-market passes** (h2h/spreads/totals via `/odds`, cost = markets ×
regions regardless of game count — 3 credits per pass for the whole slate, verified against the
Odds API quota rules 2026-10-12), polled every 10 min inside T-3h; per-event markets stay on
the slower tip-scaled cadence**; `pending` is a graded gate like the
others, so the audit shows what the wait cost. The card may not recommend a pending game. Where the first news is the line itself
(mid-majors), the alarm is the only source — and the audit grades what the alarm caught vs what
the box later showed.

### 3.4b Year-1 availability model

**Year-1 availability model, simplified (review 2026-10-12).** The consensus line already
carries the market's expected availability, so a lineup mixture adds information only relative
to what the market assumed. The shift is applied **after the solve, on the residual**: only the part of the
absence the current consensus has not yet moved (the `pending` rule and the market-reaction
race exist to measure exactly that); feeding points-lost back into the efficiency solve either
cancels out or re-bets an absence the close already contains. Year 1 therefore runs: (a) a **points-lost-per-player** impact for
high-majors only (`lineup_unmodelled` elsewhere), with **Beta priors per status × tier**
(seeded from the NBA numbers, updated daily from status → box outcomes on the forward archive);
(b) **player value fitted from market reactions**, with a design, not a bare regression: an
**event window** around each confirmed-out post (last consensus before → first settled after),
**differenced against the same-window consensus movement of unaffected games** that day, by
player tier; the year-1 report states **how many events the sample actually holds** (expected:
low hundreds for high-majors — tier-level, not per-player, estimates); raced against the
box-score impact **in the report only** — the race moves the median only after its own
pre-registered event count and MDE are met (written in `preregistered.yaml`); until then the
live shift is the un-moved market residual and nothing else; this also measures how much of the news is priced by the time we could act (the
latency study, §8). The full minutes-rebuild / next-man-up machinery is Phase 1b.

### 3.5 Travel, fatigue, schedule (no weather)

**Situational terms are one regularized model, not a sequence of accept/reject tests.** Margin
residuals around the close have SD ≈ 10, so a subgroup mean has SE ≈ 10/√n — ~1.3 pts at n=60
(altitude), 0.6 at n=300, 0.26 at n=1,500 — while plausible effects are 0.3–1 pt; picking 5 of
~15 candidates one at a time is a forking path even with a cap. So all candidate terms enter
**one ridge / horseshoe model on close residuals**, effects estimated on the long
Torvik-centred history and **scaled by the fraction the close already prices** — estimated from
the frequent terms (rest, home, neutral) and carried to rare terms (altitude, MTE) as a **wide
prior, not a point estimate**, since the market prices obscure terms less well than it prices
rest; `preregistered.yaml` publishes a **minimum detectable
effect per term**, and any term whose MDE is far above a plausible effect is research-only
from the start. Terms are tested **on pace and variance first** — the anchored design gives
them a home there (totals, team totals, shape) — and only then on the rating mean, which may
have `w ≈ 0`. The inventory:

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

### 3.5b Home court, conference familiarity, stakes (all fitted against the close)

**Home court.** The market prices ~3.5–4.5 pts of it already, so the engine's job is not the
league number but the *deviation*: HCA is estimated per team (shrunk to the venue-tier mean,
split-half tested for repeatability, §3.1) and allowed a **month term** — the hypothesis is that
it is largest in Nov–Dec with young road rosters and shrinks by February [HYPOTHESIS]. Audit
cuts home / away / neutral already exist; a **home-fav vs home-dog** cut is added because the two
fail differently.

**Conference play (familiarity).** A `conf_game` flag (from Torvik's schedule) enters two
places: (a) the spread model as a **talent-gap compressor** — hypothesis: the fitted
rating-margin coefficient is < 1 in conference play because scheme/personnel familiarity
flattens the gap; (b) the totals model as a variance/mean term. A curated **rivalry list**
(static yaml, ~60 pairs) is a separate flag tested for a home-dog effect; if it does not beat
the close it stays research-only.

**Stakes (what the broadcasters call "games needed").** From Torvik's daily WAB and the NET
quadrant record, each team gets a `stakes` state per game: `lock`, `bubble` (|WAB − bubble
line| ≤ 1.5), `auto_bid_only`, `eliminated`, plus `seed_line_live` in March. Enters the model
only as fitted terms — bubble teams late, eliminated power teams in February, mid-majors whose
only route is the conference tournament — and feeds the worksheet's matchup block as a visible
column regardless. Conference-tournament games already carry the MTE/turnaround terms (§3.5).
None of these ship as weights until the probation rule (§4) passes them; the audit reports
each as "active / research / rejected" with its CI.

### 3.6 Game distribution — anchor the mean, own the shape

**Design rule.** Our five engines say the close owns the main-line mean. So the model channel
does not fight it there: the **consensus main spread and total anchor the location** of the
distribution (the ratings move it only through `w`, §3.7, which may be ~0), and the model
**owns the shape** — variance, skew, key-number mass, margin–total correlation, 1H/2H split,
home/away team-total split. Three consequences are written down so nobody rediscovers them:
(i) **at the main line `p_model ≈ p_fair` by construction** — the model's edge surface is
derivative markets (team totals, 1H, alt lines), which carry more hold, lower limits and the
most expensive per-event capture; shape is worth a few points of probability at best against
several points of hold, so **that gap is measured before more is built** (the
derivative-consistency study below); (ii) **the anchor is the devigged median, not the posted
number** — the prices at the line (−115/−105) say where the true median sits, so devigged
prices are inverted to a continuous location and the sim's shape converts median to mean;
(iii) **the sim is calibrated to the anchor by an explicit solve step**: stage 1's per-game
efficiencies are solved so that stage 1 + stage 2 + OT reproduces the consensus median margin
and total — **identified by holding the possessions distribution fixed** (pace model, or
Torvik tempo in year 1) and solving the **two efficiency levels** for the two targets (root-finding on common random numbers / a precomputed response surface, cheap);
without it stage 2's +3–6 points silently violate the anchor. **Stage 2 and tempo are
redistributions, not shifters**: the foul bump and OT move tails and the OT probability, and
the simulated median must still match the target within a tight tolerance (0.1 pt) *after*
them; a row whose sim median misses the tolerance is a **graded refusal** (`anchor_miss`), not a
buy. The solve has no opinion of who is better — two unknowns, two constraints, the mean is
spent before the sim starts. Derivative markets (alt lines, team totals, 1H) are mostly priced
mechanically off the main line, so shape error is where mispricing is more likely. The catch is
stated in the credit budget (§8): those are the per-event markets that cost the most to capture,
so which of them are captured is a budget decision made in Phase 0, not assumed.

**v0 ships first, and everything below is a rung above it.** v0 = consensus-centred
distribution: devigged median (margin, total), SD of **`final − close` residuals** per pre-close
bucket, key-number push and parity masses fitted separately, a margin-conditional foul bump
that redistributes only. It prices from day one; correlation, the full stage-2
sub-model, player scaling, halves carry, ratings are each a rung that must beat v0.

**`p_model` is one distribution object, not a set of per-market numbers**: median, scale, skew,
discrete margin and total pmfs (and the half pair); every market probability is read from it.
**Coherence tests run on every pass** and fail the pass as an integrity error: monotone across
alt lines; team totals sum to the game total; 1H + 2H = game; P(win) ≥ P(cover) when the line
is negative; game Over and team-total Over consistent in sign. **Monte Carlo is deterministic
per game-pass**: fixed seed per **`(event_id, model_version)`** — the same random numbers on every pass, so
probabilities move only when inputs move — common random numbers across the solve step and
the shape rungs, and enough paths (or control variates) that the standard error of any
tiered probability is < ⅓ of the tier's edge threshold — so a Strong does not flip to Watch on
noise between passes. The path count is **capped** (`max_paths` in `preregistered.yaml`); if the
cap is hit and the error is still above ⅓ of the edge, the row is a graded refusal (`se_cap`),
never a buy on a tiny edge bought with an enormous simulation. **Parameter uncertainty is priced**: shape and rating parameters are
sampled from their posteriors, the edge must exceed **k × its sampling spread** before tiering (k = 1.0 to start, written in
`preregistered.yaml`, revisited only at the review cadence), and the spread is a column on
the card.

**The distribution is a two-stage simulator, not a closed form.**
1. **Stage 1 — to the checkpoint.** Possessions drawn from a fitted distribution that depends on
   pace mismatch (the slower team dictates), expected closeness and game state (leaders stall,
   trailers run); efficiencies per team with team-specific dispersion (higher for 3PA-heavy,
   lower for slow / FT-heavy, shrunk to conference/tier) and a fitted home/away correlation;
   simulate to the end-game checkpoint (margin at ~4:00; checkpoint time fitted on PBP).
2. **Stage 2 — the end-game sub-model conditioned on the checkpoint margin.** Trailing team
   fouls intentionally inside a fitted margin window (starting hypothesis ≤ 6 at 4:00; CBB
   fouls earlier and wider than the NBA because of 20-minute halves and the quicker bonus),
   adding free throws and stoppages that have nothing to do with the earlier pace. Scaled, where
   the player layer exists, by the leading team's shrunk FT% of its top-4 by projected late-game
   usage and the trailing team's 3PA rate / shrunk 3P%; **where PBP (D12/ESPN) is not verified,
   stage 2 is a simple margin-conditional foul bump** and the player scaling waits. Expected
   effect +3 to +6 points in games within 6 at the checkpoint, ~0 in blowouts [HYPOTHESIS].
   OT (~5%) is a third draw from the same machinery.

Margin–total correlation (close games run higher totals), key-number mass (sides and totals),
cover probabilities and alt lines then **fall out of the simulation** instead of being added by
hand. A **diffusion / random-walk score model** (drift, volatility, pace, an end-game foul
regime; time-indexed so halves fall out naturally) is a rung beside the possession-level sim —
few parameters, easy to validate; possession-level ships only if the ladder says it wins.
**Alt lines are capped at ~±1 SD of the main line** until tail calibration is proven out of
sample (data is thin past ±20 and alt lines are the high-hold product). The simulator is validated by reproducing the **joint distribution of (margin, total)
residuals by spread bucket on the 2024-25 closes** before any price uses it; the 2008–2025
Torvik-centred finals only say how width moves with tempo and 3PA rate, not the level.

**Halves are a correlated pair, not a separate model.** 1H and 2H are simulated jointly (2H has
more fouls/FTs and timeouts; 2H > 1H by a few points — measured on D6 half scores 2015–2025;
1H/2H pace split from the game-state model; the rotation-model minutes split is Phase 1b) and the full game is their sum, so 1H markets are consistent
with the game by construction. **Halftime ties** are a real outcome at a few percent and books
settle `h1_ml` differently (push vs three-way) — the tie mass is carried explicitly and
`book_rules.py` records the settlement per book. **1H → 2H carry** — a **rung that has not shipped** (Phase 1b/5); year 1 freezes the 1H share
and half-to-half correlation and leaves the carry terms **off**. When it ships: key-big foul count at the half
(≥2 on a top-2 rim protector ⇒ fitted 2H defensive-efficiency penalty), 1H pace vs projected,
halftime margin, and the 1H team-foul deviation (team fouls at the half vs the team's season 1H
rate; men's team fouls reset at the half so this is an aggression / officiating signal, not a
carried bonus state) — each a fitted hypothesis term.

**Shape level is fitted on residuals around the close — `final − close` for margin and total,
per bucket — never on the dispersion of the closes themselves** (that measures how games differ
from each other, which the median already carries); Torvik-centred history gives relative
structure only. Torvik's projection errors are wider than the market's, so SDs fitted around Torvik's
mean are too wide for a close-anchored sim. The bought seasons (close-centred, ~5k games each)
set the level — global shape parameters are estimable there, per-team / per-bucket tail
parameters are not; the **SD level is fitted per market-quality bucket** (hierarchical toward
the global) — a bucket is defined only by **information available before the close it is fitted
on**: book count, high-major vs not, hours the two-way market has been up, spread size; never by
how tight the close turned out (that leaks the quantity being measured). Push and parity spikes
at the key numbers (margins 1, 2, 3; integer totals) are **fitted as their own masses**, not left
to a smoother that can eat them and show a fake edge on exactly the numbers CBB pushes on.
**1H share and half-to-half correlation are estimated from residuals conditional on the
full-game close** (NBA's 1H runs ≈ +0.77 pts/team, CBB's pace and foul splits are at least that
large) and frozen; 1H totals stay in shadow until the 600-close study passes — because a thin-consensus mid-major line has a noisier mean than a power-conference
line and the effective dispersion around it is larger; 2008–2025 Torvik-centred games supply only *how* SD varies with tempo,
3PA rate and closeness, rescaled before use.

**Variance is fitted on outcomes, with the market relation as a regularizer** — not copied from
the market's ML↔spread↔total conversion (CFB #382: a flat SD handed every dog value; but a
variance *copied* from the market inherits its favourite–longshot habits). Both fits are run and
compared; the audit reports which one the held-out season prefers.

**A second ladder, inside the distribution** (same rule as §3.7 — each rung must win OOS or is
cut; scored by **log score / CRPS on outcomes**, each rung with a **pre-registered minimum
effect in those units** — the MDE per rung is derived from pilot variance on the development
seasons and published in `preregistered.yaml` before the first slate, so Brier at the posted
line is never the arbiter of a shape rung; the "market-implied distribution" from one main
line per game is, stated plainly, *Normal around the close with a fitted SD* — that is inner
rung 0, and the real question is whether any shape rung beats it on outcomes): `Normal margin + total with fitted SD → + margin–total correlation → + foul sub-model →
+ player scaling → + 1H carry`. Correlation, fouling, player scaling and 1H carry do **not** ship
together as one block.

**One validation set, many rungs — the live season is the frozen holdout.** Kalman vs ridge,
additive vs multiplicative, capped margin, garbage time, hierarchical `w`, five outer and five
inner rungs would all be decided on the same one or two seasons with tiny effects; some "wins"
will be noise. So: 2024-25 (and 2023-24 if bought) are **development only**; the 2026-27 live
season is the **untouched test**, with the surviving rungs frozen in `cbb/params/` before the
first slate; rung tests are paired game-level, clustered by date, with a pre-specified effect
size and Holm / FDR across all rungs together.

**Whole-number pushes — sides and totals — empirical, not the smooth curve.** Finals do not land
smoothly: late fouling and the 2-vs-3 structure of scoring pile mass on some integers, and CBB
margins have their own lumpiness at −3/−7 etc. Phase 1 builds **push-mass matrices for totals
and spreads** from the historical finals — P(lands exactly on N) by posted number, conditioned
on the projected margin bucket and the foul-state at the checkpoint — **one push mass**: the mass stored on the row is the distribution's own fitted key-number mass,
and `market/ev.py` subtracts *that* mass (never a second empirical matrix) from the EV of a
whole-number line, with the no-push conditional on the remaining mass — so `p_model`, the
tier and the EV agree on the number being bet. The historical matrix is a *fit input* to those
masses, not a parallel estimate. The matrices are
**smoothed** (a parametric curve plus **explicit integer / parity spikes**, hierarchical across
numbers) because number × closeness × foul-state cells are sparse. The spike set is
**estimated on the 2008–2025 finals**, not hand-listed: which margins are over-dense (hypothesis:
2, 3, 5 from FT pairs and late threes; a trough at 1 in foul-state games) is read from the
pooled data and only then imposed as a penalty structure on the sparse cells, so the smoother
cannot flatten a real step. Monthly the audit compares
the sim's integer landing distribution with the matrices; divergence > the matrix CI on any
number flags the model.

**Whole-distribution calibration, monthly:** PIT histograms for margin and total, tail
frequencies (|margin| > 15, total outside ±12 of the line), key-number hit rates, and the bias
tables (Over/Under lean, favourite/dog lean) cut by projected closeness, pace mismatch and
month — a right mean with wrong tails is exactly where alt lines, dogs and key numbers get
mispriced, and Brier at the posted line cannot see it.

**Derivative-consistency study — model-free, first, on forward archive.** Before shape is
built: compare captured team totals, 1H totals and alt-line ladders against simple conversions
of the main line, with hold by market and market-quality bucket. If derivatives are priced
consistently off the main line, that alone does **not** kill the build — books can be
self-consistent and still wrong against outcomes. So the study has a **second leg on the bought
2024-25 closes**: does rung 0 (Normal around the close) misstate the realised outcome density
at the derivative targets (1H totals from ESPN half scores, alt-spread cells within ±1 SD,
team-total splits)? Shape is built where **outcomes disagree with the mechanical translation**,
whether or not the books do; it is not built where both agree. If the books are inconsistent
with each other, the same study feeds the market-only channel directly.

**Timing — the study cannot finish before the first slate.** Leg 1 needs forward-archive
derivative prices, which exist only once capture runs on live slates (~Nov 3). So: **before the
first slate** v0 and the generic shape (leg 2, historical outcomes) are frozen; **derivative
markets are priced in shadow only** (ledgered `shadow`, graded, never bought) until a
pre-declared archive exists (**≥ 600 derivative closes with a settled main line**, ~3–4 weeks),
and the first fixed review after that decides, by the pre-registered rule, whether derivatives
leave shadow. **Every rejected or deferred rung is shadow-run live the same way** (same
simulation, near-zero cost) and scored, so the review pass has real out-of-sample evidence
without a bet riding on it.

**Derivative benchmark.** Consensus CLV is validated on main-line closes; team totals and 1H may
have one or two quoting books, and CLV against a soft book is weak evidence. For markets without
a validated consensus the benchmark is the **main-line conversion** (plus a thin-book median
where ≥ 3 books quote), and **outcome-based scoring (log loss vs rung 0, date-clustered CI) is
co-primary with CLV** for promotion. "Derivative closes are no better than main-line
conversions" is one of the pre-registered hypotheses.

**Lineup mixture is shape, not a mean-shifter.** Availability uncertainty (§3.4) widens /
bimodalises the distribution and drives the pending alarm; an `AMBIGUOUS` T-90 status enters
stage 1 as an **explicit variance multiplier keyed to P(plays)** (max at 0.5, fitted on the
forward archive from status → outcome dispersion), so our own derivative prices and the
`pending` gate see the wider shape before the T-30 lineup — the market-only channel does not
use it (it compares books to each other, not to the model); information-latency edges (the T-90
beat-feed read) belong to the market-only channel, not to a rating shift.

**Market order.** Totals and team totals are priced first — pace, 3PA rate and fouling are where
this structure differs most from the market; sides stay `research_only` until the §3.7 ladder
says the ratings add something to the main line (if they never do, sides are the market-only
channel's business, not the model's).

### 3.7 Fair price, market anchor, complexity ladder, tiers, gates

**Fair price is a consensus, never the book we bet.** `p_fair` = power-devigged consensus of
the sharp reference (`novig`; Pinnacle via `regions=eu` *if* Phase 0 verifies it quotes CBB) and
the **all-book median excluding the execution books** (DK/BetMGM are never in their own
benchmark), with DK/BetMGM used **only as the execution price** (edge = p_final vs
the DK/BetMGM quote). Measuring CLV against the same soft book we bet mostly measures that
book's own movement; the ledger carries two CLV columns — vs execution book and vs consensus —
and the consensus one is the promotion criterion — **once the consensus has earned it**: a
Phase 1 **consensus-validation study** scores consensus close vs each single book's close on
Brier and log-loss against outcomes (bought 2024-25 closes); if the consensus does not beat the
best single book, the benchmark is redefined (that book, or Pinnacle-eu if verified) before any
CLV number is called promotion evidence. The study is **re-run every August on the previous
season's own captured closes** — sharp books and the soft books' shadowing of them change, so
a benchmark validated on 2024-25 is not assumed valid in 2027. Because stale-line/market-only buys are the quickest
route to a limited account, capacity is a first-class metric: **limit-adjusted CLV** = CLV ×
placeable units (from the recorded max-bet), **fill quality** (price still available when the
buy is placed — `filled_at_price` = the §4.a executed price, `moved_before_fill` = executed price minus the archive's quote at the auto-stamped entry time), and a `max_per_book_per_day`
cap with buys spread across the books held. Weekly audit reports placeable units per week, not
just edge. New execution outlets are a terms/legality decision for Franz, not a plan item.

**Two channels, same probation.**
* **Market-only channel** (cheapest, built first): soft-book quote vs sharp consensus, plus a
  stale-line flag (quote unchanged while the consensus moved ≥ ½ pt / 5¢). No model at all. Our
  five engines' audits say the close beats the ratings; this channel tests whether the *books*
  disagree enough to pay, using data capture already produces.
* **Model channel — blended in location space, one simulation for every market.** Not
  `p = w·p_model + (1−w)·p_fair` per market (that breaks the coherence the simulator provides:
  blended ML / ATS / total / alt / team-total probabilities would no longer share a
  distribution). Instead the **margin and total means are shifted** by
  `w_m · (rating mean − consensus median)` and `w_t · (…)` and every probability is read off
  the single calibrated simulation (§3.6). Two `w`s (margin, total) per market-quality bucket,
  hierarchical, **prior centred at 0** (not ½ — four sports of evidence say the close wins; a
  prior near ½ leaks weight into thin buckets), so a bucket's `w` leaves zero only by beating
  its published MDE; a missing Torvik file skips the blend; the ESPN box-score rating stays
  report-only, including as a fallback. The stale-line / soft-vs-sharp flag is **never an input
  to `p_model`** — it is hypothesis #1, judged on fill-verified CLV, and putting it in the
  distribution would count one slow book twice. `w` is **not a free cell per market × month × quality** (with one season of closes most cells would hold a few hundred rows — a
  convincing backtest and a dead ledger). It is a **parametric / hierarchical form**: a global
  `w₀` with partial-pooled offsets for month-of-season and market-quality, shrunk to
  `w₀`; `w` is allowed to go to 0. `m_unmodelled` (§3.3) is one of those offsets, not a separate
  free multiplier.

**Complexity ladder (the rule that decides what ships).** Each rung must beat the rung below
out-of-sample on consensus-CLV and Brier on the held-out season/weeks by a **pre-specified
minimum effect** (written in `preregistered.yaml` before the fit: Brier ≥ 0.0010 and mean
consensus-CLV ≥ 0.5¢, both with the slate-day-clustered bootstrap CI excluding 0); "no clear
win" means cut, never "keep because it's plausible":
`close alone → + Torvik projected spread/total (super_sked) as the year-1 rating mean → + HCA /
neutral / rest → + own Kalman/ridge rating (Phase 1b) → + player layer (high-majors only, Phase
1b) → + situational terms (the single regularized model of §3.5, MDE-screened)`. The Torvik
projection rung is what breaks the chicken-and-egg: it is a free rating mean that lets `w_m`,
`w_t` be tested from November without building `prior.py`, and it is admitted only after the
as-of leakage check on `super_sked` passes (first Phase 1 task). The sheets' metrics
(§6) sit outside the ladder entirely.

**Honest data budget.** Torvik-side fits (ratings, prior inputs 1–5, hand-off vs Torvik's own
projections) have 2008–2025. Anything fitted *against the close* has the bought seasons only
(2024-25; 2023-24 if credits allow). The futures/opening-line prior input (#6) has **no
history** — D2b is forward capture — so in year 1 it enters with a **hand-set, labelled weight
(`[HAND-SET year-1]`)**, audited weekly by the hand-off curve, and is fitted only once a season
of its own data exists. **How it is chosen:** five shadow priors — **all futures weights zero** (the null, scored like
the rest so "no futures weight" can win week 5; picking the best of four positive weights ships
a prior the games did not ask for), plus futures weight low / mid /
high = 0.15 / 0.30 / 0.45 of the November blend, plus **mid-halved-for-brand**: 0.15 for teams
with preseason AP votes / brand tier, 0.30 otherwise — the hand-set stand-in for the fitted
public-hype shrink until it has data) run in parallel from day one, logged and
Brier-scored against outcomes and the market, never bet; **until week 5 the card mean is the
consensus median — no futures weight is applied to a live price before the test**; at week 5 the
**review pass itself** (the fixed-cadence Monday audit, §4.6) scores
**all five** on OOS residual log-score vs the real November closes; **the null (all-zero) stays
unless a positive weight beats it by the pre-registered MDE** (two November weeks will crown
0.15 or 0.30 by noise otherwise, and that weight would move live totals all season); the result is written into
`cbb/params/prior_weights.json` — versioned, commit hash in the audit header, **once**; it is
not re-chosen at later reviews (that would be chasing), only re-fitted next summer with a
season of data. Any coefficient tagged HAND-SET appears in the audit's header.

* Tiers from `cfb_engine/market/tiers.py` (rank on edge, `min_ev=0`, `max_edge` reads big
  disagreement as model error). Price bands per market × market-quality bucket (§2). One buy
  per game-direction; a team total Over and the game Over are one opinion — dedupe.
* Gates, every one a named `pass_gate`: `starter_unknown`, `one_way_quote`,
  `settlement_unverified` (`book_rules.py`, 60-day expiry), `ml_sharp` (research, neutral when
  absent), `drift` (measured, not vetoing), `pending` (news alarm), `thin_market`,
  `stale_quote`, plus the simulator's own refusals **`anchor_miss`, `se_cap`** and one name per
  failed coherence test (`halves_sum`, `team_totals_sum`, `monotone_alt`, `win_vs_cover`) —
  a row the simulator refuses lands in the same graded-refusal table as a thin market, never
  as a dropped pass. **The gate funnel is simulated on the first two weeks of archive** (how many
  games survive each gate, by market quality) before any gate is tightened — the mid-major
  thesis and the gates pull in opposite directions and the plan must know which wins.
* **Exposure and limits.** Per-slate cap on correlated buys (same conference tournament day,
  same-team parlays of game + team total are already deduped); max N units per slate and per
  conference; a book-limit event (max bet below 1 unit on a quote) is recorded per quote, the
  quote is flagged `limited`, and the audit reports ROI/CLV with and without limited rows.
  Max-bet / limit per quote and a stale-line flag are recorded from day one even if unused.

---

### 3.8 Machine learning — where it earns a place, where it doesn't

No engine in this repo uses boosted trees or a logistic classifier to predict outcomes, and
CBB won't either as its "brain": in five engines the close beat the ratings, so a more flexible
model on the same inputs cannot create information the market lacks; margin residuals around
the close have SD ≈ 10 pts against situational effects of 0.3–1 pt on ~5–6k games per bought
season (trees fit noise there and tuning adds forking paths); and a model that emits P(cover)
directly bypasses the one-simulation design and its coherence tests. ML enters only as
**pre-registered ladder rungs that output parameters into the distribution object or rank
opportunities**, each shadow-run beside a linear baseline with the same MDE.

| Tool | Role | Why it fits |
|---|---|---|
| Logistic (2-param, logit space) | **Recalibration** of final probabilities, refit only at reviews | Platt slope/intercept answers "does a 4% edge realise more than 2%" with two parameters |
| Logistic, hierarchical | **Blend diagnostic**: outcome ~ logit(p_fair) + model signal | Slope ≈ 0 on the model term is a clean verdict on `w` |
| Logistic | **P(plays)** ~ status × tier × source × time-to-tip | Smoother than separate Beta cells as the archive grows (§3.4b) |
| Logistic | **Fill probability**: P(stale quote still available at next featured pass / next card) | Supplies the §4.e fill haircut |
| GBM / GAM | **Closing-line-movement forecaster**: consensus close − current quote from book dispersion, time-to-tip, which books moved, steam, quality | Continuous, far less noisy target; many forward-archive rows; ranks/sizes the market-only channel. Baseline = rule-based stale flag; promoted only if it beats the rule on out-of-sample CLV |
| GBM / GAM | **Shape targets** on the ~100k-game Torvik-centred history: possessions, half split, foul-game FT volume, dispersion | Strong signal, outcome-labelled (not price-labelled); feeds simulator shape parameters — the "own the shape" half of the design |
| GBM | **Opportunity quality**: predict CLV (not W/L) for candidate buys | Trained on all candidate rows, never buys only |

**Not used for:** direct W/L or ATS prediction or end-to-end bet selection (one bit per game,
overfit); replacing the Kalman/ridge ratings or the anchored mean; props (no price history);
anything trained on live buys only (selection bias).

**Guardrails for any rung:** season-blocked, time-ordered validation; hyperparameter budget
fixed in `preregistered.yaml`; shallow trees, strong regularisation, monotone constraints
where logic demands; as-of features only (caught by §4.y); per-row contributions (SHAP) written
into `edge_by_rung` so attribution survives; seeds, library versions and `code_sha` pinned.

**Order:** (1) logistic recalibration + fill-probability model (cheap, review-cadence);
(2) GBM/GAM shape targets on Torvik history, Phase 1; (3) closing-line-movement forecaster
once a few weeks of forward archive exist, against the rule baseline; anything else only if one
of those earns its place.

## 4. Audit function

Ported from NHL (`nhl_engine/audit/ledger_report.py`, PR #468) and CFB probation, with the CBB
cuts of §2. One command, `cbb-engine audit`, run in the 03:00 pass and again before the card:

1. **Grade yesterday** from ESPN finals + half scores + boxes (DNP → void for props), write once
   to `cbb/ledger/graded_<date>.csv.gz` on engine-state; each priced pass is its own immutable
   `priced_<stamp>_<tag>.csv.gz`.
2. **Stamp closes**: last snapshot before tip with ESPN `status=pre`; a close after tip is
   dropped and counted (CFB #377 found 66 in-play closes).
3. **Rebuild the whole-ledger report** (md + PDF, attached to the card email like NHL). The
   report is a **fixed set of DuckDB views over the row grain
   `(event_id, market, side, line, book, pass_id)`** — `v_clv` (live `bet_of_record` rows only:
   consensus ¢ and execution ¢, fill-verified and haircut, slate-day-clustered CI; main-line
   Brier is a *column* here, a coherence check, never the headline), `v_shape` (predicted vs
   realised push and parity by number, live totals), `v_rung` (incremental CLV and CRPS of each
   named rung over the rung below), `v_gate` (refused-row CLV by market × tier × quality incl.
   `anchor_miss`, `se_cap`, coherence names), `v_persist` (stale flags still available at the
   next featured pass and at the next card, by book and market), `v_prior` (five November
   variants on residual log-score vs the close, same games). **The views are the analysis; the
   PDF is a rendering.** Order: **page-one lights (§4.d) first**; then W-L-P / ROI / PPV under
   them next to the base rate and the break-even the price required (ROI descriptive until
   several thousand buys, §4.6) → exact sign test on CLV → CRPS and cut 9 (the scores that can
   move — at the main total `p_model` sits on `p_fair` by construction, so model-vs-close Brier
   ties even when the push masses are wrong; Brier is informative only on shadow derivatives) → totals
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
6. **Probation — CLV-primary, pre-registered.** A flat-unit ROI has SD ≈ 1u, so SE(ROI) at
   n=100 is ≈ 10 points and ">1 s.e., both halves agree" is noise applied to dozens of cells.
   The rule is therefore: **consensus-CLV sign test and mean-CLV 95% CI > 0** on n ≥ 100 as the
   promotion trigger (CLV converges far faster than ROI); ROI and W-L are **descriptive** until
   several thousand buys. **~5 primary hypotheses are pre-registered** in `cbb/params/
   preregistered.yaml` before the first slate (**live** candidates, all on fill-verified consensus CLV of game totals: market-only channel
   beats 0 CLV; the end-game component removes the Under lean; `starter_unknown` refusals have
   worse CLV than admitted rows; featured passes beat the 15-min subset (cut 10). **Research /
   shadow** candidates, scored on `v_prior` and research rows, never on live probation: the
   prior beats all-zero through game k; mid-major ATS beats power-conference ATS); every other audit cut is descriptive or Benjamini–Hochberg
   controlled. **Dependence and peeking:** buys on one night / one conference are not
   independent, so every CLV test resamples **by slate day** (cluster bootstrap), not by row;
   the review runs on a **fixed cadence** (every 4th Monday, dates listed in the yaml) and a
   promotion between reviews is impossible — no continuous peeking. `preregistered.yaml` is
   **committed to git before the first slate** (the commit hash is printed in every audit
   header) so pre-registration is provable. **Kill switch = futility, not "CI includes 0"** (which fires
   from lack of power at a few hundred clustered buys even when a small edge exists): the
   market-only channel stops when the **upper bound of its consensus-CLV CI falls below the
   minimum worthwhile CLV (+0.5¢)** at a pre-declared minimum n (if per-bet CLV SD ≈ 3¢ and
   slate-day clustering inflates SE ~1.4×, the MDE at n=100 is ~1¢ and futility at +0.5¢ needs
   ~440 buys — so **CLV variance is estimated before launch**, §4.e), *and* the ladder keeps
   nothing above `close alone`; the audit prints the **minimum detectable CLV for the n
   actually in hand** beside the verdict so "not yet" and "no" are distinguishable. Then the
   engine becomes a logging/research tool — capture, ledger and audit
   keep running, the card prints no buys, and reopening requires a new pre-registered
   hypothesis. Review retires/reopens; never a hand edit. **`w` changes only by refitting the
   hierarchical model at a review** — no trailing-window auto-shrink per bucket (that is the
   peeking this rule bans, on the thinnest cells). **Drawdown pause (operational, scaled to volume):** a fixed
   "15u in 14 days" is ~1.3 SD at ~10 buys/day and would trigger on noise in one window in five
   to ten. The pause is therefore defined in **SDs of the expected variance for the bets
   actually placed** (drawdown > 2.5 SD of Σ stake-variance over the window) **and requires
   CLV deterioration in the same window** (mean consensus CLV < 0 on those rows); then live
   buys pause (rows printed `paused`, still ledgered and graded) until the next review. A
   drawdown with healthy CLV is reported as variance, not acted on. **Optional anytime-valid
   monitoring** (confidence sequences) may drive *stop* decisions between reviews; promotion
   never moves off the fixed cadence. **Fallback drift table:** the ESPN box-score rating (10.11)
   is scored daily against Torvik on MAE/Brier even when Torvik is up, by §2 cut; it is a
   reported table that validates the fallback and may nominate a ladder rung, never a
   per-cut weight change by rule.
7. Weekly audit email (Monday) with the regression report and the sheet ledger audit (§6).
8. **Standard side/situation cuts** (descriptive, n and CI printed, never a promotion
   criterion): favourite vs underdog, home vs road favourite, home vs away underdog, rest vs
   no-rest and rest differential buckets, conference vs non-conference, neutral, MTE/tourney,
   month — each with the naive baseline on the same rows.
9. **Shape on shape**: realised push rate and parity rate vs the mass the distribution
   predicted, **by number**, for buyable totals — a model that matches the median but mis-states
   landing-on-the-number leaks on the prices bet most often.
10. **Featured-pass value** (pre-registered, decides the credit spend): CLV of rows priced on a
    10-minute featured pass inside T-3h **vs the CLV the 15-minute pass would have caught** on
    the same games (the 15-min quote is a strict subset of the archive, so this is computable
    from the tape); if the extra passes do not buy CLV above their MDE by the 4th Monday, the
    cadence falls back to 15 min and the reserve is released.

### 4.a Learning from the ledger — three levels, not one

A flat-unit bet has SD ≈ 1u, so one result carries almost nothing; the audit above answers
"is the engine working in aggregate". Learning needs three levels: **per bet** (what did we
know, was the price real), **per cohort** (CLV, calibration), **per component** (which part
earned the edge). The sections below add the first and third.

**Ledger row = decision snapshot, immutable at write time.** Beyond book / quote / captured_at /
gates, every priced and refused row carries:

| Group | Fields |
|---|---|
| Identity | `event_id` (never matchup text), market, side, line, book, `pass_id`, **`pass_kind`** ∈ {board, featured, close}, **`bet_of_record`** (the card pass allowed to write; close passes are archive-only and never create a second decision row), **`status`** ∈ {live, shadow, research}, `channel`, **`stale_flag`** as its own column (a model buy that happens to be stale stays a model row) |
| Provenance | `model_version`, frozen-params hash, **`code_sha`, `env_lock_hash`, input-file hashes (Torvik / ESPN / RotoWire / board snapshot)**, seed, rung set, `channel` (market-only / model), data-freshness stamps |
| Decision | anchor median, **target median vs simulated median, path count, standard error**; quality bucket id **plus the pre-close inputs that defined it** (book count, high-major, hours two-way, spread size); **predicted push mass and parity mass at the line bet**; predictive summary (median, scale, skew; margin + total pmfs stored, not just hashed), edge, edge sampling spread, tier, every gate's value and threshold, **`edge_by_rung` / `p_by_rung`** in probability **and in points of margin and total**, over the **fixed ordered rung list in `preregistered.yaml`** (v0 shape → foul redistribution → half-correlation → blend shift → availability residual; each value = change from the rung beneath; a rung not on the list cannot be attributed later); availability: points applied, which estimator supplied them, event count in hand; **all five prior-variant probabilities on every November row** |
| Context | availability state at T-90 and T-30, consensus vs execution quote, open, quote priced on (archive shard id), T-30 line; **both close prices, both close times, `close_status` ∈ {pre_tip, post_tip, pulled}, and the shard id each close came from** |
| Execution | `placed` ∈ {Y, N, **unknown**} with N reason codes (`line_gone`, `limit`, `bankroll`, `judgment`), executed price, stake, book, **entry time auto-stamped** (never typed), limit, **archive quote id at entry**; `unknown` stays unknown at 03:00 |

A past row is therefore replayable (`cbb-engine replay --row`) and attributable.

**Placed bets are entered, not assumed — and entry must be near-frictionless or it rots.**
`cbb-engine placed` (one line per bet), a CSV import from the book export, **and** a one-tap
path (a link per row in the card email / a phone shortcut that posts the row id + price)
record what was actually bet; rows with no entry by 03:00 are `unknown`, never silently `N`; the audit reconciles to book
statements and reports **recommended vs executed equity curves**, slippage by book, skipped
buys and unlisted bets. Fill quality's data comes from here, nowhere else.

### 4.b Post-game tags — fixed taxonomy, versioned, attached automatically

After grading, each row gets tags from ESPN boxes (and PBP where D12 verifies): `ot_flip`,
`foul_game_swing` (margin at ~4:00 vs final crossed the line), `lineup_surprise` (projected
starter DNP or unplanned starter), `key_number_miss` (lost / pushed by 1), `pace_miss`
(possessions outside the 80% band), `shooting_variance` (3P% / FT% outside the posterior
band), `garbage_time`, `steam_agreed`, `steam_against`, `late_news_after_price`. The taxonomy
is written **before the first slate** (`cbb/params/tags.yaml`, versioned) so it cannot become
after-the-fact storytelling. Outcome-linked tags are defined **side-symmetrically** (`key_number` = |final − line| ≤ 1
regardless of direction; `ot`, `foul_game_swing`, `steam` likewise) and reported as **net
swing — bad beats minus lucky wins** — so a tag cannot be a loss by construction. The audit
compares tags **among losses vs among wins** (or regresses result on the tag set), with
**FDR control across the tag × market × quality family**; over-representation points at a
defect, no tag over-represented is labelled **variance** — the explicit "right and lost"
bucket. A **tag → review-action map is written in advance** (`tags.yaml`): lineup surprises up
→ starter gate; pace misses up → pace model; foul-game swings up → stage-2 trigger; steam
against up → drift gate / capture cadence — so a tag result has one pre-agreed consequence. Every graded game also stores a **PIT value** and a **surprise score**
(−log p of the outcome under our distribution); the Monday digest lists the week's biggest
surprises and biggest CLV winners / losers.

### 4.c Cohort and component learning

* **Luck-adjusted P&L**: for each buy, realised result vs what the closing consensus implied;
  cumulative "results vs close-expected" with a band. Down but inside the band = process fine.
* **Edge calibration + tier monotonicity**: buys binned by claimed edge → realised CLV per bin;
  the test is a **pre-specified trend (CLV slope on claimed edge, clustered by slate day)**,
  not pairwise tier comparisons, which are underpowered at a few hundred buys; "unordered" is
  never declared from a sample below the published n. Recalibrated only at a review.
* **Component attribution**: CLV and calibration split by channel (market-only vs model), by
  rung (v0 shape / foul bump / halves / Torvik projection), by market-quality bucket — "what
  earned the edge". Attribution uses `edge_by_rung` per row, so each rung's CLV is the
  increment over the rung below. **Challenger vs champion on the same games, concurrently, is
  the primary evaluation of any change**; before/after around a change date is descriptive
  only (November and March are different populations). **Structural changes are capped at two
  per review** so attribution survives.
* **Gate value, stratified**: raw admitted-vs-refused is confounded (gates select on
  features); gate cost/benefit is estimated within market × tier × quality strata, with a
  minimum n before a gate may be retired. The §4 table stays, labelled descriptive.

### 4.d Records, corrections, scorecard

* **Corrections table, append-only** (`cbb/ledger/corrections.csv`): ESPN final / box
  corrections, postponements, voids, book-specific settlements; each graded row carries
  `grading_status` (`graded` / `regraded` / `void` / `pending_correction`) and is re-graded
  from the correction, never edited in place.
* **Hypothesis registry + change log** (`cbb/params/registry.yaml`, append-only, schema'd so
  the scorecard links to entries): every model or gate change links hypothesis → evidence →
  review date → version hash → champion-vs-challenger result on the pre-specified metrics.
  The registry also holds the **exact formulas**: expected units per bet under the closing
  consensus, the variance used for the band, push handling. Decisions are not re-litigated.
* **Page-one scorecard**: 8–10 pre-registered traffic lights (market-only CLV, model CLV,
  calibration, tier monotonicity, integrity, capture gaps, execution slippage, tag
  over-representation, drawdown-vs-CLV), each with effect size and CI against its threshold;
  every light has a **grey "underpowered" state** distinct from red (n below its published
  MDE), lights change status **only at reviews** except integrity and capture alarms, and one
  light is **opportunity persistence** (below). The long tables move to an appendix.

### 4.e Phantom CLV — the market-only channel is judged on fills

A stale-line flag (consensus moved ≥ ½ pt, soft quote didn't) has positive consensus-CLV
almost by construction — *if* the price could be bet at that moment. Stale quotes vanish fast
and the card is an email at 11:30/16:30 on 15-minute captures. So: (1) **opportunity
persistence** — the share of flagged stale quotes still available at the **next featured pass** and,
separately, at the **next card** (`v_persist` splits the two), by book and market — is a scorecard light; if most vanish, the emailed card cannot carry the
market-only channel and real-time alerts become a decision, not an assumption; (2)
**hypothesis #1 (market-only beats 0 CLV) is judged on fill-verified rows**, or on all rows
with a **fill-probability haircut estimated from the executed sample** — never on flags alone;
(3) **pre-launch estimate**: before the first slate, buy a small sample of **~10 dense
historical slates** (the historical endpoint has 5–10 min snapshots) to measure stale-line
frequency, persistence and the per-bet CLV SD — this sets the MDEs in `preregistered.yaml`
and says whether the channel is workable at card cadence. Costed against the credit budget
first and bought only on your go-ahead, like the main 2024-25 purchase.
* **What is learnable when**: week 1 — integrity, capture gaps, slippage; weeks 2–4 — gate
  funnel, tag frequencies, availability-read accuracy; ~8–12 weeks — first CLV verdict on a
  pre-registered hypothesis; a season — calibration, tier order, component attribution;
  several seasons — any P&L verdict. ROI stays descriptive until then.

**Storage.** Git (`engine-state/cbb/`) keeps only the small ledger, params and audit outputs.
The quote archive — tens of millions of rows over a season — is **immutable daily Parquet files
with content hashes** (per-pass shards, daily manifest), encrypted off-site copy (rclone crypt), **DuckDB** as the
query layer (single file, no server; cohort and tag cuts stay fast). **Daily ledger files are
hash-chained** (each file embeds the previous day's hash, printed in the audit header) and the
**chain head is anchored externally** (weekly OpenTimestamps stamp, plus the hash pushed to a
second remote) — a chain one operator controls proves consistency, not independence; the
anchor makes "never rewritten" actually provable. Off-site copies hold bets and statements, so
they are **encrypted** (rclone crypt).

---

### 4.z Frozen vs adaptive — what the live holdout may and may not touch

The 2026-27 season is the untouched test only if the line between frozen structure and
pre-declared adaptive state is written down. Everything adaptive is stamped as-of and replayable.

| Frozen before the first slate (changes only at a fixed review, by pre-registered rule) | Adaptive during the season (update rule declared here; no other state moves) |
|---|---|
| model structure; which rungs are live / shadow / off; hyperparameters (SD level per bucket, correlation, foul bump, push spike set); `w_m`, `w_t`; `k`; tier thresholds; gate parameters; MDEs; `preregistered.yaml` | Beta priors per status × tier — conjugate update from each day's status → box outcome |
| the consensus definition and book set | status → outcome dispersion for the `AMBIGUOUS` variance multiplier — refit nightly, as-of |
| the Torvik projection rung (as-of only) | Torvik/ESPN inputs themselves (data, not parameters) |
| the shadow-prior variant (chosen once at week 5) | credit budget allocation (operational) |

### 4.y Time-travel leakage test (automated)

`cbb-engine audit --time-travel D` replays the full pipeline as of date `D` on a sample of
slate days each week and asserts that **no input row carries a stamp after D** (Torvik file
date, injury `posted`/`seen`, quote `captured_at`, super_sked projection date). A failure is
an integrity fail in the report. Phase 1 specifically verifies that D5's "pre-game projection"
was not computed retroactively (if it was, Torvik's baseline is leakage and is replaced by the
as-of Kalman replay).

### 4.x Line movement vs our side (graded, never a veto)

Every priced **and refused** row records three quotes per market: first archived (open), the
quote we priced on, and the pre-tip close. The audit then reports, by market × tier × market
quality: share of rows where the market moved **toward** our side (steam agrees), **away**
(market disagrees), or flat (< ½ pt / < 5¢); W-L, ROI and CLV for each bucket; and the same for
refused rows (did the gate refuse sides the market later agreed with?). This is the CFB
`drift` / `lineagree` design — a graded table only. It becomes a filter only if probation
passes it, because the MLB movement rules did not port to CFB and we have no CBB evidence yet.

## 5. Outputs, delivery, schedule

**Workbook `PayoffPitch_CBB_<date>.xlsx`** (tabs as NBA §8): Buys · Games (every game × 9
markets: model line, market line, model %, fair %, edge, gates) · Props (research_only) ·
Regression · Injuries & Schedule (status + timestamp, minutes impact, rest, trip game #, miles,
tz, altitude, MTE/tournament flag) · Line Moves · Fades (refused rows with their gate) · Audit ·
Legend. (**Worksheet** and **Totals Sheet** tabs are Phase 4, §6/§8 — not in the year-1 package.)

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

**Audio** (**Phase 4, after probation** — not in the year-1 package): MP3 narration of the card with edge-tts (MLB/NHL pattern, no AI key), ~4–6 minutes:
buys, the three biggest injury/line stories, audit lead. Research-only props are not narrated.

**Email**: one message at the card pass with PDF + xlsx + txt + md + the newest ledger
audit md/PDF (NHL `email.send_files` pattern), via the existing SMTP config.

**Schedule (Mac launchd, `scripts/cbb/macos/`, Desktop/engines shortcuts, idempotent installer
like NHL):**

| Time (ET) | Job |
|---|---|
| tip-scaled, 09:00–23:30 on game days | `capture` (board + 1H + team totals + props), idempotent; per-event cadence **scaled to time-to-tip**: 60 min at T-6h+, 30 min T-6h→T-3h, 15 min inside T-3h and to tip (so the close is clean); futures once in October. The watch tick below spends **no** Odds API credits (ESPN/RotoWire only) and never triggers a board pull |
| every 10 min inside T-3h | `featured` pass (h2h/spreads/totals, 3 credits) — the clock `pending` counts on; `pass_kind=featured` |
| every 5 min 11:00–23:55 | `watch` news alarm — **ESPN + RotoWire only, never pulls a board, spends no credits**; a hit forces the next featured pass early |
| 03:00 | `audit` (grade, closes, ledger report) |
| 11:15 | preview card (no buys) — weekday noon tips exist (MTEs, conference tournaments) |
| **11:30** | `card --email` **on any day with a tip before 16:00 ET** (MTEs, holiday and weekend blocks, conference tournaments) — the ESPN schedule decides, not a calendar. The **whole slate is priced**: early games are the card's buys and are written to the ledger; night games appear as **"preliminary — re-priced at 16:30"** and are *not* written |
| **16:30** | `audit` + `card --email`. Evening-only slates: the only card of the day. Split days: a **second email** — night games priced on fresh lines/lineups and written; the morning's early games are shown with status/result only (write-once ledger: never re-priced). So a split day produces two emails, an evening-only day one. **The default times are measured in Phase 1 by replaying the card at T-6h/T-3h/T-90/T-60/T-30 against archived closes** (NBA §10) and moved if CLV says so |
| per game, T-60 → T-5 | `close` passes (pre-tip only, `pass_kind=close`, **quote-archive rows only — never a decision row**) + starting-lineup read at T-30 |

One cadence table owns the clocks: **60 / 30 / 15 min board by time-to-tip; 10-min featured inside T-3h; 5-min watch on ESPN/RotoWire only.** Every shard carries `pass_kind` so `v_persist` can tell them apart.

Tournament weeks (conference tourneys, NCAAT: 16 games/day from noon) are the early-tip case
above: preview at 09:00, card at 11:30, automatically from the ESPN schedule.

---

## 6. The CBB worksheet and totals sheet (the MLB hand sheets, translated) — **Phase 4, deferred**

Neither sheet is built or emailed in year 1 (chart, §8 Phase 4, §9.4, scope). The spec below is kept for that phase.

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

## 7. Architecture — `engine_runtime/` + a thin `cbb_engine/` (decision 31)

CBB is **not a sixth copy**. The sport-independent machinery lives in a new `engine_runtime/`
package (built for CBB alone; existing engines untouched; `engine_common` stays math-only):

```
engine_runtime/
  oddsapi.py       one Odds API client; calls credits.debit(sport, cost) before every paid request
  credits.py       shared daily credit ledger (fail-open; budget file with per-sport slices + CBB featured reserve)
  quotes.py        per-pass Parquet shards (encrypted off-site via rclone crypt), daily manifest hash → git, DuckDB query layer
  devig.py         power de-vig; all-book median excluding the execution books
  ledger.py        NBA-shaped row generalised: grain (event_id, market, side, line, book, pass_id), status, pass_kind,
                   bet_of_record, §4.a snapshot groups, write-once, hash-chained, append-only corrections
  runner.py        pass runner: capture → price → write_once → grade; pass_id, code_sha, env_lock_hash, credit spend,
                   snapshot hash on every step; missing close/capture ⇒ alert, never a silent CLV gap
  views.py         the DuckDB views (v_clv v_shape v_rung v_gate v_persist v_prior) over the ledger grain
  state.py         engine-state push/pull for the small git side (ledger, params, manifests)

cbb_engine/        sport logic only — no runtime code, no re-implemented client/ledger/store
  config.py        CBBE_* env, books DK/BetMGM, market map (basketball_ncaab keys → status live/shadow/research)
  preregistered.yaml  rung list (ordered), k, max_paths, MDEs, hypotheses, prior candidates
  data/            espn.py, torvik.py (as-of), rotowire.py, beat_feeds.py, futures_scraper.py, vsin_splits.py,
                   arenas.py, teamnames.py, book_rules.py
  models/          distribution.py (residual width, push masses, solve step, foul redistribution, OT, 1H pair),
                   availability.py (status log, Beta priors, post-solve residual shift, race report)
  market/          gates.py (incl. anchor_miss, se_cap, coherence names), priceband.py (totals juice band),
                   blend.py (w, prior 0), stale.py (market-only channel, never into p_model)
  output/          excel.py, report.py (PDF rendering of the views), card.py, email.py
  cli.py           capture | watch | close | card | audit | placed | replay   (worksheet/totals/audio: Phase 4)
  Phase 1b (not year 1): features/strength.py, prior.py, players.py, minutes.py
scripts/cbb/       studies (residual fit, push masses, half shape, dispersion, send-time replay), macos/ launchd
tests/cbb/, tests/runtime/
```

Rule: `engine_runtime/` carries no sport thresholds; `cbb_engine/` carries no I/O client, store
or ledger of its own. Details and the credit-ledger contract: `docs/integration_master_plan.md`.

## 8. Build order (Devin sessions; season tips ~Nov 3)

| Phase | Work | Sessions |
|---|---|---|
| **0 — capture first** (before the first slate) | scaffold, config, team alias table; Odds API capture with a **written daily credit budget computed from the §5 cadence table** (60/30/15 board by time-to-tip, 10-min featured passes inside T-3h as the reserved slice, close passes; 1H/team totals/props only if the budget allows — per-event markets are the dominant live spend, costed before they are switched on); **October futures / win totals captured now** (one-time window); ESPN schedule/finals/halves; RotoWire + beat-feed seed; VSiN; engine-state `cbb/`; launchd; `book_rules`; **verify D11, D12, super_sked column map and Pinnacle-eu early**; record max-bet/limit and stale-line per quote from day one; **daily raw archive of Torvik + ESPN files** (unofficial sources change; the as-of replay needs the originals); the plain card + email. **First two deliverables, before anything else is built: the written credit budget and the gate-funnel simulation** on the first archive days — they decide feasibility, not just speed. **Dry-run week (last week of October)**: fake slates through the whole pipeline, both emails, write-once rules, time-travel test, with `CBBE_DRY_RUN=1` and no ledger writes to `cbb/`. | 2 |
| **0b — dry-run week go/no-go** (before ~Nov 3) | checklist, each item pass/fail in the audit header: credit budget written and a simulated 80-game Saturday fits it; gate-funnel sim run; D7/D11/D12/super_sked/Pinnacle-eu each verified or marked off with its fallback active; both emails sent from the dry run; write-once and time-travel tests green; `preregistered.yaml` committed with MDEs and `k`; `year1_scope.md` matches the code. Any fail ⇒ research mode for opening week, not a live card. | within 0 |
| **1 — data + shape** | Buy 2024-25 game-line closes (2023-24 too if credits allow); Torvik loaders with as-of slicing; **market-only channel first**; **derivative-consistency study** on the forward archive (decides how much of the shape layer is built); devigged-median anchor + solve step; inner ladder rung by rung (Normal-SD → corr → foul bump → halves; diffusion model as a rung) scored on outcomes; push matrices (smoothed); whole-distribution calibration; gate-funnel simulation; beat-feed latency study; time-travel test; frozen `cbb/params/` before the first slate. Deliverable: versioned fits + the ladder table. | 3 |
| **1b — ratings, only if earned** | Ratings as an outer rung: Kalman vs joint ridge, lineup-adjusted observations, form tests, hierarchical `w_m`/`w_t`. **prior.py (portal / NIL / coaches), player posteriors, minutes model** are built here, *after* the anchored shape engine shows `w > 0` is worth testing — not before. Until then the ratings feed shape features, team-total splits and the news alarm only. The October futures capture (Phase 0) still happens: free, one-time window, and the prior will want it if it is ever built. | 1–2 (deferred) |
| **2 — engine + card** | whichever rungs survived; calibrated simulator + halves + end-game component; tiers/gates/alarm; ledger; card PDF/xlsx/txt/md + email; exposure caps. | 2 |
| **3 — audit report + live probation** | `ledger_report` port with CBB cuts, pre-registered hypotheses, weekly email; Nov–Dec live; monthly probation review. | 1 + ongoing |
| **4 — deferred until the core passes probation** | worksheet + totals sheet (logged plainly, no fitted bands until they beat always-Over / Torvik's pick), MP3, long PDF, props research_only, podcast box. | 1–2 |
| **5 — extensions** | 2H/intermission markets, alt lines, tournament-mode card, cloud capture node if the gap trigger fires. | later |

~11–13 sessions plus the calendar the probation bar demands. Phase 0 is two sessions (capture +
budget + funnel + verification of five unverified sources, then card/email + dry-run) and must
finish before ~Nov 3; the October futures window closes first, so D2b capture is the first
command run in session 1.

---

## 9. Success criteria

1. Every buy has a pre-tip consensus close and two CLV columns; the markets we keep after 100+
   buys have consensus-CLV 95% CI > 0 (slate-day cluster bootstrap) on a pre-registered
   hypothesis, with placeable units per week reported beside it.
1b. The complexity-ladder table exists and every shipped rung beat the one below it OOS; the
   market-only channel's verdict is stated before the model channel's.
2. Shape scores on every audit: **CRPS, cut 9 (push/parity by number) and Brier on shadow
   derivatives** per market and §2 cut; main-line Brier is a coherence column (a tie is the
   anchor holding, a gap is `anchor_miss` leaking into buys). Markets the close wins by a wide
   margin are anchored (`w→0`) or shut by probation, not by hand.
3. `v_prior` has scored all five futures-weight variants on the same November games and the
   written weight is the one that beat all-zero by its MDE — or all-zero itself.
4. Worksheet and totals-sheet ledgers are **deferred** (Phase 4); when they exist they are graded
   daily with naive baselines on the same games and never touch a price.
5. No unmeasured coefficient ships unlabelled (year-1 `HAND-SET` entries are listed in the audit header); `ruff`, `mypy`, `pytest -q` green; `cbb-engine card` runs
   free with `CBBE_DATA_DIR` scratch, `CBBE_STATE_SYNC=0`, long odds TTL.

---

## 10. Decisions (taken 2026-10-01 after review; two modified)

1. **Historical closes — decided: buy 2024-25 game-line closes in Phase 1, defer 1H.** The prior
   fit without a close curve would be volatility with no benchmark. Blocker: the key in this
   VM shows 497 credits; the Mac key / NBA balance must be confirmed first.
2. **KenPom — decided: offline verification only, never in the live pipeline.** Torvik is the
   spine; a Torvik error halts the card until the ESPN fallback is validated, then degrades it
   (decision 11 supersedes the §1 halt rule).
3. **Coverage — decided: all 364 D-I teams ingested and modelled**, segmented by
   `market_quality`; the `thin_market` and `starter_unknown` gates protect the ledger and are
   graded like every gate.
4. **Books — modified.** Buys and CLV against the best of DK/BetMGM (the accounts held). A
   sharp reference is added for a *second* CLV column **only from a book the Odds API actually
   carries**: today that is `novig` (no-vig exchange) in the US regions; Circa and Bookmaker are
   not in the feed [VERIFIED], and Pinnacle (`regions=eu`) is verified in Phase 0. **Superseded
   on limits (2026-10-12):** account limits are a model-output question — limit-adjusted CLV,
   fill quality and `max_per_book_per_day` are first-class metrics (§3.7); the ledger records
   the book and max-bet of each quote so a per-book capacity table exists from day one.
5. **Card time — modified.** 11:30 ET card on any day with a tip before 16:00 ET (MTEs,
   holidays, weekends, tournaments — decided by the ESPN schedule), 16:30 ET on evening-only
   slates with an evening-only re-price on early days; the Phase 1 send-time replay can move
   either. A fixed 11:30 every day would price Big Monday 21:00 tips nine hours out with no
   lineup information, which the NHL and NBA ledgers say is the expensive end.
6. **Mac daemons** for Sports-Reference and the JS-guarded Torvik pages — as `nba_api`.
7. **Fair price = consensus (novig / Pinnacle-if-verified / all-book median); DK/BetMGM =
   execution only.** Two CLV columns; consensus-CLV is the promotion criterion.
8. **Probation is CLV-primary with ~5 pre-registered hypotheses; ROI is descriptive.**
9. **Market-only channel is built before the model channel** and judged first.
10. **`w` is hierarchical, not per-cell; at most 5 fitted situational terms; player layer
    high-majors first.**
11. **Torvik is not a single point of failure for long**: an ESPN box-score ratings fallback
    (plain opponent-adjusted efficiency, as-of) is built in Phase 1 so a Torvik outage degrades
    the card (`ratings_source=espn_fallback`, wider variance, Strong tier disabled) instead of
    halting it; the halt remains the rule until the fallback is validated against Torvik.
12. **Sheets, MP3 and the long PDF are deferred** until the core passes probation; the plain
    card + email ships from day one.
13. **State rules**: the alias table carries each team's state and the per-state prop /
    in-state-college restrictions for the books held; refused as `state_restricted`.

14. **Consensus is validated before it is the benchmark** (consensus-close vs single-book close
    on Brier/log-loss, Phase 1); execution books are excluded from the median.
15. **CLV tests cluster by slate day; review every 4th Monday; `preregistered.yaml` committed
    before the first slate; ladder rungs need a pre-specified minimum effect.**
16. **Three shadow futures-weight priors (0.15/0.30/0.45) logged, mid used until week 5.**
17. **Limit-adjusted CLV, fill quality, `max_per_book_per_day`** reported weekly.
18. ~~Kill switch: 8 review weeks with CLV CI ∋ 0~~ **superseded by 28** — kill switch is futility
    (CLV CI upper bound < +0.5¢ at a minimum n), §4.
19. **Phase 0 = two sessions**; credit budget + gate-funnel simulation are its first deliverables;
    daily raw Torvik/ESPN archive; dry-run week in late October.
20. **No new execution outlets in year 1** — Franz's terms/legality call, not the plan's.
21. **Capture cadence is tip-scaled** (60/30/15 min), watch ticks are credit-free; **1H
    team-foul deviation** enters the 2H carry as a fitted term; the shadow prior set is four
    variants including a brand-tier-halved weight — the hand-set weight is never forced down
    by hand, the week-5 shadow table decides.
22. **Model layer (review 2026-10-12): anchor the mean, own the shape.** Consensus spread/total
    is the mean; the model fits shape only (variance, skew, key numbers, margin–total
    correlation, halves, team-total split). Two-stage simulator (to the ~4:00 checkpoint, then
    the foul sub-model on the checkpoint margin), validated on the historical (margin, total)
    joint by spread bucket; halves as a correlated pair with an explicit halftime-tie term;
    Kalman observations lineup-adjusted and possession-weighted; a joint time-decayed ridge
    raced against the Kalman; multiplicative / capped-margin / pace-mismatch rungs; a second
    ladder inside the distribution scored by log score / CRPS; PIT / tail / key-number
    calibration; push-mass matrices for spreads too; variance fitted on outcomes with the market
    relation as regularizer; shape parameters shrunk to conference / tier; totals and team
    totals first, sides research_only until the ladder earns them; margin-conditional foul bump
    only if PBP is unverified.
23. **Audit as a feedback loop, bounded.** The week-5 shadow-prior pick is automated inside the
    review pass (written once to versioned params); the ESPN fallback is graded against Torvik
    daily (table, not a rule); `w` moves only by review-cadence refits — no trailing-window
    auto-tuning; ~~a 25u-from-peak / 15u-in-14-days drawdown pause~~ **superseded by 28** (pause in
    SDs of actual bet variance + CLV deterioration);
    the consensus-validation study is re-run every August.
24. **Third model review (2026-10-12): location-space blend, calibrated anchor, shape first.**
    Two `w`s (margin, total) shift the means, one simulation prices every market; the anchor is
    the devigged median, matched by an explicit solve step; shape level fitted on close-centred
    seasons, Torvik history for relative structure only; inner rung 0 is stated as
    Normal-around-the-close; 2026-27 live is the frozen holdout with Holm/FDR across all rungs;
    alt lines capped at ~±1 SD; smoothed push matrices; diffusion model as a rung; lineup
    mixture is shape + alarm, not a mean-shifter; a model-free derivative-consistency study runs
    before shape is built; **prior.py, player posteriors and minutes move to Phase 1b, gated on
    `w > 0` being worth testing** — the futures capture in Phase 0 stays.
25. **Game-projection review (2026-10-12).** v0 ships first (consensus-centred distribution +
    push mass + foul bump); `p_model` is one distribution object with per-pass coherence tests;
    deterministic seeds + CRN + path count tied to tier thresholds; parameter-uncertainty spread
    must be exceeded before tiering; year-1 availability = points-lost-per-player for
    high-majors with Beta priors per status × tier, player value raced vs market reaction;
    `pending` defined (no move > ½ pt / 5¢ over two passes); situational terms as one
    regularized model with a published MDE per term, pace/variance first.
26. **Derivative study has an outcomes leg** (rung 0 vs realised density on 2024-25 closes;
    build where outcomes disagree with the mechanical translation); push smoother carries an
    estimated integer/parity spike set; `AMBIGUOUS` status is a fitted stage-1 variance
    multiplier feeding our derivative prices and `pending`, not the market-only channel.
27. **Fourth review (2026-10-12).** Torvik `super_sked` projection is the year-1 rating rung
    (after the as-of check); v0 + generic shape frozen before the first slate, derivatives in
    shadow until ≥ 600 derivative closes, rejected/deferred rungs shadow-run live; frozen vs
    adaptive table (§4.z); `pending` polled on featured markets (3 credits/pass) every 10 min
    inside T-3h; solve identified by fixing possessions; seed keyed to `(event_id,
    model_version)`; SD per quality bucket; inner-ladder MDEs in log-score/CRPS units; `k`
    = 1.0 in `preregistered.yaml`; derivative benchmark = main-line conversion with outcome
    scoring co-primary; event-window / control-game design for market-reaction player values;
    wide prior on "fraction already priced" for rare terms; consistency pass (§0, §3.4, §3.5,
    §3.6, §3.7); `year1_scope.md` is the builders' sheet; Phase 0b go/no-go checklist.
28. **Audit review (2026-10-12): scorekeeping → learning.** Ledger row is a full decision
    snapshot (provenance / decision / context / execution); placed bets entered via
    `cbb-engine placed` and reconciled; fixed versioned post-game tag taxonomy with
    loss-share vs base-rate, PIT + surprise score, Monday digest; luck-adjusted P&L, edge-bin
    calibration, tier monotonicity, component attribution, stratified gate value; kill
    switch = futility (CI upper < +0.5¢ at min n, MDE printed); drawdown pause in SDs for
    the actual bet count **and** CLV deterioration; anytime-valid monitoring for stops only;
    append-only corrections + hypothesis registry/change log; page-one scorecard; Parquet +
    DuckDB quote archive, hash-chained ledger; fav/dog, road-fav/away-dog, rest,
    conference cuts standard.
29. **Audit review 2 (2026-10-12): phantom CLV and replayability.** Market-only hypothesis
    judged on fill-verified rows / fill haircut; opportunity persistence is a scorecard light;
    ~10 dense historical slates bought pre-launch (on go-ahead) for CLV SD + stale persistence;
    rows carry `edge_by_rung`, `code_sha`, `env_lock_hash`, input hashes, stored pmfs; `placed`
    Y/N/unknown with reason codes, auto-stamped, one-tap entry; side-symmetric tags, losses vs
    wins with FDR, tag → action map; trend test for tiers; challenger-vs-champion primary,
    ≤ 2 structural changes per review; grey "underpowered" lights; structured YAML registry
    with exact formulas; per-pass encrypted Parquet shards off-site + manifest in git; chain
    head anchored externally; decisions 18/23 marked superseded.
30. **ML scope (2026-10-12).** No boosted-tree or logistic outcome model as the engine; ML only
    as §3.8 rungs — logistic for recalibration, blend diagnostic, P(plays), fill probability;
    GBM/GAM for closing-line movement, Torvik-history shape targets, candidate CLV scoring —
    shadow-run with a linear baseline, outputs into the distribution object, SHAP into
    `edge_by_rung`. Not for W/L or ATS, ratings/anchor replacement, props, or buy-only training.
31. **Integration (2026-10-12, option b).** CBB is built on a new `engine_runtime/` package (one
    Odds client, Parquet/DuckDB quote store, power de-vig + execution-excluded median, NBA-shaped
    ledger row with the §4.a snapshot, pass runner); the five live engines keep their pipelines;
    the only cross-sport change is a shared fail-open daily credit ledger with the CBB
    featured-pass reserve as a budget line; existing audit emails are re-led by CLV as a
    reorder only. Details: `docs/integration_master_plan.md`.
32. **Fifth projection review (2026-10-12): pin the mean harder.** Width = `final − close`
    residuals per bucket, never close dispersion; buckets from pre-close information only; foul
    bump / tempo / OT are redistributions with a 0.1-pt median tolerance, miss ⇒ graded refusal;
    `w` prior centred at 0, missing Torvik skips the blend, ESPN rating report-only; "all weights
    zero" is the fifth week-5 prior candidate; path cap ⇒ graded refusal; 1H share and
    correlation from residuals conditional on the close; push/parity realised-vs-predicted table
    by number (audit cut 9); stale flag never enters `p_model`; availability shift applied
    after the solve on the un-moved residual, race kept, event count published.
33. **Consistency pass (2026-10-12).** Scope-sheet Ships table rewritten to the pinned-mean
    spec; week 5 scores all five priors with all-zero allowed to win and no futures weight on a
    live price before the test; the availability race stays in the report and moves the median
    only past its pre-registered event count and MDE; chart box 2 (prior / ratings / player
    posteriors) drawn as Phase 1b, worksheet / totals sheet / MP3 as Phase 4, watch 5 min on ESPN/RotoWire, featured pass 10 min; audit cut 10 = 10-min featured passes vs 15-min CLV, the test that justifies the
    reserve.
34. **Record = analysis (2026-10-12).** `status` ∈ {live, shadow, research} on every row, promotion
    test on live `bet_of_record` only; one row grain `(event_id, market, side, line, book,
    pass_id)` in one store, quotes in shards, row stores the numbers used + shard ids; close
    passes archive-only, the card pass is the bet of record; one cadence table (60/30/15 board,
    10-min featured inside T-3h, 5-min watch on ESPN/RotoWire with no credits) and `pass_kind`
    on every shard; `anchor_miss`, `se_cap` and the four coherence names are gates in the funnel;
    main-line Brier demoted to a coherence column, CRPS + cut 9 + shadow-derivative Brier are the
    movable scores; 03:00 report = `v_clv v_shape v_rung v_gate v_persist v_prior` DuckDB views,
    PDF a rendering, page-one lights first; §4.a Identity group and the field table added; rung
    list fixed and ordered in `preregistered.yaml`.
35. **Sixth review (2026-10-12): leftovers that would still move a number.** One v0 definition
    (residual width, fitted key-number masses); one push mass — the row's, used by `p_model`,
    tier and `ev.py` alike; week-5 null kept unless a positive weight beats it by its MDE;
    Torvik staleness stops the model channel only (capture, card and market-only continue);
    Phase-0 budget computed from the cadence table; §7 rewritten to `engine_runtime/` + thin
    `cbb_engine/`. Doc fixes: §2 status cells for 1H rows; totals-juice live band, dog cap to
    research; carry terms off in year 1; joint check on 2024-25 close residuals; capture-gap
    alarm per `pass_kind`; off-site shard not git mirror; watch tick ESPN/RotoWire only;
    promotion hypotheses split live vs `v_prior`/research; §4.e persistence split next featured
    pass / next card; criterion 3 rewritten; worksheet / totals / MP3 marked Phase 4 in §5/§6.

Still needed from you: confirmation of which Odds API key holds the credit balance, and the
go-ahead to spend ~40–60k credits on the 2024-25 closes.
