# NHL Engine — Outputs & Audit Master Plan

Companion to `nhl_engine_master_plan.md` (model/market/phases). This document
covers the two reader-facing halves: **(A) the daily slate package** — the PDF
card and its sidecars — and **(B) the audit** — the ledger, CLV, probation and the
graded report. Both are built to the conventions the MLB/CFB/NFL packages already
follow, so a reader who knows the CFB card or the MLB Audit Desk knows this one.

Conventions inherited unchanged:

- **Nothing in the brief moves a price.** The pipeline scores (or deliberately
  refuses to score) every fact; the brief is what a reader needs to *parse* the
  card (`cfb_engine/output/brief.py` rule).
- **Every card can be rebuilt offline.** `GameBrief` and `GamePreview` persist on
  each recommendation in `predictions_<date>.json`, so `nhl-engine card` re-renders
  without a network or a re-sim.
- **Labels are earned.** A word like "Buy", "Play", "regressing" appears only
  when the ledger has graded that bucket at the probation bar; until then the
  word is "Watch" (power-screen PR #372/#373 rule).
- **House style**: Georgia serif, navy/red masthead, A4 via WeasyPrint, MP3
  narration via edge-tts, one email per package via the Gmail app password.

---

# Part A — Daily slate package

## A.1 Package inventory (what lands in the inbox)

| Artifact | File | When | Analogue |
|---|---|---|---|
| Bet card workbook | `nhl_recommendations_<date>.xlsx` | 11:00 card, re-issued 17:00 with goalie confirms | MLB/CFB `excel.py` |
| **Slate article** (PDF + HTML + MP3) | `PayoffPitch_NHL_Slate_<date>.pdf/.mp3` | 11:00 and 17:00 | CFB `card.py` + `brief.py`, MLB `daily_preview.py` |
| **Regression Report** — skaters & goalies | `PayoffPitch_NHL_Regression_<date>.pdf/.mp3` | once daily | MLB `regression_article.py` (arms + bats) |
| **Crease Report** — goalie stat cards | `PayoffPitch_NHL_Crease_<date>.pdf` | 17:00 (after confirms) | MLB `PayoffPitch_Mound` cards |
| **Unit Rankings worksheet** | `nhl_worksheet_<date>.xlsx/.pdf` | once daily | MLB `daily_worksheet.py` (+2/−2 hand sheet) |
| Injury & lineup sheet | inside the slate PDF + `lineups_<date>.csv` | 17:00 | CFB `_out_line`, NFL injuries |
| Props research sheet (not a card) | `props_research_<date>.csv` | 17:00 | NFL `props.py` (`research_only`) |
| Yesterday's audit summary (text in body) | from Part B | 03:00 email | MLB `totals_audit` body text |

Two passes a day is the minimum: NHL starters are confirmed at ~17:00 ET and the
MLB engine learned (PR #327, four passes) that a card priced before the lineup is
a different card. The 11:00 card is labelled **"projected goalies"** in its
masthead; the 17:00 card is the one the audit grades (`predictions_<date>.json`
is write-once from the 17:00 pass; the 11:00 pass writes `…pregame_am.json` for
the "did the goalie move the line" study).

## A.2 Slate article — page structure

### Front page
1. **Masthead**: date, games count, first/last puck drop (ET), which pass (AM
   projected / PM confirmed), Odds API credits remaining, pipeline version.
2. **Slate best bets block** (`_slate_best_block`): every Strong/Moderate buy
   across games, ranked by edge, with model %, market %, edge, EV, book, price,
   `pass_gate` for refused near-misses in grey ("refused: goalie_unconfirmed").
   Blank-state text when nothing clears ("Nothing on the board clears 2 points
   of edge tonight; the market and the model agree within noise").
3. **Slate at a glance table**: one row per game — time, away @ home, confirmed
   goalies (✓/~/?), market ML / PL / total, model ML, exp goals, game shape tag
   (see A.3), rest flags (B2B, 3-in-4), our lean.
4. **Regression one-screen cheat sheet** (from the Regression Report): 5 buy-low
   skaters, 5 sell-high skaters, 3 goalies each way — names only, details inside.

### Per-game section (one per game, in puck-drop order)
Each block, in order, each sentence emitted only when the data exists:

1. **Header + market line**: `TOR @ BOS · 7:00 ET · TD Garden` then the board:
   best ML each side, main PL ±1.5 with prices, main total with prices, and the
   *period-1* 3-way ML and total beneath it in smaller type.
2. **Game shape tag** (`_game_shape`): `Coin flip / Lean / Clear favourite` ×
   `Low-event / Average / Track meet` from exp margin and exp total, with the
   sim's OT probability ("28% to go past regulation").
3. **Team table** (two rows): record, points%, streak, last result, **5v5 xGF%**
   with rank, xGF/60 rank, xGA/60 rank, PP% rank, PK% rank, PDO (flagged when
   outside .985–1.015 as "running hot/cold"), goals-above-expected last 10,
   MoneyPuck pregame win % (outside benchmark, like FPI on the CFB card).
4. **The take** (`_take`): 5–8 plain sentences — who visits whom, last time out,
   the widest unit mismatch (A.4), the goalie duel in one line, injuries in one
   line, "Our sim has BOS by 0.4 goals with about 6.1 total; MoneyPuck 58%",
   moneyline model-vs-market gap sentence (≥3 pts either way, or "within noise").
5. **Goalie duel** (the headline matchup; hockey's starter duel):
   - Each starter: status (`confirmed 16:42 ET` / `probable` / `projected —
     backup likely, B2B`), season Sv%, **GSAx/60 shrunk** with the unshrunk beside
     it, last 5 starts GSAx, workload (starts in last 7 days), career vs opponent
     (n shown, never a verdict under 5 starts), high-danger Sv%.
   - Backup line if the starter is unconfirmed: who it would be and what the
     swing is worth in the sim ("backup costs 4.1 pts of win prob").
   - **Cold-start / call-up flag**: days since last NHL start when >14
     ("first start in 23 days"), and `call-up: N NHL minutes` when under the
     `MIN_GAMES_PRIOR` floor, with the anchored expected Sv% the sim used shown
     beside the tiny observed sample so the reader sees why the price moved.
   - Verdict phrasing borrowed from `starter_duel`: "clear edge / edge / even",
     sized on the *shrunk* gap only.
6. **Player matchups** (A.5): top line vs top pair, PP1 vs PK, the
   deployment note (who the home coach can match), 2–4 sentences.
7. **Regression bits** (`_reg_bits`): the skaters and goalie in this game who
   appear in the Regression Report, one clause each ("Marner: 9 G on 5.1 xG —
   selling high; Nylander 3 G on 7.4 xG — buy low").
8. **Injuries / lineup**: out and day-to-day per team with role (`1C`, `1D`,
   `PP1`, `G1`), the replacement where known, and the tag CFB uses: `[already
   priced in]` if the absence predates the line, `[reported, not scored]` if new
   and unscored, `-x.x` if the sim charged it.
9. **Unit ranking edge** (A.4): a one-line sentence naming the biggest gap in the
   +2/−2 sheet, e.g. "BOS's PK (+2) against TOR's PP (−1) is the widest gap".
10. **Venue/schedule**: rest days each side, B2B / 3-in-4 flags, travel miles and
    time zones crossed, altitude (COL/UTA), afternoon start, "first game back
    from a 6-game trip" — reported as context, priced only where the study says
    (master plan §5.4).
11. **Period read**: the sim's period-1 3-way probabilities and P1 total lean,
    with the note that period markets are the thinnest on the board.
12. **Sharp money line** (`_sharp_line`): VSiN handle vs tickets each side when
    posted; the ML gate's verdict.
13. **Best bets for this game** (bold), then greyed refused rows with their gate.

### Back matter
- **Methodology footer**: what moves a price, what does not; the market-anchor
  weight in force; the probation verdicts currently shutting a market.
- **Fine print**: "model preview, not betting advice"; data timestamps
  (MoneyPuck file date, NHL API pull time, odds pull time, goalie source).

## A.3 Game-shape tags (thresholds to fit in Phase 1, provisional values)
`|exp margin| < 0.25` coin flip, `< 0.6` lean, else clear favourite. `exp total
< 5.6` low-event, `> 6.6` track meet. OT probability printed always; "shootout
watch" when regulation-tie probability > 26%.

## A.4 Unit rankings worksheet (the +2/−2 hand sheet, hockey edition)
Same rule as the MLB worksheet: per metric, best 3 teams +2, worst 3 −2, everyone
else +1, each metric read on the window where it is most reliable (Phase 1
reliability study sets the window per metric).

| Unit | Metrics |
|---|---|
| **5v5 offense** | xGF/60, CF/60, HDCF/60, shooting% (long window, shrunk), rush chances/60, zone-entry carry-in % if available |
| **5v5 defense** | xGA/60, CA/60, HDCA/60, shots-against quality, blocked-shot rate |
| **Power play** | PP xGF/60, PP%, PPO drawn/60, PP shot rate, PP1 TOI share |
| **Penalty kill** | PK xGA/60, PK%, PIM taken/60, PK shot suppression |
| **Goaltending** | GSAx/60 shrunk, HD Sv%, rebound control (rebounds/60 allowed), starter workload |
| **Special-teams net** | PP xGF/60 − PK xGA/60 |

Per game: a row per team — 5v5 O vs opp 5v5 D, PP vs opp PK, PK vs opp PP,
goalie — weighted (5v5 ×3, PP ×1, PK ×1, goalie ×2; weights to be fitted like the
MLB starter-share weighting), away-minus-home disparity, slate ranked by
disparity. Written to `worksheet_ledger.csv` with the best ML/PL price and VSiN
splits at the time, graded the next morning by gap band against the market's
implied rate (Part B.6). **Nothing here feeds a price. It is a record.**

## A.5 Player matchups section (data → sentences)
- **Where injuries and lineups are parsed** (same keyless, timestamped pattern
  as `nfl_engine/data/injuries.py` / `mlb_engine/data/rotowire.py`):
  - Injuries: RotoWire hockey injury table (designation, player, team) paired
    with the RotoWire NHL RSS for posting time; NHL API roster/boxscore scratches
    as the official late cross-check. Every capture writes `(source, posted_at,
    seen_at, status)` to the availability log.
  - Goalies: RotoWire NHL lineups page (Confirmed/Expected/Unconfirmed) as the
    primary, DailyFaceoff as a second vote (disagreement → `projected`), NHL API
    pre-game roster and boxscore starter as ground truth. Manual CSV overrides.
  - Lines: derived from NHL API shift charts (below); RotoWire/DailyFaceoff
    projected lines override when they differ.
  - All scrapes ship with saved-HTML fixture tests and a blank state ("goalies
    unconfirmed", "no injury report captured") — never a crash, never a guess.
- **Lines & pairs**: from the previous game's shift chart (NHL API) — TOI-together
  clustering yields L1–L4, D1–D3, PP1/PP2 — overridden by a lines CSV drop-in
  when present (DailyFaceoff-style, manual, like FanGraphs exports).
- **Top line vs top pair**: L1 xGF/60 on ice vs opp D1 xGA/60 on ice, home coach
  gets the matchup note ("BOS at home can hard-match McAvoy–Lindholm to Matthews").
- **PP1 vs PK**: PP1 unit xGF/60 vs opp PK xGA/60; flag if a PP1 regular is out.
- **Individual watch**: 2 skaters per team by (i) shots/60 × projected TOI —
  the SOG prop candidate, (ii) largest G−xG gap (regression), (iii) point streak
  (context only, streaks are not priced). Each printed with the market's SOG /
  points line beside our projection — projection, not a buy, while props are
  `research_only`.

## A.6 Regression Report (skaters + goalies), in the MLB article's voice
Sections mirror "Which Arms Are Lying to You Tonight":
1. **Finishers the goal column hasn't caught up to** — G well below ixG (buy-low),
   shot volume intact.
2. **Goal totals borrowing against the future** — G far above ixG on ordinary
   shot rates; sell-high, fade anytime-scorer prices.
3. **Playmakers the assist column is hiding** — on-ice xGF high, A low; and the
   reverse (secondary-assist luck).
4. **Goalies the box score is slandering** — GSAx negative but HD Sv% and shot
   quality faced say unlucky; **and the mirage** — Sv% .930 on a .905 expected.
5. **Teams**: PDO extremes with the both-way read (shooting% and Sv% separately).
6. **One-screen cheat sheet** + the small-sample footer (under N shots/N starts
   is "directional").
Every claim carries the numbers (G, ixG, shots, window) and the reliability
caveat the study produced; the article never recommends a price.

## A.7 Narration (MP3)
Same `_narration` pattern: slate lede, per-game 2–3 spoken sentences (goalies
first — that is what a hockey bettor asks first), best bets, sign-off. Regression
article gets its own narration.

## A.8 Delivery & schedule
`scripts/nhl/{macos,linux}/install_schedule.*`: 11:00 `run --card --email`
(projected), 17:00 `run --card --email` (confirmed; also props/period capture),
`close` passes at 18:30/19:30/20:30/21:30/22:30 ET pregame-only per game, 03:00
`audit --report --email`. Regression/Crease/worksheet ride with the 17:00 pass
via `email_daily_package --sport nhl`. Weekend matinees: schedule reads the NHL
API and adds a 12:00 close pass when any game starts before 16:00.

## A.9 Build order (sessions)
1. `brief.py` + `card.py` for game markets only, front page + per-game §1–4, 8,
   10, 13 (1 session, lands with Phase 2 of the master plan).
2. Goalie duel + Crease Report + injury/lineup sheet (1 session; needs
   `data/goalies.py`).
3. Unit rankings worksheet + ledger (1 session).
4. Regression Report + cheat sheet + narration (1 session).
5. Player matchups & period read (0.5 session, after props archive exists).

---

# Part B — Audit (the receipt)

## B.1 What the audit is for (unchanged from MLB/CFB)
Win rate says whether the pick was right; **Needs %** says what the price
charged; **CLV** says whether the price was good, in dozens of bets instead of
thousands; **probation** says when a market or screen has earned a change. The
NHL audit produces all four every night and *changes nothing by itself* — a
condemned market is shut by a config change quoting the verdict.

## B.2 Grading rules (the hockey-specific part)
Every ledger row carries `ot_rule` and grades against the NHL API final with
per-period scores and the OT/SO flag:

| Market | Grades on | Notes |
|---|---|---|
| `game_ml` (2-way) | final incl. OT/SO | |
| `game_ml3` (regulation 3-way) | regulation score | tie leg wins on any OT game |
| `game_pl` ±1.5 | final incl. OT/SO (OT win = 1-goal margin; SO win recorded as 1 goal) | book rule verified per book in Phase 0 |
| `game_total`, `team_total` | final incl. OT/SO | SO adds exactly one goal to the winner |
| `p1_*`, `p2_*` | that period's goals | |
| `p3_*` | 3rd period regulation only | confirm per book; EN goals count |
| props | official boxscore (SOG, G, A, PTS, BLK, saves, PPP) | `research_only` rows graded into `props_graded_<date>.csv`, never the ledger (NFL rule) |

A game that is not final, postponed, or missing a period line is "a short record"
— the slate is not graded until every game the card priced is final (PR #357
rule), and a re-audit replaces that date's rows.

## B.3 Ledger (`~/.nhl_engine/audit/ledger.csv`, synced to `engine-state` under `nhl/`)
`LedgerEntry` = CFB's plus: `period`, `ot_rule`, `goalie_status_at_bet`
(confirmed/probable/projected), `goalie_home`, `goalie_away`, `rest_flag`,
`sharp_div`, `pass_gate`, `drift`, `close_odds`, `close_prob`, `clv`, `clv_ev`,
`clv_pts` (for PL/totals moved in the number: ±1.5 rarely moves, totals move in
halves — converted at the local slope of the goal distribution, the CFB
`linevalue` idea with a Poisson slope), `source` (`engine` | `moneypuck` |
`teamrankings` for benchmark rows, never mixed into engine metrics).

Workbook `ledger.xlsx` sheets: **Overall** (engine p≥.5 row, tiers, Buy S+M,
Needs %, ROI, units) · **Daily PPV-NPV** · **By market** (ML, ML3, PL, totals,
team totals, P1/P2/P3 each market) · **Period sheet** (period markets alone —
they are thin and the first candidates for shutting) · **Goalie sheet** (buys by
goalie status at bet time: the question "does betting before confirmation cost
us" answered in the ledger, not the argument) · **Rest sheet** (buys by B2B /
3-in-4 flag) · **Closing Line Value** · **Price buckets** (plus/minus money, six
bands, gap vs need, ≥15 rows) · **Screens** (probation table: live screens,
candidates) · **Funnel** (candidates → priced → +EV → price screen → gates →
bought, per market, first-failing gate named) · **Bets**.

## B.4 CLV (`close` passes, pregame only)
Staggered puck drops mean one close does not fit a slate: `close` runs every
hour 18:30–22:30 ET and captures **only games that have not started** (CFB PR
#351/#377: in-play quotes purged, `repair-closes` command ported). The audit uses
the last pregame quote per game. CLV per market on the sheet plus the two
hockey-specific cuts: CLV by goalie status at bet time, and CLV on the 11:00 card
vs the 17:00 card for the same selections (the number that decides whether the AM
card should exist).

## B.5 Probation (ported verbatim, NHL parameters)
`min_n` 100, mean return worse than zero by >1 s.e., both halves of the window
agree. NHL volume: ~6–8 games/night → 20–40 buys a night across markets, so a
market reaches 100 in 1–3 weeks. Live screens graded from day one:
`goalie_unconfirmed`, `price_band_<market>`, `drift_gate`, `ml_no_sharp_money`,
`one_way_quote`, `max_edge`. Registered **candidates** (graded on the buys they
*would* refuse, never acting): B2B-second-night fade, period-1 under band,
market anchor 0.5 on ML, PL −1.5 ceiling at +150, "confirmed-goalie only".
Verdicts: `WATCHING / HOLD / SHUT (market) / RETIRE (screen) / SHIP (candidate)`.

## B.6 Sidecar audits (each "a record, not a price")
- **Worksheet audit**: the +2/−2 gap by band vs the market's implied rate (A.4),
  `worksheet_ledger.csv`, rows stamped with the band version.
- **Props research grade** (`props_grade`): Brier of model vs de-vigged fair vs
  base rate on every archived line, shadow return at captured prices on rows only
  `research_only` stopped, per market, cumulative — the bar that decides whether
  SOG ever opens.
- **Goalie latency log** (`availability.py`): first sighting of a confirmation vs
  the ML at that moment and at close — "did we hear it before the number moved".
- **Outside benchmark**: MoneyPuck pregame probabilities and TeamRankings picks
  graded on their own rows (`source≠engine`), AUC/Brier beside ours per market.
- **Dispersion check**: nightly, the sim's implied ML↔PL↔total relationships vs
  the market's on that board (RMSE) — the L13 guard, trended weekly.

## B.7 Audit report (daily + weekly PDF/HTML/MP3)
Same layout as the MLB Audit Desk sample, with hockey rows:
1. **Executive summary**: whole-engine PPV vs Needs %, ROI, CLV mean, one
   sentence on where the leaks are — *generated from thresholds, no adjectives the
   numbers do not earn* (PR #378: no recommendations on noise; NPV blank when
   nothing was faded).
2. **Core metrics** table (whole engine, Strong, Moderate, Buy).
3. **Market scorecard**: every market incl. period markets, PPV/NPV/ROI/CLV,
   Needs %, verdict `Play / Neutral / Fade` by the fixed rules, `Min p to Play`;
   markets under 5 favoured picks stay Neutral.
4. **Price buckets** over the whole ledger with slate count.
5. **CLV section**: per market, by goalie status, AM vs PM card.
6. **Probation table**: live screens + candidates with verdicts.
7. **Hockey diagnostics**: OT rate predicted vs realised; empty-net pulls and
   goals predicted vs realised, by coach (the dynamic EN model's receipt; alt
   puck lines and 3rd-period totals depend on it); PL push/1-goal-game rate vs
   sim; period scoring shares vs sim; goalie-swing realised (games where the
   starter changed after the AM card); **cold-start ledger cut** — buys and
   fades involving a goalie with TSLS >14d or under the call-up floor, graded
   apart, so the penalty's size is refit from evidence; PK-fatigue candidate
   verdict (from the probation table).
8. **Most common errors** (rule-based, e.g. "dogs bought at +150 and longer won
   31% needing 40%"), **Recommendations** mapped to ↑PPV / ↑NPV / ✕FP / ↓FN, only
   when a bucket clears n≥15 and 3 pts.
9. Funnel for the graded slate; fine print with data timestamps.

## B.8 Commands
`nhl-engine audit [--date] [--report] [--email]` · `report --period daily|weekly`
· `probation` · `scorecard` · `calibrate --holdout 2` · `repair-closes` ·
`latency` (goalie) · `props-grade` · `worksheet-audit`. All pull `engine-state`
first and push after; `NHLE_DATA_DIR`/`NHLE_STATE_SYNC=0` scratch rails.

## B.9 Build order (sessions)
1. `grade.py` with `ot_rule` + per-period finals, `ledger.py`, `scorecard.py`,
   `clv.py` (pregame-only, multi-pass), `state.py` — lands with master plan Phase 2
   (1 session).
2. `probation.py`, `priced.py`, `funnel.py`, price buckets, `audit_report.py`,
   email (1 session).
3. Sidecars: goalie latency, worksheet audit, dispersion check, outside benchmark
   (1 session).
4. Props grade (with master plan Phase 3, 0.5 session).

Tests: one per grading rule (OT win on PL, SO goal on total, 3rd-period EN
goal, tie leg on ML3, postponed game = short record), CLV multi-pass selection,
probation half-window consistency, report blank-states.
