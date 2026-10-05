# NBA Payoff Engine — Master Plan v3.6

Fifth engine in `payoff-pitch-` (`nba_engine/`, beside `mlb_engine`, `cfb_engine`, `nfl_engine`,
`nhl_engine`). **Paper-only** until each market passes probation (§11). Studies live in
`scripts/nba/`, never in production scoring.

Labels: **[MEASURED]** our ledgers/studies · **[VERIFIED]** checked from the VM · **[HYPOTHESIS]**
must be studied before it prices · **[DEFAULT]** provisional, refit before shipping ·
**[FIT]** the value comes out of a Phase 1 study; nothing is typed in by hand.

v3 adds five NBA-specific adjustments: a news alarm (§5d), on/off backup profiles (§4.2a),
team-constrained props (§4.5a), a blowout dimmer (§4.4a) and nightly team ratings (§4.8).
v3.1 hardens each: a settle gate on the alarm, an opponent-quality correction on star-off
rates, usage/shot ceilings on the split, probabilistic blowout pulls, and fatigue-weighted
nightly updates. v3.2 adds four: a next-man-up hierarchy for vacated usage (§4.5a), the close
taken at T-1 instead of T-5 (§1, §10), an Over-bias term on props (§4.7), and pricing a
questionable player as a probability instead of banning the game (§5b). v3.3 routes minutes
lost to foul trouble down the same next-man-up chain (§4.4), groups players by fitted playstyle
family instead of position (§4.5a), tests the Over bias after late news (§4.7), maps injury
reasons to fixed categories (§5b), and adds a teams + tip-time fallback match for event IDs (§1). v3.4 tags every off-floor stint by
why the player sat and keeps foul-trouble and blowout stints out of the baseline profiles
(§4.2a), takes ceilings from role-expanded games (§4.5a), measures the 1H/2H scoring gap (§4.4),
alerts on new injury wording (§5b), flags NBA Cup games (§1), drops pulled props from CLV (§10)
and keeps fitted values in a versioned params file instead of `config.py` (§13). v3.5 fits the
edge floor separately for one-book and two-book rows (§10), makes ceilings per-minute and
family-specific (§4.5a), splits the 1H/2H gap into pace and efficiency (§4.4), and keeps pulled
props in the record with their own pre-pull CLV (§10). v3.6 adds a rule-change register (§3a):
the 2025-26 heave rule, the 2026-27 push-off emphasis and the 2026-27 Cup venues. Every number quoted as
an example below is a starting hypothesis; the shipped value is fitted.

---

## 0. The short version

The four engines' ledgers say our ratings do **not** beat the closing line on main markets; the
money is made or lost on selection, price and timing. So the NBA engine is **market-anchored**:
the no-vig close is the baseline, the model has to beat it, and every filter must earn its place
against the close or ship at zero. Three seasons (2023-24 → 2025-26) of real closes — game lines,
**first halves and player props** — come from The Odds API historical endpoints (approved; the
key reports 5,000,000 credits [VERIFIED]), so the first backtest runs before opening night.

---

## 1. Databases — what each one feeds

| # | Database | Seasons | Feeds | Status |
|---|---|---|---|---|
| D1 | **The Odds API historical featured odds** | 2020-21 → 2025-26 (5-min snapshots since Sep 2022) | Game ML / spread / total: open, every 5 min, **close = last pre-tip snapshot, requested at T-1** (the T-5 request returned a snapshot 9.4 min before tip [MEASURED]); line-movement filter history | [VERIFIED] paid plan |
| D2 | **The Odds API historical event odds** | since 2023-05-03 | **1H ML / spread / total**, props **PTS, 3PM, REB, AST, PRA** — open → close | [VERIFIED] docs |
| D3 | The Odds API live (`basketball_nba`) | from Phase 0 | Every pass's prices, closes, line movement; the forward ledger | key live |
| D4 | SportsbookReviewsOnline | 2007-08 → 2022-23 | Long-run schedule/rest/travel studies vs the close; 1H lines | [VERIFIED] |
| D5 | Covers.com team results | 2023-24 → 2025-26 | Free cross-check of D1 closing spread/total (no prices) | [VERIFIED] 2023-24 |
| D6 | **Kaggle — Wyatt Walsh NBA DB** (SQLite, CC BY-SA, updated 2026-06-29) | 1946 → 2025-26 | Box scores, team/player game logs: priors, reliability (split-half) fits, minutes model | [VERIFIED] metadata; needs token or Mac download |
| D7 | `shufinskiy/nba_data` + pbpstats | 2000s → 2025-26 | Play-by-play, possessions, lineup stints: pace, RAPM, quarter/half scoring, shot quality | [VERIFIED] |
| D8 | **nba_api** (stats.nba.com), run on the Mac | 1996 → | Tracking (shot quality, rim/3 frequency), lineups, advanced box, hustle | blocked on VM; Mac daemon |
| D9 | Basketball-Reference | all | Cross-check of advanced team/player numbers; schedule/arena history | [VERIFIED] |
| D10 | ESPN site API | live | Schedule, scores, box, PBP, injuries; grading feed | [VERIFIED] |
| D11 | **Official NBA injury report PDF** | 2021-22 → | Availability with league timestamp — **published hourly 11 AM–9 PM ET** | [VERIFIED] 2025-03-05 & 2025-12-10 |
| D12 | RotoWire lineups/injuries | live | Projected/confirmed starters, minutes-restriction notes, timestamped | [VERIFIED] |
| D13 | Arena table (lat/lon, altitude, time zone) | static | Travel miles, time-zone shifts, altitude (DEN/UTA) | built once |
| — | BallDontLie, TeamRankings, BetIQ | — | Not used (key-gated / current season only / aggregates) | — |

Every pulled row carries `source`, `captured_at`, `season`, and stable IDs (Odds API `event_id`
+ NBA `game_id` + NBA `player_id`). Names are display only. Each game carries `cup_stage`
(none / group / quarterfinal / semifinal / final). Home court comes from the game's own venue
and ESPN's neutral-site flag, never from the stage: from 2026-27 the semifinals are at the
higher seed's arena (quarterfinals already were), and the 2026 final is at Hinkle Fieldhouse,
Indianapolis, not Las Vegas. Travel and time zones are computed to that venue. Stars played +0.8 min over their own average in the 2023–25
Cup knockouts (126 appearances, 95% −0.6 to +2.0) [MEASURED], and there are only 3 finals. So
the flag is recorded and graded, and the pull curves get no Cup term until one passes a
holdout. The market anchor carries what the model can't fit. If an event ID is missing at a
capture, the game is matched once by both teams' codes and commence time within 15 min, and the
match is logged. NBA Cup knockout games, 2023–25 (21 games) [MEASURED]: every one matched to
ESPN by teams + date and carried all 11 markets. The Cup final does not count in the
standings; it stays in ratings and is flagged in records.

---

## 2. Markets and what is measured on each

| Market | Model output | Settlement |
|---|---|---|
| Game **ML** | P(home win) incl. OT | per book rule |
| Game **ATS** | P(cover) at every line | incl. OT |
| Game **Total** | P(over) at every line | incl. OT |
| **1H ML / ATS / Total** | same, first half | regulation half; 1H ML push on tie per book |
| **Points** | P(over) at line | `dnp_void` |
| **3PM** | P(over) | `dnp_void` |
| **Rebounds** | P(over) | `dnp_void` |
| **Assists** | P(over) | `dnp_void` |
| **PRA** | P(over), from correlated components | `dnp_void` |

**Measured for every market, every slate, kept apart (lesson L2):**

| Layer | Metric | Window |
|---|---|---|
| Model accuracy | Brier, log loss; MAE of margin/total/stat | ledger to date, and per month |
| vs market | Same metric for the **no-vig close** and the **open** on the same rows | same rows |
| Selection | Buys vs model-favoured side vs all rows vs naive baselines (favourite, home, always-Over/Under, always-Under props) | same games |
| Price | **CLV** in probability points and in EV at the close (buys) | per market and price band |
| Result | W-L-P, units, ROI at the price taken | per market, tier, band |
| PPV / NPV | vs the market's base rate (MLB #105) | per market |
| Calibration | Reliability curve, model vs market, on buys | per market |
| Filters | Every filter's refused rows graded: win %, ROI, CLV (the false-negative table) | per filter |
| Integrity | Rows with no model read, stale quotes, one-way quotes, closes after tip (must be 0), void/DNP counts | per slate |

---

## 3. Most predictive metrics, each at its own stabilization window

Rule (user + `engine_common.shrink`): every rate is an **empirical-Bayes posterior**
`rate = (k·prior + n·observed)/(k+n)`, with `k` **fitted from split-half reliability** on D6/D7
(`k_from_split_half`), in the metric's own exposure unit. No L5/L10 rolling windows; no blanket
sample cut-off. Each metric is admitted only if it predicts the **next** games out of sample
**and** adds information beyond the closing line.

| Level | Candidate metric | Exposure unit | Expected stability [HYPOTHESIS, fitted in Phase 1] | Use |
|---|---|---|---|---|
| Team | Net rating per 100 poss (opp-adjusted ridge) | possessions | medium | spread/ML prior |
| Team | Off / Def efficiency per 100 | possessions | medium | spread, total |
| Team | Pace (poss/48) | games | **fast** | totals, 1H totals, props |
| Team | Shot-quality eFG (rim / mid / 3 frequency × league make rates) | FGA | fast | regression filter, totals |
| Team | 3PA rate, FT rate, ORB%, TOV% (Four Factors volume) | FGA / poss | fast | efficiency projection |
| Team | **Actual 3P%, opp 3P%, opp FT%** | 3PA | **slow (mostly noise)** | regression filter only |
| Team | Close-game record vs net rating (Pythagorean residual) | games | ~noise | regression filter only |
| Team | 1H share of points / 1H pace | halves | medium | 1H markets |
| Player | Projected minutes (role, status, rest, blowout risk) | games | fast | every prop |
| Player | Usage rate, FGA & FTA per minute | minutes | fast | Points |
| Player | 3PA per minute | minutes | **fast** | 3PM volume |
| Player | 3P% | 3PA | **slow** — shrink hard to career/league | 3PM efficiency |
| Player | FT%, 2P% at rim | FTA / FGA | medium/slow | Points |
| Player | REB% (ORB%/DRB%) | team rebound chances | fast | Rebounds |
| Player | AST% / potential assists (D8) | team FGM while on floor | fast | Assists |
| Player | RAPM / box prior impact | possessions | slow | lineup rebuild, team strength |

The fitted `k` table (metric, unit, split-half r, `k`, reliability at 20/41/82 games) is the first
Phase 1 deliverable and is quoted in `nba_engine/config.py` docstrings — the NHL
`reliability_study.py` pattern.

---

### 3a. Rule-change register

Every rule or officiating change gets a dated entry. Any rate fitted across that date carries a
pre/post term or is refit on post-change games only, and the walk-forward holdout reports
pre- and post-change games separately.

| Change (source) | From | What it touches | In the engine |
|---|---|---|---|
| **Heave rule:** a shot from 36+ ft on a backcourt play in the last 3 s of Q1–Q3 is a team attempt; a make still counts for the player (NBA.com, 2025-09-10) | 2025-26 | Player 3PA and 3P%; 3PM gets a few extra makes | Player 3PA and 3P% are rebuilt from PBP **excluding heaves in every season**, so pre- and post-rule rates mean the same thing. Heaves are a separate term per player: heave attempts per quarter-end × the measured make rate, ~4% (SportRadar via AP). Their makes add to 3PM and points. No typed deflator and no hand-widened 3PM variance |
| **Push-off emphasis:** "Four F's" off-arm test, plus screen shoves and off-ball holding (ESPN, 2026-09-29) | 2026-27 | Offensive fouls → turnovers; personal fouls → foul-trouble minutes; defensive holding → FTA | Season break: prior seasons get less weight in the foul, turnover and FTA rates, so the nightly update moves to the new level quickly. A player's past offensive-foul rate per touch (PBP) goes into the family clustering and the foul-trouble term. The change in fouls and turnovers is measured by family as 2026-27 games come in, not set at +15–20%. The audit charts league offensive fouls, turnovers and FTA per 100 against 2023–26 weekly, to see whether the spike fades |
| **Cup venues:** semifinals at the higher seed (NBA.com, 2025-09-10) | 2026-27 | Home court and travel in Cup semis | `cup_stage` split (§1); home court from the venue |
| **Replay Center rules on proximate fouls in out-of-bounds challenges** (2025-26) | 2025-26 | Stoppage length only | Nothing to change. The engine never priced replay time as rest, and possessions are unchanged |

## 4. Model pipeline

1. **Season prior** — last season regressed at fitted year-to-year r + roster carry-over (impact ×
   projected minutes) + futures-implied strength (devigged win totals).
2. **Player layer** — EB posteriors (§3) → per-minute rates; minutes model with availability
   state; missing minutes redistributed by a fitted rotation model (not pro rata).

   **2a. On/off backup profiles.** A backup's whole-season rates describe him playing 10 minutes
   next to the starters; they do not describe him running the offence for 35. So every player
   carries rates **split by which teammates are on the floor**, built from lineup stints (D7):
   for each of the team's top-usage players S, the player's per-minute usage, FGA, 3PA, AST
   chances and REB chances in minutes **with S off the floor**. When S is ruled out, the profile
   switches to the S-off rates, each EB-shrunk toward the player's own overall rate by the S-off
   minutes he has actually logged (`k` fitted per stat, §3), not toward the league. Two stars
   out uses the both-off stints where they exist, else the product of the single-off shifts
   (tested on the archive before it prices). Minutes come from the rotation model fitted on
   games where S actually sat, not from the backup's season average.

   *Why he sat.* Every off-floor stint is tagged by why the player was off: normal rotation
   rest, foul trouble (≥2 fouls in Q1 / ≥3 by half / ≥5), injury exit, ejection, blowout (§4.4a),
   or a whole-game absence. Star-off profiles come from whole-game absences plus normal rotation
   stints. Foul-trouble and blowout stints feed only their own tables (§4.4, §4.4a) and never
   the baseline posteriors, because those minutes are played in a different game. Pooling any
   two tags needs the holdout to show they don't differ.

   *Opponent quality.* Star-off minutes are disproportionately logged against opposing bench
   units, so raw star-off rates overstate what a backup does as a starter against a starting
   five. Every stint carries the opponent lineup's quality (its RAPM sum, D7), and the
   star-off rates are re-expressed at the opponent quality he will actually face: the
   correction is a fitted slope of the player's per-minute efficiency on opponent lineup
   quality, pooled by archetype and shrunk, not a flat haircut (a 5–8% cut against a top-10
   defence is the hypothesis to test).
3. **Team strength for the game** = Σ(projected minutes × impact) + shrunk team residual. An
   absence moves the number **only** through this rebuild (no double-count).
4. **Game distribution** — possessions × efficiencies → bivariate (home, away) points, dispersion
   and correlation **fitted to the market's own ML↔spread↔total relation** (CFB #382 lesson);
   1H from **first-half minutes**, not a flat half-share: each player's 1H minutes come from
   his own rotation pattern in the stints (D7), with a fitted foul-trouble term (first-half
   minutes vs personal-foul rate per minute; 0 if it does not survive the holdout; carries the
2026-27 push-off break, §3a).
   *Halves differ [MEASURED].* 2023–26 regular season, ESPN period scores: per team, 1H
   57.1 vs regulation 2H 56.4 (+0.77, 95% +0.49 to +1.04, clustered by game). In games decided by
   under 10 it is still +0.45 (+0.06 to +0.96), so the gap isn't only garbage time. The 1H
   distribution carries it through 1H minutes and two separate fitted terms: a **pace shift**
   (possessions per minute, 1H vs 2H) and an **efficiency shift** (points per possession). The
   ESPN period scores can't separate them, so both come from the PBP possessions (D7). Pace
   moves every counting prop (rebound and assist chances scale with possessions). Efficiency moves
   only points and makes. The 1H distribution is checked against the 1H total closes before 1H
   prices. Minutes a
   starter loses to fouls go to the players who replaced him in his foul-trouble stints, through
   the same next-man-up table and ceilings (§4.5a), not to the whole bench by share. OT from the regulation-tie probability. Every line (main, alt,
   1H) is read off the same distribution; both sides sum to 1.

   **4a. Blowout dimmer.** The game is simulated, not just summarised: per simulated game the
   margin path through the second half sets each player's minutes. Once the margin passes a
   fitted threshold late enough, starters' remaining minutes are pulled and handed to the bench
   at the bench's own rates, and pace and efficiency move to their measured garbage-time
   values, conditioned on **both** benches' quality (a young bench chasing minutes plays a
   different garbage time from veterans running the clock). Garbage-time **assist rate per teammate make and usage dispersion** are fitted
   the same way (bench units may pass less and shoot more on their own); the multipliers come
   from blowout stints, not typed constants like 0.30 / 1.45. The minute curves (starter share of remaining minutes by |margin| × time left), the
   threshold and the pace/efficiency shift are **fitted from real 2019–26 blowouts** in the
   play-by-play stints (D7); none is typed in. The pull is **probabilistic, not a cut-off**:
   at each simulated margin × time-left state the starters are pulled with a fitted
   probability that depends on the coach's own history (some close out games with starters)
   and the schedule (second night of a back-to-back pulls earlier), so some simulated blowouts
   keep the stars in and the upper tail of their props survives. A hard threshold would
   truncate that tail and bias the engine toward Unders. Every prop and the 2H part of the total are read
   off these simulated minutes, so a star's Over carries its blowout risk and a big favourite's
   bench props carry their upside.
5. **Props** — minutes distribution × per-minute rate × pace × opponent factor × teammate-absence
   role shift; negative binomial with fitted variance; PRA from correlated components.

   **5a. Team-constrained props (teammate math).** Player props are drawn **from the team's
   simulated game**, not independently. Each simulated game first sets the team's totals —
   possessions, points, FGA/3PA, made shots (assist chances), and rebound chances (own and
   opponent misses) — then splits them across the players on the floor by their (on/off-adjusted)
   shares: points and shots by usage, rebounds by REB% of the available chances, assists by AST%
   of teammates' made shots. Shares are normalised so the players' rebounds sum to the team's,
   assists never exceed teammates' makes, and points sum to the team score. A player projected
   higher automatically takes from his teammates; PRA, and two props on the same team, inherit
   the real correlation, which the exposure gate (§6) uses to treat them as one bet.

   *Next man up.* An absent player's usage, shots, potential assists and rebound chances go
   first to the **specific teammates who took them** in that player's off-floor stints (D7),
   in that order, each at his star-off share (§4.2a). A team with a backup point guard gives
   the vacated assists to him, not to every player by usage. Each player's vacancy table is
   EB-shrunk toward his playstyle family's table by the star-off minutes behind it.
   *Playstyle families, not positions.* Families are clusters fitted on how players play
   (usage, AST%, 3PA rate, rim/mid/three shot shares, REB%, on-ball time), refit each season.
   A playmaking forward and a defensive wing are in different families even though both are
   listed F. The family stands in wherever the plan says archetype or group.
   *Several out at once.* The vacancy table for an absence set falls back in order: that exact
   set's stints → the single-absence tables combined → the playstyle family's table → plain
   shares, each EB-weighted by the minutes behind it. Every step then goes through the same
   ceilings (archetype p99). It never routes everything to one ball-handler. `allocate()`
   always terminates: each pass freezes at least one player.

   *Ceilings.* Each share is water-filled under a per-player ceiling (usage, FGA, 3PA, AST
   chances) taken from the player's archetype's fitted historical maximum, **measured in
   role-expanded games** (a top-usage teammate out). Green-light nights, when a coach turns a
   role player loose, sit inside the ceiling instead of being cut off at his usual role's p99.
   Ceilings are **per-minute rates** (usage %, FGA/36, 3PA/36, AST chances/36), never totals,
   so a defensive center who plays 38 minutes on an injury night at his usual low usage doesn't
   raise his own ceiling. The role-expanded widening is fitted **per playstyle family**: how
   much a family's p99 usage actually rises when a top-usage teammate sits. A family whose
   usage doesn't rise gets no widening, and nothing is assigned by hand. A player at his
   ceiling is frozen and the rest keeps flowing down the hierarchy, then to the others by share.
   Only the part the stints show **vanishing** reaches team turnovers and lower possession
   efficiency: the drop in team FGA and efficiency when that player sits, measured, not a
   catch-all. When two high-usage players are out and the backups are at their ceilings, the
   remainder is still not forced onto a low-usage defender's shot count
   (`nba_engine.models.allocation`, built and tested on mock numbers in Phase 0).
6. **Calibration** — isotonic per market on a holdout (`engine_common.isotonic`).
7. **Market anchor** — `p_final = w·p_model + (1−w)·p_fair_close_now`, `w` and an edge cap
   **fitted per market**; power de-vig (NFL study), re-measured on NBA props.
   *Hold with a questionable star [MEASURED].* At the 2023–26 close, books' hold with a ≥30-mpg
   star Questionable on the last report ≥1 h pre-tip (508 of 2,670 games) vs without: ML
   4.17% vs 4.18% (−0.01 pts, 95% −0.03 to +0.01); spread 4.63 vs 4.63; total 4.68 vs 4.68;
   1H spread/total +0.07 pts. The ML does not widen, and its hold is the lowest of the six,
   so there is no ML volatility cap and no ATS preference. The EV gate already charges each
   market its own hold.

   **Over bias (props).** The no-vig close **overstates the Over** on all five props. Over all
   books, 2023-26 at the archived close (1,986,631 paired quotes, 3,651 games), Overs hit 48.0%
   against 49.5% implied: −1.6 pts (95% −1.9 to −1.3, bootstrap clustered by game) [MEASURED].
   | | PTS | 3PM | REB | AST | PRA |
   |---|---|---|---|---|---|
   | Over hit − no-vig P(over) | −1.5 | −1.8 | −1.9 | −1.0 | −1.6 |

   It is negative in every season (−2.3 / −1.0 / −1.3) and flat across price levels. At the
   quoted price, Overs returned −9.6% and Unders −3.6%. So `p_fair_close` for a prop Over is
   the de-vigged price **minus a fitted bias per market and line band**, refit monthly. An
   Over has to clear that much more model edge to buy, and Unders get the same amount back.
   It is fitted, not a fixed "Overs are always X% rich". Game totals get the same test before
   any term is used.
   *After late news [MEASURED].* Teammates of a ≥30-mpg star ruled out inside 3 h of tip
   (241 games, 56,470 quotes): Over gap −1.8 pts (95% −3.6 to −0.2), against −1.5 with no such
   news. The book does not over-correct in a way that erases the bias, so there is no
   dial-back after news. The refit may condition the bias on a news flag if the holdout supports
   it; until then the pooled value applies. The archive has no T-30 snapshots, so a "line moved
   in the last 30 min" condition is tested once live capture has recorded it.
8. **Nightly team ratings.** Two update speeds. Team offensive/defensive ratings, pace and the
   player posteriors (§3) update **every morning** from last night's games: each game moves the
   rating by the Kalman/EB weight its possessions earn, so a team that played last night is
   current for tonight. The *formulas* — the `k` table, rating step size, the market blend `w`,
   filter thresholds, minute curves — refit **monthly**, walk-forward only, so one hot week
   cannot rewrite the method. The step size is itself fitted (how fast ratings must move to
   predict next game best on 2019–26), not set to "daily = fast". A game played under a scheduling extreme
   (second night of a back-to-back, 3-in-4, altitude, long road trip) moves the rating **less**:
   the step is multiplied by a fitted fatigue weight per schedule state, since the schedule
   filter (§5b) already prices that game's fatigue and the rating should carry only the
   repeatable part (30% of a normal step on a back-to-back is the hypothesis to test).

---

## 5. The three filters

Each filter writes its verdict on **every** row (`flag_*` columns), refuses through `veto_gate`,
and its refused rows are graded like buys. A filter ships at its **measured** effect or exactly 0;
a filter whose refused rows out-earn its admitted rows is retired (NPV rule).

### 5a. Regression filter (luck vs skill)
| Signal | Computed as | Direction |
|---|---|---|
| Team shooting luck | actual eFG − shot-quality eFG (rolling to date, EB-shrunk) | hot team → fade its side / Over |
| Opponent 3P% luck | opp 3P% − league, on opp 3PA volume | defence "luck" regresses |
| FT luck | opp FT% − league | pure noise, regresses fully |
| Close-game luck | win% − Pythagorean win% from net rating | ML/ATS of lucky teams |
| Player shooting heater | 3P% / FG% over EB posterior | props Over on a heater → flagged |

Use: (1) on the card, a "Regression" column with the signed gap and its reliability; (2) as a gate
only where the study shows the **close does not already price** the regression (tested on D1/D4
2007–2026). Expected (from CFB/NHL): mostly priced → flag-only.

### 5b. Injury & schedule filter
| Signal | Source |
|---|---|
| Out / doubtful / questionable / probable / starting, with timestamps | D11 hourly + D12 |
| Minutes-restriction tags, return-from-injury games | D12 + box history |
| Back-to-back (2nd night, home and away separately), 3-in-4, 4-in-6 | schedule |
| Road-trip length (game # of trip), first home game after a long trip | schedule |
| Travel miles since last game, time zones crossed (E→W vs W→E) | D13 |
| Altitude (DEN, UTA) | D13 |
| Rest differential | schedule |
| Star rest risk (2nd night of B2B, national TV, load management history) | D6 + D11 |

**Back-to-backs move availability, not minutes [MEASURED].** Stars (≥30 mpg, ≥40 games), regular
season, ESPN boxes. On the second night of a B2B they missed **18.3% vs 13.0%** of games, 2023–26
(+5.3 pts, 95% +4.0 to +6.8, clustered by game; 2022-23, before the participation policy: +7.4).
When they played, minutes were unchanged: **33.2 vs 33.3** (−0.1, 95% −0.4 to +0.2). So the B2B
term goes into `P(plays)` (§5b), and the minutes model gets no B2B cut. A flat 10–15% cut would be
3–5 minutes the data does not show. Rest and fatigue curves fit on 2023-24 onward; 2022-23 is the
pre-policy check.

Use: injuries enter the **model** through the lineup rebuild (§4.3); schedule terms are each
measured against the **closing line** on 2007–2026 and enter the model at the fitted value or 0.

**Questionable is a probability, not a ban.** A questionable player is priced as a mixture:
`p = P(plays)·p_with + (1−P(plays))·p_without`, with both lineups rebuilt. `P(plays)` is fitted
from the official reports and the box scores, by status, report hour vs tip, path through the
day and reason. Base rates, from the last report at least an hour before tip, on every other
2023-26 date [MEASURED]:
| Status | n | Played |
|---|---|---|
| Probable | 648 | 91.7% |
| Questionable | 1,328 | 74.0% |
| Doubtful | 65 | 13.8% |
| Out | 16,739 | 0.3% |
| Questionable at the first report → Available later | 1,085 | 87.1% |
| Questionable at the first report → still Questionable | 1,319 | 74.1% |

The market's own view comes from the line: the mixture price is compared with the no-vig line,
and the gap is the bet. When the status changes (ESPN feed or a new official report), the news
alarm (§5d) re-prices that game at the new `P(plays)`. It is released once the line settles and
a pricing run newer than that has seen it, about two minutes, not at the next scheduled card.
There is no warm-up feed; the status change is our proxy for "he went through warm-ups".
Statuses are already the official five-level scale. Reasons are mapped by a fixed table to
categories: injury by body region, illness, rest, G League, personal / not with team,
suspension, reconditioning, concussion protocol, trade. That covers 173 distinct official
prefixes across 2023–26. RotoWire and ESPN notes go through the same table, and an unmapped
string is logged and counted rather than guessed.
The categories enter `P(plays)` as separate indicators, not as a 1–10 "severity" number: a
rest tag and a fracture tag are not points on one scale. An unmapped string raises a
high-priority alert (alarm log + a banner on the next card). That player's `P(plays)` falls back
to the status-and-hour model (never 50/50), and his game can't be a buy until the string is
mapped. When the official PDF and the ESPN feed disagree, the PDF governs. The conflict is
logged and re-checked at the next report.
The study has to answer first whether our re-price beats the market's own move after a status
change, on the hourly snapshots (D1/D2, Phase 1b). If it doesn't, Questionable games stay
priced but un-bet until that bar is met.
"Already priced in" tag when the line moved by ≥ our rebuild delta before the pass.

### 5c. Line-movement filter (ML, ATS, totals; 1H and props too)
| Signal | Computed as |
|---|---|
| Open → now move | no-vig prob / points since open (D1/D3) |
| Move direction vs our side | with us / against us |
| Steam | ≥ X points across ≥ N books inside 10 minutes [FIT] |
| Reverse line movement | price moves against the publicly favoured side (only if a public % source is verified; else omitted) |
| Book dispersion | best price vs consensus fair; stale-book detection |
| Key numbers | ATS crossing 3/4/5/6/7 (NBA keys are weak — measured, not assumed) |

Use: the **drift gate** (MLB pattern) — refuse when the market has moved against our side by more
than a fitted amount since open (the market knows something); allow "with us" moves. Fitted on
D1/D2 2023–26: does a move against us predict negative CLV/ROI on our side? Ships at the fitted
threshold or off.

### 5d. News alarm (latency)
A scheduled card prices against whatever it last saw; a star scratched 20 minutes before tip
moves the line at once, and a stale card reads the move as value on the short-handed side. So:

| Watched every 5 min | Fires when | Then |
|---|---|---|
| Featured board (D3, 3 credits) | consensus no-vig ML moves ≥ 3 pts, spread ≥ 1, total ≥ 1.5 [DEFAULT → FIT] | re-capture that game's 1H + props; mark it **pending** |
| ESPN injury feed (free, timestamped per change) | any status change for a player on a team playing today | same |
| Official report (D11, new hourly file) | any status change vs the previous file | same |

**Settle gate.** After news, a line rarely moves once: it steps for several minutes. So an
alerted game is fast-polled (game markets for that event, every 60 s, up to 4 polls per tick)
and stays **pending** until two consecutive polls each move less than 0.5 points on spread and
total and 1 point of no-vig probability on ML [DEFAULT → FIT] (`alarm.settle`, recorded as a
`settled` alert). A game still moving at the end of the tick stays pending into the next one.
Moves are measured from the board seen at the game's last alert (or the day's first board),
not just the previous tick, so a slow creep under the threshold still adds up.

A **pending** game cannot be recommended until its line has settled **and** a pricing run
strictly newer than the settle has re-priced it (`alarm.pending`); the re-price runs for that
game only, not the slate. Every
alert is archived with its timestamp, so the audit grades what the alarm caught (buys it
blocked vs what they would have done) and what it missed (moves with no alert). Thresholds are
fitted on the 2023–26 archive: the move size that best separates moves that followed a real
availability change from noise. Phase 0 ships the alarm (`nba-engine watch`) and its archive;
the re-price hook lands with the model.

---

## 6. Selection and tiers

Buy = calibrated edge ≥ fitted floor **and** price in the market's fitted band **and** both sides
of the same book paired **and** quote age ≤ fitted minutes **and** passes 5b gate **and** passes
5c drift gate **and** not a duplicate exposure on the same game state (one bet per correlated
cluster, e.g. Over + star Points Over). Tiers Strong/Moderate/Pass, ordered only if the ledger
shows the order holds (MLB #150). Probation markets print as "Lean (probation)", never "Buy".

---

## 7. Audit mechanism (03:00 ET, CFB/NHL pattern)

1. Grade yesterday from ESPN finals (game, 1H from period scores, props from box; DNP → void).
2. Stamp closes (last pre-tip snapshot, `pre` status only; anything later is dropped and counted).
3. Write graded rows to `nba/ledger.csv` on engine-state (pushed by the Mac).
4. Audit PDF + MP3 + ledger workbook, emailed once:
   - **Lead**: the slate's buys W-L-P and units, then ledger to date (CFB #391/#392).
   - Per market: model vs close vs open Brier; buys vs model-favoured vs baselines; CLV (prob
     points and EV) by price band; calibration; PPV/NPV vs base rate.
   - **Filter tables**: for 5a/5b/5c and every gate, admitted vs refused record, ROI, CLV — the
     false-negative scoreboard.
   - Integrity block (§2 last row) and probation status with required-n remaining.
5. Monthly probation review: retire / reopen per §11.

---

## 8. Excel recommendations workbook (`PayoffPitch_NBA_<date>.xlsx`)

| Tab | Contents |
|---|---|
| **Buys** | Tier, game, tip ET, market, selection, line, book, price, model %, fair %, edge, EV, filters (Reg / Inj-Sched / Move), CLV-so-far |
| **Games** | Every game × ML/ATS/Total + 1H ML/ATS/Total: model line, market line, model %, fair %, edge, filter flags, gate |
| **Props** | PTS/3PM/REB/AST/PRA per player: projected minutes, mean, line, P(over), fair, edge, availability, regression gap, gate |
| **Regression** | Team and player luck gaps with reliability |
| **Injuries & Schedule** | Status + timestamp, minutes impact, `[priced in]/[reported, not scored]/−x.x`, B2B, trip game #, miles, tz, altitude, rest diff |
| **Line Moves** | Open → now per market, move vs our side, steam flags, best book |
| **Fades** | Refused rows with the gate that refused them |
| **Audit** | Ledger to date per market (same tables as §7) |
| **Legend** | Every column, every filter, evidence label |

Colour tiers and layout follow the MLB/NHL workbooks (neon → fade).

---

## 9. Daily slate PDF (same style as MLB/CFB/NFL/NHL cards)

Per game block: title `AWAY @ HOME — tip ET, arena, TV`; **market line under the title**
(spread, total, ML, 1H); records (overall, home/road, ATS, O/U); net/off/def/pace ranks; key
players with projected minutes; injuries with report timestamp; rest/travel line (B2B, trip game
#, miles, tz, altitude); regression note; line movement since open; model shape (margin, total,
1H); the game's rows table with gates; storylines. Front page: slate summary, buys, top props.
Plus MP3 sportscaster audio (NHL/MLB pattern) and the regression report.

---

## 10. Automation — when the email goes out

The official injury report updates **hourly** [VERIFIED], most tips are 7:00–7:30 PM ET with West
games at 9:00–10:30 PM ET. The best send time is **measured, not guessed**: Phase 1 replays the
2023–26 card at T-6h, T-3h, T-90, T-60, T-30 (5-min price snapshots + the hourly report that
was public then) and picks the pass with the best CLV per bet after availability risk. Defaults
until that study lands:

| Time (ET) | Job | Email |
|---|---|---|
| every 5 min 11:00–23:55 | **news alarm** (`watch`): board + ESPN feed + official report; re-capture moved games | — (alerts archived) |
| 11:15 | morning board + props capture, injury report 11 AM | **Preview** PDF (no buys) |
| **17:15** | after the 5 PM report; main card for 7:00–8:30 tips | **Card**: PDF + Excel + MP3 |
| 20:15 | after the 8 PM report; card for 9:00+ tips | Late card (only if late games) |
| T-5 and **T-1** per game | close capture, 1-min `close` job; the T-1 quote is the graded close. Kept only if ESPN still shows the game pre-tip **and** the quote is stamped before the scheduled tip (clock lock; ESPN's status can lag the jump ball). T-5 is the fallback (~110 credits per game per capture). Recommendations for a game stop at its last scheduled pass, well before tip | — |

A prop the book pulls before the T-1 capture keeps its row. It is graded on the outcome (ROI,
PPV/NPV and calibration include it) with `close = pulled`. Its CLV goes in a separate **pre-pull
CLV** column, measured against the last quote we captured (last scheduled pass or alarm
re-capture). It never shares a column with true CLV, because the two snapshots are taken at
different times. The audit also compares the pull rate of our buys with the board's.
*Do pulls carry information? [MEASURED, 100-game T-1 sample]* Of 56,054 archived-close prop quotes,
860 (48 games) were pulled by T-1. Their Overs hit 49.0% against a 49.4% no-vig price
(−0.4 pts, the same as kept quotes at −0.8). So in this sample the books weren't pulling
lines that turned out wrong. The 1,056 whose **line moved** were different: Overs at the old line hit 46.5% vs
49.9% (−3.4 pts), so a line move carries news. Sample is small, so it gets re-measured on live
T-1 capture.
Bets are straight wagers only. No same-game parlays; same-game props are one exposure (§6).
Every book is captured and archived, and the consensus is built from all of them. **Buys are
priced only at DraftKings and BetMGM** (`EXEC_BOOKS = ("draftkings", "betmgm")`), at the
better of the two. The ledger records which book priced each row, and CLV is graded at that book's
close. Archive coverage, 2023–26 closes [MEASURED]: DraftKings has the game in 100% of games,
PTS props in 99% and 1H ML in 98%. BetMGM has the game in 99%, PTS props in 96% and 1H ML in only 72%.
Hold: DraftKings ML 4.3% / PTS 6.4%, BetMGM ML 4.6% / PTS 7.1%. A market neither book posts is
priced and graded but can't be a buy. Every fitted gate (price band, edge floor, Over bias) is
checked at these two books before it ships.
*One book vs two [MEASURED, archived close, EV vs all-book consensus fair].* Taking the better
of the two is real value: on PTS props the better side averaged −4.97% EV, against −6.03% at
DraftKings and −6.56% at BetMGM; on 1H ML −2.77% vs −4.03 / −4.05. A row posted at only one
book doesn't get that gain (DK-only 1H ML −4.31%). The 1H ML is DK-only in 2,176 of 7,784
sides. So the **edge floor is fitted separately for one-book and two-book rows** on the
walk-forward. It is not raised by a typed amount. Because "better of two" also selects the noisier quote, the
two-book floor is fitted on the price actually taken, not on either book alone. The audit
reports buys by book and by one-/two-book so a drift toward DK-only rows is visible.
**Over bias at the two books [MEASURED, archived close, proportional de-vig].** DraftKings
−1.56 pts (95% −1.83 to −1.29; PTS −1.3, 3PM −1.7, REB −2.0, AST −1.3, PRA −1.3) across 224K
quotes. BetMGM −1.27 (−1.51 to −1.01; PTS −1.0, 3PM −1.5, REB −2.0, AST −0.6, PRA −1.3) across 247K.
Overs returned −9.1% at both, Unders −3.2% (DraftKings) and −4.3% (BetMGM). The bias is fitted
per market × book for these two, shrunk toward the all-book value.

**T-1 vs the archived close [MEASURED, 100 random 2023–26 games, 10,970 credits].** The T-1
request returned a snapshot 4.4 min before tip, against 9.4 for the archive (98 of 100 newer).
In those five minutes the consensus no-vig ML moved 0.2 pts on average, and 2% of games moved ≥1 pt.
The spread moved in 16% of games (1% by ≥1 pt) and the total in 24% (2% by ≥1 pt). Among 57,041
prop quotes, 3.4% changed line or were pulled, and the rest moved 0.2 pts no-vig. The Over gap
was identical at both times. Live capture uses T-1, since it costs the same as T-5. The 2023–26
archive stays at its 9.4-min close for grading and is not re-pulled.
| 03:00 | grade, audit, push `nba/` to engine-state | **Audit** PDF + MP3 + ledger |

Mac: `scripts/nba/macos/` launchd plists + `.command` shortcuts (run_predictions, run_audit,
nba_capture, open_ledger, install_schedule), `~/.nba_engine`, `NBAE_` env prefix,
`/etc/engine.env`. Credit use at defaults ≈ 25–30K/month for the three full passes, plus ~470 a day
for the 5-min board and ~8 per alarm re-capture [estimate].

---

## 11. Probation bar (per market and per filter)

1. **CLV**: mean close-based EV > 0, lower 95% bound > 0; required n from the archive's own σ.
2. **Calibration**: model Brier on its buys ≤ market Brier on the same rows.
3. **ROI**: not significantly negative; refused rows not doing better (false-negative table).
Small samples are labelled exploratory/underpowered; date- and game-clustered bootstrap.

---

## 12. Algorithm / worksheet flow

```
┌──────────────────────────────── DATA (§1) ────────────────────────────────┐
│ D1/D2/D3 Odds API (open→close, game/1H/props)   D4 SBR 07-23   D5 Covers    │
│ D6 Kaggle box   D7 PBP/stints   D8 nba_api (Mac)   D10 ESPN   D11/D12 injury │
│ D13 arenas/travel                                                           │
└───────────────┬─────────────────────────────────────────────┬───────────────┘
                │ as-of slicing (no look-ahead)               │ timestamped archive
                ▼                                             ▼
┌──────── STABILIZED METRICS (§3) ────────┐      ┌──── MARKET LAYER ─────────────┐
│ split-half r → k per metric             │      │ pair both sides, power de-vig │
│                                         │      │ prop Over bias (fitted, §4.7) │
│ EB posterior rate = (k·prior+n·obs)/(k+n)│      │ consensus fair, open/now/close│
└───────────────┬─────────────────────────┘      └──────────────┬────────────────┘
                ▼                                               │
┌──────── PLAYER & TEAM (§4.1-4.3) ───────┐                     │
│ minutes model ← availability (D11/D12)  │
│ questionable = P(plays)·with + (1−P)·w/o│                     │
│ on/off backup profiles (4.2a)           │                     │
│ lineup rebuild = Σ min × impact + resid │                     │
│ schedule terms (fitted or 0)            │                     │
└───────────────┬─────────────────────────┘                     │
                ▼                                               │
┌──────── DISTRIBUTION (§4.4-4.5) ────────┐                     │
│ poss × eff → (home, away) pts, OT, 1H   │                     │
│ props: split team totals: next man up → │                     │
│   ceilings → measured TOV/eff rest (5a) │                     │
│ blowout dimmer on simulated minutes (4a)│                     │
│ isotonic calibration                    │                     │
└───────────────┬─────────────────────────┘                     │
                ▼                                               ▼
┌──────────────── PRICE: p_final = w·p_model + (1−w)·p_fair  ; edge, EV ──────┐
│ markets: ML · ATS · Total · 1H ML · 1H ATS · 1H Total · PTS · 3PM · REB ·   │
│          AST · PRA                                                          │
└───────────────┬─────────────────────────────────────────────────────────────┘
                ▼
┌──────── FILTERS (§5) — flag every row, refuse via veto_gate ────────────────┐
│ 5a Regression │ 5b Injury & schedule │ 5c Line movement │ 5d News alarm      │
└───────────────┬─────────────────────────────────────────────────────────────┘
                ▼
┌──────── GATES (§6): edge floor · price band · pairing · quote age ·         │
│          exposure dedupe · probation  →  Strong / Moderate / Lean / Pass    │
└───────┬──────────────────────────────┬──────────────────────────────────────┘
        ▼                              ▼
┌── OUTPUTS (§8-9) ───────┐   ┌── LEDGER (write-once per pass) ─────────────┐
│ Excel workbook          │   │ every row incl. refused, flags, gates,      │
│ Slate PDF + MP3         │   │ price taken, prior version, pass id         │
│ Email 11:15/17:15/20:15 │   └──────────────┬──────────────────────────────┘
└─────────────────────────┘                  ▼ T-1 close  ·  03:00 grade
                               ┌── AUDIT (§7) ───────────────────────────────┐
                               │ model vs close vs open · selection vs       │
                               │ baselines · CLV · ROI · calibration ·       │
                               │ PPV/NPV · filter false-negative tables ·    │
                               │ integrity · probation → email PDF/MP3/xlsx  │
                               └──────────────┬──────────────────────────────┘
                                              ▼
                               nightly: team ratings + player posteriors
                               monthly: refit k / w / floors / thresholds
                               (walk-forward only) ─── back to top
```

---

## 13. Build order (Devin sessions)

| Phase | Work | Sessions |
|---|---|---|
| **0 — now** | `nba_engine/` scaffold, config, IDs, Odds API live capture (game, 1H, 5 props), official injury PDF + ESPN injury feed, ESPN schedule/box, news alarm (`watch`), Mac daemons + engine-state `nba/` | 1 |
| **1a — now** | Historical pull D1/D2 2023–26 (game, 1H, 5 props; open + close + hourly path) into a local archive; SBR/Covers loaders. T-1 re-pull **declined**: the 100-game sample (10,970 credits) moved too little to pay ≈400K for (§10) | in parallel with 0 |
| **0b** | Live close at T-1 (1-min `close` job, ESPN pre-tip check, T-5 fallback) before opening night | with 1a |
| 1b | Stint pull (D7) → on/off profiles, next-man-up vacancy tables; `P(plays)` model and status-change re-price study; Over-bias fit and blowout minute curves; reliability study → `k` table (§3); rating step-size study; schedule/travel study vs close; regression-filter study; line-movement study; de-vig study; send-time study (§10) | 2 |
| 2 | Priors, minutes, lineup rebuild, distribution fit to market, props model, calibration, market blend; walk-forward backtest 2023–26 vs real closes | 2 |
| 3 | Gates + filters, ledger, audit, Excel, PDF/MP3, email, launchd schedule | 1–2 |
| 4 | Paper season; monthly probation reviews | ongoing |

Opening night is ~2 weeks out: Phase 0 + 1a first, so the archive and the backtest exist before
the first live card.

## 14. Still needed from you

1. Kaggle API token as a secret, or a one-time download of the Wyatt Walsh SQLite on the Mac.
2. OK for a Mac daemon running nba_api (stats.nba.com blocks the VM).
3. Confirm the Mac's `/etc/engine.env` key also shows the 5M balance (0-credit check block).
4. ~~Which sportsbooks you bet at~~: DraftKings and BetMGM (§10).

**Config vs fitted values.** `nba_engine/config.py` holds paths, defaults and gates. Every
fitted value (Over bias, `P(plays)` model, `k`, `w`, floors, alarm thresholds, ceilings) lives
in a versioned `nba/params/*.json` on engine-state with its fit date, n, CI and holdout
score. A refit writes a new version, and the ledger stamps the version that priced each row.
Measured base rates (74% Questionable, −1.5 PTS Over gap) are priors for those fits, not
constants. The Over bias is re-measured with power de-vig at T-1 before it prices. The
Kaggle SQLite is a read-only input at `NBAE_KAGGLE_DB`. Engine state stays CSV / gzipped JSON.

---

## Appendix — evidence carried over from MLB / CFB / NFL / NHL

## 1. What the four engines taught us (scoreboard from engine-state, pulled today)

| Engine | Evidence | Number | What it means for NBA |
|---|---|---|---|
| MLB | All buys, 7/19–10/04 | **3,618 buys, −334.8u, −9.3% ROI** [MEASURED] | Big model, many markets, still loses. Breadth is not edge. |
| MLB | Model vs close, all priced rows | Brier **.2198 vs .2149** (n=140,630) [MEASURED] | The market is better calibrated than the model across the board. |
| MLB | Buy CLV | +0.17 pp mean, 57% beat close [MEASURED] | Beating the close on *count* but by less than the hold ≠ profit. CLV must be measured in price, not just beat-rate. |
| MLB | Buy model prob vs fair (batter hits) | .561 vs .480 [MEASURED] | An 8-pt gap on a liquid prop is overconfidence, not edge. Edge caps and market blend are mandatory. |
| MLB | Buys by price | +200..+400 −18%, +400+ −17% [MEASURED] | Long plus-money leaks. Price bands per market, fitted. |
| MLB | Totals sheet 9/11–9/29 | SUM 112-97-9 vs always-Over 123-86 [MEASURED] | Every lean list needs a naive baseline on the same games. |
| CFB | All buys 8/29–10/03 | **192 buys, +20.3u, +10.6%**; ATS 95 +18.9u, totals 63 −5.0u [MEASURED] | Positive but small n; gates select ~11% of rows. Gates earn, ratings don't. |
| CFB | Model vs close | Brier .2151 vs .2063 (n=1,324) [MEASURED] | Same as MLB. |
| CFB | ML pricing bug (#382) | Fixed-SD normal + one-sided shrink → sides summed to .90 [MEASURED] | Fit the distribution's dispersion to the market's own spread↔ML↔total relation; never shrink one side. |
| CFB | In-play closes (#377) | 66 games' closes were in-play [MEASURED] | Close capture strictly pre-tip, per game, timestamped. |
| NFL | Ridge rating vs close | MAE 10.282 vs 9.905; disagreement explains none of the line's error (t=+0.25, n=3,450) [MEASURED] | Opponent-adjusted team ratings are a prior and a game script, not a bet. |
| NFL | De-vig study | proportional de-vig creates favourite–longshot slope; **power** within 1pp in 4/5 buckets [MEASURED] | Use power de-vig by default; re-measure on props where hold is 6–8%. |
| NBA | Prop Over bias, 2023–26 close | Over hit 48.0% vs 49.5% no-vig, 1.99M quotes / 3,651 games, every season and market negative [MEASURED] | Price the Over below the de-vigged close by a fitted amount (§4.7). |
| NBA | ML hold with a questionable star, 2023–26 close | −0.01 pts vs no Q star (95% −0.03 to +0.01) [MEASURED] | No ML volatility cap (§4.7). |
| NBA | 1H vs regulation 2H points, 2023–26 | +0.77 per team (95% +0.49 to +1.04); +0.45 in games under 10 [MEASURED] | Fitted 1H distribution, not a half-share (§4.4). |
| NBA | Star minutes, Cup knockouts 2023–25 | +0.8 (95% −0.6 to +2.0), 3 finals [MEASURED] | `cup_stage` flag only (§1). |
| NBA | 2025-27 rule changes | heave rule, push-off emphasis, Cup semis at home, OOB replay [VERIFIED NBA.com / ESPN] | Register §3a. |
| NBA | Pulled vs moved props, T-1 sample | pulled −0.4 pts (n=860), line moved −3.4 pts (n=1,056) [MEASURED] | Pulled rows kept; pre-pull CLV column (§10). |
| NBA | Best of DK/MGM vs one book | PTS −4.97% vs −6.03 / −6.56% EV; 1H ML −2.77% vs −4.03 / −4.05% [MEASURED] | Floor fitted per one-/two-book (§10). |
| NBA | Prop Over bias after a late star scratch | −1.8 pts (95% −3.6 to −0.2) vs −1.5 baseline [MEASURED] | No dial-back after news (§4.7). |
| NBA | NBA Cup knockout IDs, 2023–25 | 21 of 21 matched, all markets [MEASURED] | Teams + tip-time fallback only (§1). |
| NBA | Star minutes on B2B, 2022–26 | Absence +5.3 pts, minutes −0.1 (95% −0.4 to +0.2) [MEASURED] | B2B enters `P(plays)`; no minutes cut (§5b). |
| NBA | Close timing, 2023–26 archive | Historical snapshots sit on a 5-min grid ~30 s past each 5 min; the T-5 request returned a snapshot 9.4 min before tip (median, 3,607 games) [MEASURED] | Request at T-1 (≈4.4 min); live close at T-1 with a pre-tip check (§10). The 100-game T-1 sample barely moved, so there is no re-pull. |
| NBA | Official status → played, 2023–26 | Questionable 74.0% played (n=1,328); Q→Available 87.1% [MEASURED] | Price Questionable as a mixture, do not ban the game (§5b). |
| NFL | Props | 13,650 graded quotes, all `research_only` [MEASURED] | Archive first, price later — exactly the NBA props path. |
| NHL | Phase 0 | odds/period/prop capture to engine-state since 9/29 [MEASURED] | Copy the capture-first build order. |

### Lessons carried over (each with its NBA translation)

| # | Lesson (source) | NBA rule |
|---|---|---|
| L1 | Market beats the model on main lines (all four) | Final probability = market-anchored blend; blend weight and edge cap fitted, not assumed. |
| L2 | Separate accuracy, market comparison, selection, gates, price, CLV, ROI (all) | Audit reports each as its own table; never one "record". |
| L3 | Every refusal is recorded and graded (MLB #117, NHL plan) | `veto_gate`/`pass_gate` on every row; counterfactual grading of refused rows = the false-negative scoreboard. |
| L4 | NPV/false negatives: a gate that refuses winners is a cost (MLB #63, rescue studies) | Each gate reports refused-row ROI and CLV; a gate whose refused rows beat its admitted rows is retired. |
| L5 | Distribution shape decides ML/alt prices (CFB #382, NFL score shape) | Fit margin/total dispersion and their correlation to the market's own relations before any dog/alt can read as value. |
| L6 | No one-sided shrink (CFB #382) | Any shrink is applied to the *distribution*, then both sides are read off it; sides must sum to 1. |
| L7 | Close is pre-start only (CFB #351/#377) | Close = last snapshot before scheduled tip with `game_status=pre`; anything later is dropped. |
| L8 | Key prices by stable event ID, not matchup text (MLB doubleheader #423) | Key = Odds API event id + NBA game id; matchup text is display only. |
| L9 | Pseudo-line Brier better than base rate is necessary, not sufficient (NFL props) | Props gate on CLV against real archived prices, not on fit quality. |
| L10 | Plus-money and longshot props leak (MLB HR, CFB long dogs) | Price bands per market fitted from the archive; ship conservative bands until then. |
| L11 | Correlated legs double-count exposure (MLB, NHL saves/team total) | One exposure per game-state cluster; dedupe same-direction bets on one game. |
| L12 | Shrink by measured reliability, not a ramp (engine_common/shrink.py) | Every rate has its own fitted `k` from split-half r. |
| L13 | Rolling stats mean little; projections do (MLB #129, user rule) | Player and team inputs are multi-season EB posteriors, not L5/L10 windows. |
| L14 | Absences are reported, not scored, until a study says otherwise (CFB QB study, NFL availability, NHL §5.7) | Injury capture with timestamps from day one; pricing of absences only through the lineup rebuild, validated against the close. |
| L15 | Measured nulls ship at 0 with the study in the docstring (CFB wind, NFL September) | B2B, travel, altitude, rest, refs: ship at fitted value or exactly 0. |
| L16 | Pair both sides of the same book; never invent a partner (NHL L16) | One-way quotes are archived but never bet. |
| L17 | Coverage gaps hide as blanks (MLB totals ladder 7.5–10.5 missed playoff 6/6.5) | No ladders. Every line priced from the distribution; audit counts rows with no model read. |
| L18 | Current slate vs ledger in report leads (CFB #391/#392) | Lead sentence always states both; W-L-P when pushes exist. |
| L19 | Price the slate once per pass, write-once predictions (MLB #196, #371) | Each pass appends; graded card = last pre-tip pass per game. |
| L20 | Label normalisation (CFB `teamnames.label_matches`) | NBA team/player ID maps, never string equality on names. |
| L21 | Proxy-market backtests are upper bounds (live_edge) | No claim against a market we did not capture. |
| L22 | Audit from `origin/main` / a detached worktree (memory) | Same for NBA audits. |

---

