# MLB Prediction Engine

Automated daily MLB game & prop prediction engine with EV-based buy tiers and a
nightly self-audit. Pulls the daily slate, builds probabilistic projections from
rolling Statcast/FanGraphs windows, layers weather + regression + travel/rest
filters, prices every market against the book, and exports Strong / Moderate /
Pass recommendations to Excel — with one click from a desktop shortcut.

## What it produces

Per game, for every slate today and tomorrow:

- **Full game**: moneyline, totals, run line
- **First 5 innings (F5)**: moneyline (incl. tie), totals, run line
- **Batter props**: hits, singles, doubles, home runs, runs, RBIs, H+R+RBI, total bases
- **Pitcher props**: strikeouts, outs, hits allowed, walks, earned runs

Each is tagged **Strong buy / Moderate buy / Pass** from expected value vs. the
book (and VSIN handle/bets divergence when provided).

## Model pipeline

1. **Slate ingestion** — MLB Stats API: matchups, venue, probable pitchers,
   confirmed/expected lineups (today + tomorrow).
2. **Rolling features** (Statcast via Baseball Savant / pybaseball):
   - Pitcher form: last 4 weeks
   - Batter home/away: 3 weeks · vs RHP: 3 weeks · vs LHP: 6 weeks
3. **Regression layer** (highest sensitivity / PPV / NPV signals):
   - Batters — HR: max EV, bat speed, barrel rate, hard-hit%; XBH: sweet-spot%,
     xSLG; singles: whiff/zone-contact, xBA, sprint speed; plus **BABIP** and
     **ΔxwOBA** luck regression.
   - Pitchers — **CSW%**, **K-BB%**, 2-strike put-away whiff (K projection),
     **barrel% allowed** (HR/9), **xwOBA/BABIP allowed** (regression). Optional
     FanGraphs **Stuff+ / Location+** when a subscription feed is supplied.
4. **Weather filter** — Open-Meteo temp/humidity/wind projected onto each park's
   home-plate→CF orientation (roof-aware).
5. **Travel/rest filter** — rest days, travel miles, time-zone shift penalties.
6. **First-5 model** — non-stationary Markov base-out chain driven by per-lineup-slot
   L/R-split rates, with a times-through-order (TTO) fatigue adjustment.
7. **Full game + props** — Monte Carlo simulation (starter→bullpen, base running,
   RBI attribution) for run distributions and every player prop.
8. **RBI hard rule** — flags a batter when the preceding 3 slots average OBP > .345
   (3-week window); boosts RBI props.
8a. **Strikeout model** — stuff-based expected-K% (CSW%/SwStr%) prior so thin
    samples regress to a pitcher's stuff, pitcher vs-LHB/RHB platoon K splits,
    catcher-framing + umpire zone K/BB shifts, and a workload/early-hook cap
    (recent BF-per-start + opener detection) that drives realistic K unders.
8b. **Walk model** — command-based expected-BB% prior from Zone%, chase
    (O-Swing%), and first-pitch-strike% (fast-stabilizing discipline signals),
    blended into the BB bucket so thin samples regress to command, not the flat
    league mean.
9. **Market + EV** — no-vig fair prob, EV per $1, then Strong/Moderate/Pass tiers
   ranked on *edge over the devigged price*, not on EV: `EV = decimal_odds x
   edge`, so an EV cutoff is a cheaper bar at longer prices and filled the Strong
   tier with plus-money dogs. Knobs: `MLBE_MIN_EV` (the price must pay at all),
   `MLBE_MIN_EDGE` thin-edge guard, `MLBE_EDGE_STRONG_GAP` (extra edge for
   Strong), `MLBE_MAX_EDGE` (disagreement past which the edge reads as a model
   error), `MLBE_MAX_BUY_ODDS[_<MARKET>]` (price ceiling),
   `MLBE_NO_BUY_<MARKET>` (markets the ledger disqualified), `MLBE_STRONG_ONLY`
   for strict selection. See **Selection guards** below.
9a. **Run-line NPV gates** — veto (not nudge) selections whose failure to cover
    is highly predictable. All off by default; see below.
9b. **Pre-bet CLV** — `MLBE_CLV_GATE` vetoes a buy the market has been walking
    away from since the slate opened.
10. **Excel output** + **nightly audit** (sensitivity / specificity / PPV / NPV
    per tier from final box scores).

## Run-line NPV gates

Gates only ever *remove* a run line, trading bet volume for realized NPV. Each
is independently switchable and tags the vetoed pick with its gate name, so
`mlb-engine audit` can grade the counterfactual on the **Run Line NPV** sheet:
a gate earns its keep only when the picks it removed lost at a materially higher
rate than the ones it kept (`VETO <gate>` vs `KEPT (no veto)`). A gate never
fires on missing data.

| Env flag | Applies to | Fires when | Thresholds |
| --- | --- | --- | --- |
| `MLBE_RL_GATE_ISO_GB` **(on)** | favorite -1.5 | low-power lineup vs a ground-ball starter (no multi-run-homer path) | `MLBE_RL_ISO_MAX` (.170), `MLBE_RL_GB_MIN` (.40) |
| `MLBE_RL_GATE_DOG_SP` | underdog +1.5 | dog starter's last 3 starts show traffic **and** hard contact | `MLBE_RL_DOG_WHIP_MAX` (1.45), `MLBE_RL_DOG_HARD_HIT_MAX` (.45) |
| `MLBE_RL_GATE_DOG_PEN` | underdog +1.5 | dog bullpen cannot strand inherited runners | `MLBE_RL_DOG_PEN_XWOBA_MAX` (.330), `MLBE_RL_DOG_PEN_K_MIN` (.18) |
| `MLBE_RL_GATE_TOTAL` | favorite -1.5 | low-scoring game trending to a 1-run margin (redundant with the simulated margin distribution — validate before using) | `MLBE_RL_TOTAL_MAX` (7.0) |

Turn on one gate at a time and re-run `mlb-engine audit` before adding the next.

Eight graded slates (19-26 Jul 2026) decided the current defaults. `iso_gb`
removed six favorite -1.5s that went 1-5, against 59.6% for the run lines it
kept, so it ships on. The two underdog gates removed 35 +1.5s that won at
66-75%: the simulation already prices weak starters and weak bullpens, so the
gates double-count and delete winners. They stay off.

### Starter window

`MLBE_PITCHER_FORM_DAYS` defaults to **42**, not 28. Across 2,894 starts by 273
pitchers, with every rolling profile rebuilt from the days *before* the start,
the six-week read is the stronger predictor on 66 of 100 metric/target pairs
tested head to head on identical rows, and it wins every held-out target: next
start xwOBA R² 0.087 against 0.075, innings 0.067 against 0.055, K% 0.140
against 0.138. Replaying the 54-slate history moves favoured PPV from .5831 to
.5867, date-clustered 95% CI **[+0.04, +0.64] pp** with 98% of resamples
positive — the effect is concentrated where the study says it should be, on
pitcher strikeouts (+1.54 pp) and F5 run lines (+2.72 pp). Feeding a model both
windows is worse than either alone; they are the same information twice.

Two related findings that did *not* become knobs. A starter's four-week form
*relative to his own six* predicts nothing (next-start xwOBA ρ = +0.023,
p = 0.27), so there is no case for a recency override. And results metrics are
far less repeatable than command metrics. Correlating a six-week window against
the *next single start* gives 0.90 velocity, 0.33 whiff, 0.30 K%, 0.18 xwOBA
allowed, 0.06 HR/BF, 0.05 barrel — but that number is not a reliability, because
one start is itself mostly noise, and it understates every rate. Splitting the
season into adjacent, non-overlapping six-week blocks and correlating one block
against the next (112 pitcher-pairs, ~155 batters faced a block) is the honest
version:

| metric | six-week block repeats at |
| --- | --- |
| Velocity | 0.95 |
| K% | 0.52 |
| Whiff% | 0.52 |
| CSW% | 0.50 |
| K−BB% | 0.46 |
| BB% | 0.40 |
| GB% | 0.39 |
| Hits/BF | 0.33 |
| xwOBA allowed | 0.31 |
| Hard-hit% | 0.24 |
| wOBA allowed | 0.22 |
| BABIP | 0.10 |
| Barrel% | 0.09 |
| HR/BF | 0.02 |

So a starter 40 points of xwOBA better than league over six weeks projects about
12 points better, not 40 — and his barrel and home-run rates project at league.

`MLBE_STARTER_CONTACT_SHRINK` acts on that, **default `0.0` (off)**. At `1.0` it
applies each rate's measured empirical-Bayes weight — `keep = bbe / (bbe + k)`,
with `k` solved so a league-sized six-week sample keeps exactly the reliability
above — to the four contact rates that drive the hit and HR multipliers (xwOBA
and wOBA allowed, BABIP, hard-hit, barrel), and leaves the command and stuff
signals alone. `PitcherRegression.raw_contact` keeps the unshrunk rates.

It is off because the measurement says it does nothing, not because it is
untested: over 54 slates and 182,215 graded picks it moves favoured PPV .5867 →
.5864, date-clustered 95% CI **[−0.26, +0.21] pp**, Brier +0.0001. It is very
far from a no-op — 79% of rows move, mean |Δp| 0.015, and hits+runs+RBI moves on
99% of rows by 0.022 — the accuracy just does not follow. The one
market that clearly dislikes it is hits+runs+RBI (−1.53 pp), which is the most
barrel-driven prop on the board and the place shrinkage bites hardest; make that
survive before turning it on. The honest reading is that the reliability
correction is *right* about the metrics and the simulation was already damping
them enough that re-damping them adds nothing.

### Fastball velocity, and what one start measures

Every reliability number above is a *block* read: six weeks against the next six
weeks. The question the slate article kept raising is narrower — what can be
read off a single outing? Correlating each metric across a pitcher's consecutive
starts (3,256 starts, 253 pitchers, 2026 season through 8/15):

| read off ONE start | repeats next start |
| --- | --- |
| Release height | 0.97 |
| Release extension | 0.95 |
| **Four-seam velocity** | **0.93** |
| Four-seam spin | 0.91 |
| Four-seam IVB | 0.84 |
| Whiff / swing | 0.20 |
| K per PA | 0.20 |
| CSW% | 0.15 |
| xwOBA allowed | 0.10 |
| Exit velo allowed | 0.09 |
| BB per PA | 0.07 |

The split is by *what is being counted*, not by how interesting the metric is:
one start is ~90 radar-measured fastballs and ~22 results. So a velocity read
off one outing is a measurement and a contact read off one outing is not, with
nothing in between.

Which window, then? Shorter is better, monotonically. Held-out RMSE on the next
start's strikeout rate, one velocity read added to the levels the engine already
prices (1,652 starts, chronological halves):

| velocity read | K rate | xwOBA allowed |
| --- | --- | --- |
| none | .10551 | .08200 |
| 7 days | **.10391** | **.08118** |
| 14 days | .10399 | .08121 |
| 21 days (what the article used) | .10408 | .08125 |
| 42 days | .10422 | .08134 |
| 7-day half-life decay | .10403 | .08124 |
| season level + last-start deviation | **.09987** | **.08096** |

Pooling sinkers and cutters in halves the gain (.10477): a sinker-heavy start
otherwise reads as lost velocity. The deviation is asymmetric in the velocity
itself — 30% of a dip survives to the next start against 55% of a spike — but a
one-sided *outcome* fit did not beat the linear term, so the simple version
ships.

Two consequences:

**The article now reads velocity as his last start against the whole window**
rather than three weeks against three weeks. SIERA and CSW% keep their halves;
three weeks is the shortest sample that measures them at all.

**`MLBE_VFA_K_WEIGHT` prices it, default `0.0` (off).** At `1.0` both terms — his
level against a 94.7 league four-seamer at 3.7%/mph, and his last start against
his own window at 7.8%/mph, each clipped — multiply the *blended* strikeout rate.
Scored per PA (2,082 starts / 48,120 PA, binomial deviance, six-week
K%/CSW%/xwOBAcon controlled, 60/40 chronological holdout): 1.05839 for the priced
levels alone, 1.05732 adding the level, **1.05661** adding both.

It stays off because the two ways of scoring it disagree, which is worth stating
precisely.

As a *rate forecast* it passes the bar that retired the stuff multiplier — weekly
walk-forward wRMSE on the next start's K rate, 3,086 starts, no multiplier
0.09749, stuff 0.10400, **velocity 0.09634** — and a dose search keeps ~0.8 of it
where it kept none of stuff, because CSW% and SwStr% are already inside xK% and
`release_speed` is in no other term. The blended rate is flat across last-start
velocity, which is the finding:

| last start vs his window | starts | blended | with velocity | realised |
| --- | --- | --- | --- | --- |
| below −1.0 mph | 75 | .2348 | .2168 | .2177 |
| −1.0 to −0.4 | 388 | .2293 | .2194 | .2146 |
| −0.4 to +0.4 | 1,339 | .2276 | .2280 | .2227 |
| +0.4 to +1.0 | 447 | .2272 | .2381 | .2370 |
| above +1.0 mph | 79 | .2319 | .2502 | .2688 |

As a *price*, replaying nine graded slates at both weights (54,269 graded picks,
identical inputs), it does not: strikeout Brier .20197 → .20166 but log loss
.60567 → **.62127**, and 16 of the 18 other markets get worse. Scaling a
starter's K rate rescales every other outcome he allows, so a mph of fastball
moves his walks, his hits and the game total too — the same coupling that made
the stuff multiplier a worse price than no multiplier at all. A better rate
forecast is not yet a better price, so the columns quote it and no bet pays for
it.

It buys nothing on contact and is not applied there: on hits per *non-strikeout*
PA the level is t = −2.15 for .0002 of deviance and the last-start deviation
makes the holdout worse — the same verdict inverse-BABIP and ΔxwOBA drew when
they were measured on the starter's contact term. IVB is the mirror image: it
hurts a strikeout forecast (z −5.3) and carries home runs (z +13.1), which is
the one place the engine already uses it.

```bash
python -m scripts.velocity_read_study reliability   # what one start measures
python -m scripts.velocity_read_study window        # 1 to 8 weeks, and decays
python -m scripts.velocity_read_study k             # the deviance bar, strikeouts
python -m scripts.velocity_read_study hits          # the same on contact
python -m scripts.vfa_k_price_study                 # the bar that retired stuff
python -m scripts.vfa_k_backtest                    # graded slates, both weights
```

### Bullpen windows

A bullpen's last three weeks is about 270 batters faced spread over a dozen
arms, and at that size the *results* move far more than the *ability* does.
Split-half reliability across the 30 pens (6/16–7/27, non-overlapping halves):

| metric | repeats at | keep |
| --- | --- | --- |
| Velocity | 0.67 | 67% |
| K% | 0.66 | 66% |
| Whiff% | 0.58 | 58% |
| wOBA allowed | 0.47 | 47% |
| xwOBA allowed | 0.37 | 37% |
| BB% | 0.19 | 19% |
| Hard-hit% | 0.13 | 13% |
| HR per batter faced | 0.06 | 6% |

Repeating is not the same as forecasting, so the read was scored the way it is
used: 330 team-windows (April–July, a 21-day read against the **next** 21 days of
relief wOBA allowed), regressing what happened next on what the window said.

| read the simulator gets | sd across pens | slope on the next window | RMSE |
| --- | --- | --- | --- |
| raw | .0371 | 0.15 | .0504 |
| flat 60-PA prior (was the default) | .0300 | 0.19 | .0462 |
| fitted per-outcome priors | .0106 | 0.62 | .0398 |
| assume every pen is league average | — | — | .0397 |

A slope of 0.19 means the pen line was being used at roughly five times its
worth, and the top two rows forecast the next three weeks *worse than knowing
nothing about the pen at all*. `MLBE_PEN_SHRINK` is therefore **on by default**;
the residual 0.62 says the fitted priors are still mildly optimistic, but
strengthening them further buys under .0002 of RMSE and one more fitted constant.

**wOBA or xwOBA?** Over the same windows, xwOBA is the better forecast of the
pen's next three weeks (r = +0.185 against wOBA's +0.141, better in 9 of 11 start
dates), and in a joint regression it takes all the weight (+0.0070 per sd against
+0.0004). It is not a large edge, and both are weak. The simulator needs an
outcome vector rather than a single number, so the pen's contact quality enters
as the xwOBA *level* term in `features/regression.py` (fitted on 16,547 relief
rows, t = +4.5) while the rate vector carries the rest.

| Knob | Default | What it does |
| --- | --- | --- |
| `MLBE_PEN_SHRINK` | `1` (on) | Shrinks the pen's outcome rates with the fitted per-outcome priors (1B k=1834, HR 708, BB 344, K 241, OUT 410; 2B/3B pinned to the league pen, where their entire observed spread is binomial noise). `0` restores the flat 60-PA prior. |
| `MLBE_BULLPEN_SKILL_DAYS` | `0` (off) | Reads the pen's stuff/command signals over a longer window than its rates. Set to `42`: out of sample against the following three weeks, relief K% scores 0.73 on 42 days against 0.66 on 21, and in a joint regression the 42-day read takes weight +0.68 against +0.14 for the last three weeks. Results-based rates stay on `MLBE_BULLPEN_DAYS` (21), where they belong — xwOBA scores 0.37 on 21 days against 0.32 on 42. |
| `MLBE_BULLPEN_XWOBA_SHRINK` | `0.37` | Share of a pen's distance from the league mean (.306) to keep, set to the measured reliability. The underdog run-line gate deliberately keeps reading `BullpenProfile.xwoba_raw`: its .330 cut was calibrated on unshrunk means, and at 0.37 even the league's worst pen (.353 raw on 8/16) lands at .323 — reading the shrunk value would retire that gate silently rather than on evidence. |

### Two bullpen numbers that do not survive being scored

Both were printed in the daily preview as verdicts, and both are gone from it.

**Arm-to-arm "volatility"** — the standard deviation of wOBA allowed across a
pen's individual relievers. Over two adjacent three-week windows it repeats at
**r = −0.10** across 27 pens (−0.07 on the following window pair); the observed
spread (.021–.027) sits *below* the binomial noise floor for arms with ~33
batters faced (.031–.035), leaving no measurable talent spread at all in it; and
the same reliever's wOBA allowed repeats
window to window at **r = −0.04** (n=61). "Volatile" and "uniform" were two names
for one coin flip. The number is still captured on `BullpenLine.arm_spread` for a
later study; it is no longer read out.

**The "gassed arms" workload proxy** — scored over 970 team-games against the
relief wOBA that pen actually allowed that night: **r = −0.035**, and pens at the
depleted threshold allowed **+0.005 ± 0.018** more than the rest. It is a true
description of who threw yesterday and not a forecast, so the preview now reports
it as usage with no colour and leaves it out of the narration.

The same proxy was also spending money. `features/ml_gate.py` demoted a
full-game moneyline buy whose own pen was depleted and at least 15 fatigue points
worse off than the opponent's — the reasoning being that the full game is decided
in the innings those arms cover. Rebuilt per team-game over **3,956 team-games**
and scored against the game it was about to be spent in, the sides it would have
demoted won **.529** and **.507** across the two windows against those same teams'
own rates of .493 and .506 — no worse than usual, and arguably better — with
r(fatigue, win) of −0.005 and +0.000. The fatigue branch is therefore **off by
default** (`MLBE_ML_PEN_FATIGUE=1` restores it). The Rotowire *availability*
branch, which reads arms actually declared unavailable rather than inferring
tiredness from pitch counts, is a different signal and stays on — ungraded, for
want of any history of that feed to grade it against.

All four measurements are reproducible: `python -m scripts.pen_read_study
{forward,spread,fatigue,mlgate} --cache <statcast pickle>`.

One caveat worth carrying: team-level *velocity* is the most reliable bullpen
number and also the most misleading one. Detroit's pen appeared to lose 2.2 mph
between those two windows, the largest drop in baseball; restricting to the eight
arms who pitched in both, it was +0.01. The whole move was roster churn, and
league-wide 18% of a window's relief pitches come from arms who were not there
three weeks earlier.

Calibrate the thresholds against this engine's own scales, not FanGraphs/Savant
leaderboards: hard-hit% and GB% are computed off the tracked-batted-ball slice
(same convention as `build_pitcher_regression`) and read a few points lower than
the public versions, and WHIP is the PA-derived proxy. On a full 15-game slate
the published defaults (.45 hard-hit, .50 GB) fired on 0 of 60 run lines, while
.25 / .40 fired on 12 — start loose enough to get a sample, then tighten on what
the `VETO` rows show.

### Bridge innings

A pen's 8th+ profile is its setup man and closer, its two best arms. The
simulator used to hand every post-hook inning of a close game to that profile,
so a 6th-inning hand-off was priced as if the closer were already in — the
overrating is largest for the pens with the widest closer-to-middle-relief gap,
which is one way a favourite's moneyline edge gets manufactured. Relief before
the 8th is now read as its own profile (`BullpenProfile.bridge`, 20 PA minimum,
otherwise the aggregate) and covers the innings up to the 8th; the leverage arms
take over from there.

| Knob | Default | What it does |
| --- | --- | --- |
| `MLBE_PEN_BRIDGE` | `1` (on) | Prices the innings between the starter's hook and the 8th off the arms that cover them. `0` restores the old behaviour, for repricing a slate both ways. |
| `MLBE_PEN_ARSENAL` | `1` (on) | Extends the starter's arsenal matching (mix usage x per-class SwStr% against the hitter's per-class whiff/xwOBA) to the pen, read separately for its bridge and leverage subsets. |

## Install

```bash
git clone https://github.com/<you>/mlb-prediction-engine.git
cd mlb-prediction-engine
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

One-time setup helpers are in `scripts/{windows,macos,linux}/setup.*`.

## Usage

```bash
# today's slate -> Excel + a blank VSIN template to fill in
mlb-engine run

# a specific date, with market prices, custom sim count
mlb-engine run --date 2024-07-19 --vsin-csv ~/.mlb_engine/vsin_today.csv --sims 20000

# snapshot the closing market just before first pitch (scores CLV in the audit)
mlb-engine close --game-only

# capture an outside model's take on the same props (free, expires in a day)
mlb-engine opta

# grade yesterday, update the scorecard, and append to the running ledger
mlb-engine audit

# carry the ledger/closes/pregame picks across machines by hand (run/close/audit
# already do it themselves)
mlb-engine state pull --date 2024-07-19 && mlb-engine state push

# refit the calibration map from the ledger (per-market, validated out of sample)
mlb-engine calibrate --holdout 2

# run the slate, build the reader-facing card, and email it in one shot
mlb-engine run --email --to you@gmail.com

# (re)build the card from a prior run's predictions without re-simulating
mlb-engine card --date 2024-07-19 --email

# grade yesterday AND email the formatted audit report (md/html/pdf) in one shot
mlb-engine audit --report --email --to you@gmail.com

# render the daily or weekly audit report from the existing ledger (no grading)
mlb-engine report --period daily  --email
mlb-engine report --period weekly --email
```

### Closing line value

Win rate answers "was the pick right"; **closing line value** answers "was the
price good", and it answers it in dozens of bets instead of thousands. Run
`mlb-engine close` as late as possible before the first game, then `mlb-engine
audit` fills four columns per graded bet — the closing price, its no-vig
probability, the probability points the market moved toward our side (`clv`),
and the EV of the price we took under the closing probability (`clv_ev`) — and
summarizes them per market on the workbook's **Closing Line Value** sheet.
`--game-only` costs 3 credits for the whole slate; without it, props are priced
per event. Bets with no captured close stay blank rather than reading as zero.

Why it is now the primary scoreboard: nine retro-priced slates put the card at
-5.4% ROI with a 95% interval spanning [-16%, +6%] — hundreds of graded bets and
still no verdict. Over the same bets the market out-forecast the model in every
single market (Brier .2347 vs .2408), so the ROI ambiguity was hiding a clear
result. Every metrics sheet now also reports **Needs %**, the win rate the
prices actually charged for, next to the win rate achieved: 59.6% of favoured
bets won into a 60.5% break-even, which is why 58% PPV never became profit.

### An outside model to be judged against (`mlb-engine opta`)

CLV says whether the *price* was good. It says nothing about whether the
*probability* was good, because the market only quotes the props it chooses to.
VSIN publishes Opta AI's MLB projections beside DraftKings' prices and, once a
slate finishes, the graded outcome of every call — a second model, forecasting
the same props, scored the same way.

```bash
mlb-engine opta            # today's projections and prices
mlb-engine opta --day -1   # yesterday's, now carrying results
```

Free and uncredited, but only three days wide: the page's `day` offset clamps
at yesterday, so a slate not captured within a day of being played is gone for
good. Because of that it is not a command to remember — the morning daemon and
both one-click shortcuts run it themselves, `--day -1` first (last night's
calls, now carrying results) and then today's. The two captures merge, so the
graded outcomes land on the projections already stored for that date.

On the first two slates where both models had a probability for the same prop
(2,523 graded), Opta out-forecast the engine in all five batter markets:

| market | n | engine AUC | Opta AUC |
|---|---|---|---|
| `batter_hr` | 462 | .574 | **.701** |
| `batter_h` | 516 | .600 | **.635** |
| `batter_tb` | 464 | .534 | **.557** |
| `batter_2b` | 462 | .507 | **.553** |
| `batter_hrr` | 449 | .488 | **.530** |

Two slates is not a verdict, which is the reason the capture exists rather than
a reason to act on it yet.

### Carrying state between machines (`mlb-engine state`)

Scheduled runs are separate, disposable machines, so `~/.mlb_engine` starts
empty on each one: the 6:50pm close capture, the 11:30am card and the 2:30am
audit cannot see each other's files. Left alone, that costs both of the audit's
memories — CLV is never scored because the snapshot is on another box, and the
ledger reports one slate as "all dates" every night.

That state lives on an orphan `engine-state` branch of this repo: the pregame
`predictions_<date>.json` (gzipped, most recent 35 slates), the closing
snapshots, `ledger.csv` and `scorecard.csv`. Data only, never code.

**`run`, `close`, `audit` and `report` sync it themselves** — `close` and
`audit` pull before they work and every one of them pushes after — so a
scheduled run needs no extra commands and cannot forget them. Sync is
best-effort: no checkout, no remote, no branch or no push credentials logs a
warning and the run continues on local state alone. Set `MLBE_STATE_SYNC=0` to
turn it off (or `MLBE_STATE_BRANCH` to move it), and drive it by hand with:

```bash
mlb-engine state pull --date "$DATE"   # closes + ledger + that slate's pregame picks
mlb-engine state push
```

Syncs merge rather than replace, in both directions: closing snapshots union
per selection (latest price wins), and the ledger takes the branch's dates plus
this machine's rows for any date it graded, since that box is the one holding
the results. A push that loses a race re-pulls and re-applies rather than
overwriting the other run.

**The hand-dropped exports travel too**, for the same reason and in the same
place: the priced BAT X CSVs (`batx/`), the saved EV Analytics pages
(`evanalytics/`) and the daily projection exports (`projections/`) are
downloaded on a laptop onto one box, and the card is priced on another, so
without this the BAT X and EVA columns are blank whatever you copy locally.
Drop the files, then `mlb-engine state push`, and the next scheduled card reads
them. They are stored gzipped, keeping the newest 14 of each dated feed.

An export is one immutable download, so nothing here is merged: a pull only
fills in files this machine does not have. That asymmetry is deliberate — a
saved page is named after the page rather than the day, so a pull that
overwrote would hand today's card yesterday's board.

The pregame predictions matter as much as the closes. An audit that re-prices
the slate at 2:30am grades a different set of picks, at prices that no longer
exist, and the "bet price" the CLV is measured against becomes a post-game
quote. So pregame files are write-once on the branch — only the run that priced
the slate publishes one — and a pulled copy lands as
`predictions_<date>.pregame.json`, which nothing else writes. `audit` prefers it
over any local re-price and says so, so it grades what the card actually sent.

### Market anchoring

`MLBE_MARKET_ANCHOR` (default `0`, off) blends the devigged market price into the
probability the EV screen bets on: `0` bets the model alone, `1` bets the market,
`0.4` moves the market 40% of the way toward the model. Any market takes its own
weight from `MLBE_MARKET_ANCHOR_<MARKET>`, and a market with a per-market default
ignores the global one. Totals are pinned at `0` that way, because they are the
one market where the model beats the price on Brier (.2446 vs .2480) and the only
profitable buy bucket in the graded ledger, so raising the global toll can never
start taxing them. The model's own
probability is left untouched, so PPV/NPV, the calibration refit and the Brier
comparisons keep measuring the model rather than the blend, and both numbers are
recorded per pick (`Model %` and `Market %` on the card, `fair_prob`/`bet_prob`
in the ledger).

It exists because the market is the better forecaster in every market we bet
(Brier .2347 vs .2408), so the model should have to earn its departures from it.
Be precise about the mechanism, though: the screen is affine in the probability,
so weight `w` scales the measured edge to `(1 − w)·(model − fair)`, which against
a fixed threshold is identical to demanding `edge >= threshold / (1 − w)` — at
`0.6`, a .02 edge requirement becomes .05. It therefore **keeps** the model's
biggest disagreements and drops the small ones; it raises the toll on
disagreeing, it does not make the engine defer.

Nine priced slates: -5.4% at `0`, -4.1% at `0.4`, -3.5% at `0.6` on a third of
the bets, -12.9% at `0.8`. Every one of those intervals still spans zero, so this
shrinks a loss rather than earning a profit. Judge a weight on CLV, not on ROI.

It still ships off even though 27 graded slates put the market ahead of the model
on Brier and log loss in every market the engine bets except totals, and for a
mechanical reason rather than a lack of evidence: every edge floor, price band and
probability floor on the card was fitted against unanchored probabilities, and a
global weight rescales all of them at once (`edge -> edge x (1 - w)`). Raise it
one market at a time and re-grade that market's floors underneath it.

The moneyline is the first market that re-grade was asked of, and the answer was
no. Rescale its edge floor with the weight, and the blend changes no graded row up
to `0.5` — its only remaining bite is the toll against the vig-inclusive
break-even. Leave the floor alone, and the blend *is* the floor: 0.02 at `w`
selects exactly the rows 0.02/(1−w) selects unanchored, and every higher floor
re-grades worse than the shipped one (n=49 at +9.4%, against -5% to -24% at .03
through .06). It stays registered as a candidate rather than settled, so the
verdict refreshes as the ledger grows — see below.

### Grading a screen that does not exist yet

`audit/probation.py` grades three things on the same three tests (volume, a margin
past one standard error, and both halves of the window agreeing): each market on
its own buys, each live screen on the picks it refused, and each **candidate** —
a proposed price band or floor — on the buys it *would* have refused. A candidate
reads `SHIP` only when the rows it deletes lose on all three; anything else means
the floor is a fit to where the window was cut.

The candidates live in `CANDIDATE_SCREENS` and appear in the audit's probation
table beside the live screens. Two are registered, both proposed off the graded
moneyline and both refused by the consistency test: a home-side mirror of
`away_ml_refuse_odds` at -120 (its near-pick'em band ran +11.8% over the older
half and -53.3% over the newer, and the first-five rows it would have deleted were
+30.7%), and the market blend at `0.5`, graded as the EV toll it actually is.
Keeping them here rather than in a chat message is the point: another month of
slates may say something the first 27 did not.

### Selection guards

Twenty-seven graded slates put the model's *ranking* ahead of its *buying*: inside
every probability band the rows the engine bought lost 6-11 points more than the
rows it passed, at longer prices, for -12.6% ROI over 1,894 buys. Three guards
narrow what a measured edge is allowed to buy. None changes a probability, all are
switchable, and a guarded selection is still priced and graded, so the ledger can
tell you what each one cost or saved.

| Env flag | Default | Passes a buy when | Ledger evidence |
| --- | --- | --- | --- |
| `MLBE_MAX_BUY_ODDS[_<MARKET>]` | off globally, `+109` on `game_rl` and `f5_rl` | the best price is longer than the ceiling | plus-money buys 28.5% for -15.5% ROI (n=933) vs 50.7% at minus money; run lines +11.8% at -110 or shorter vs -21.2% at plus money |
| `MLBE_NO_BUY_<MARKET>` | on for `batter_h`, `batter_hrr`, `batter_r`, `batter_tb` | ever, on that market's over | every batter market except doubles lost, -169 units in total |
| `MLBE_DOUBLES_MAX_BUY_ODDS` | `+300` | the doubles over is priced shorter than the ceiling | the model is calibrated on 6,656 graded `batter_2b` rows except in the band it bets: .258 predicted, 15.0% actual (n=346). Bought rows hit 14.3% (n=70), passed rows 14.2% (n=6,586) |
| `MLBE_CLV_GATE` | on, `MLBE_CLV_DRIFT` `0.02` | the side's no-vig price has drifted `>= 0.02` against us since the slate opened | buys that beat the close returned +5.4%, buys that lost it -11.8% |

The ceiling is per market rather than global because a long price means opposite
things on a two-sided market and a one-sided prop — a home run is honestly +500 —
and the markets that are not run lines already have an aimed screen: moneylines
`away_ml_refuse_odds`, home runs their `+400..+700` band, singles a price floor,
RBI a probability floor, doubles a price ceiling.

The doubles ceiling is close to a disqualification — it refuses 69 of the 70 buys
graded so far, because a 20% event is never priced short — and is written as a
price so the rare short number stays buyable. It is a ceiling rather than a band
because no band survived the rows (`+300-350` 0-for-5, `+350-400` -14.4%,
`+400-450` -25.4%, `+450-500` -16.1%, `+500` and longer +18.1% on three winners
in nineteen); the evidence is the calibration table over all 6,656 rows, not the
70 buys, which are under the sample `probation` needs and disagree across the
halves of their window. It runs after every other screen, including the contact
floor, so a row another gate had already refused keeps that gate's name: a screen
the ledger judges on its own refusals cannot be credited with somebody else's.
Disqualification is likewise for the batter markets with
no surviving profitable pocket to screen for; home runs, singles and RBI lost
money too but keep their own fitted screens, which are sharper instruments. Both
apply to overs only: the fade is a different bet with its own screens.

The ceiling outranks the sharp-money upgrade: a confirmed +200 dog is still a
+200 dog. The CLV gate runs last, after every gate and upgrade, and needs no
future information — the pipeline snapshots the first price it sees for each
selection into `audit/board_<date>.json` and never overwrites it, so a slate's
first run defines the open and later runs measure their drift against it. On the
first run of a slate, and for any selection with no captured open, the gate is
neutral.

### Audit report (daily + weekly)

`mlb-engine report` renders the graded ledger into a reader-facing **audit
article** — executive summary, core metrics, a **market scorecard** (every
market rated 🟢 Play / 🟡 Neutral / 🔴 Fade by PPV / NPV / ROI with a *Min p to
Play* conviction floor), the most common errors, recommendations mapped to their
goal (↑ PPV · ↑ NPV · ✕ eliminate false positives · ↓ reduce false negatives),
and a play/fade call. It writes `output/audit_report_<slug>.md`, `.html`, and
`.pdf`. `--period daily` covers a single graded slate; `--period weekly` rolls up
the trailing seven days in the same layout. `mlb-engine audit --report` produces
the daily report for the slate it just graded; add `--email` to send it (PDF +
markdown attached, HTML in the body). The scorecard verdicts are rule-based
(`PPV ≥ 0.55` and positive ROI → Play; `ROI ≤ −15%` or `PPV < 0.45` → Fade;
markets with fewer than five favored picks stay Neutral until the sample grows).

#### Price buckets: judging a bet against the price it was taken at

A win rate only means something next to the break-even its price demands, so the
report carries a **price-bucket** section (also printed by `mlb-engine audit`)
grouping every buy that had a *real* book price into plus-money vs minus-money
and six bands from `-200 and shorter` to `+400 and up`. Each row reports the
realized win rate, the `1/decimal` break-even the prices demanded, and the **gap**
between them:

```
Underdogs (plus money)    n=132  win% 35.6  need 42.9  gap  -7.3 pts  ROI -16.9%
Favorite (-199 to -110)   n= 76  win% 55.3  need 56.7  gap  -1.4 pts  ROI  -2.1%
```

A dog is meant to lose most of its bets, so 35.6% is only a leak because the
price wanted 42.9% — and 55.3% on short favorites is only a loss because the
price wanted 56.7%. Rows graded at the assumed -110 fallback are excluded, which
makes this the only ROI in the report that reflects prices we actually got.
Bands are reported at `n ≥ 15` and flagged as leaks at 3+ points short. Because a
single slate cannot fill six bands, the daily report measures this section over
the whole ledger and labels how many priced slates it covers.

### Daily card (hybrid writeups)

`mlb-engine card` turns a run's recommendations into a reader-facing **betting
card** — one section per game with a short analytical read (which starter owns
the strikeout edge, which side the model favors, whether the moneyline is a
fade, the total lean) followed by the top positive-EV plays with model
probability, market-implied probability, and EV. It writes both
`output/card_<date>.md` and `output/card_<date>.html`. Add `--card` to
`mlb-engine run` to emit it alongside the workbook, or `--email` to send it
(see credentials below). It reads the persisted `audit/predictions_<date>.json`,
so it can be regenerated any time without re-running the sims.

Outputs land under `~/.mlb_engine/` (`output/` workbooks, `audit/` predictions +
`scorecard.csv`). Override with `MLBE_DATA_DIR`.

Each audit also maintains a **running ledger** across every graded slate:
`audit/ledger.csv` (one row per bet: date, market, selection, odds, tier, result,
P/L) and `output/ledger.xlsx` with these sheets:

- **Overall** — cumulative sensitivity / specificity / PPV / NPV / win% /
  **Needs %** (the break-even win rate the prices charged) / ROI / net units.
  The first row is the **whole engine** (`ENGINE (p>=.5)`): PPV/NPV
  keyed on the model's own probability boundary across *every* graded market and
  tier, so it measures the engine's raw directional discrimination (how often the
  side the model favors wins vs. how often the side it fades loses), independent
  of EV/odds/tiering. The remaining rows are the per-tier and Buy (S+M) breakdown.
- **Daily PPV-NPV** — the same whole-engine PPV/NPV computed per slate date.
- **Prop PPV-NPV** — PPV/NPV for every batter/pitcher prop market
  (`batter_hr`, `pitcher_k`, …) plus an aggregate **ALL PROPS** row.
- **Prop Insights** — per-prop recommendations mined from the ledger:
  **false positives** (favored props that lost -> tighten to raise PPV),
  **false negatives** (faded props that won -> under-rated pockets to reclaim,
  lifting NPV), and **true positives** (favored props that won -> the pockets the
  model nails, to concentrate/size up).
- **Closing Line Value** — per-market CLV: mean points the market moved our way,
  how often the close came to us, and mean EV per unit at the price we took,
  judged by the closing no-vig probability. Only appears for slates where
  `mlb-engine close` captured a closing snapshot.
- **Daily** — per-date Buy (S+M) rollup.
- **Bets** — every graded pick, win/loss/push shaded, with the market's price at
  bet time, the close, and that bet's CLV.

Re-auditing a date replaces that date's rows rather than duplicating them.

### One-click desktop shortcut

Run once, then make a desktop shortcut/alias to the launcher for your OS:

- **Windows**: `scripts\windows\run_predictions.bat` (audit: `run_audit.bat`)
- **macOS**: `scripts/macos/run_predictions.command`
- **Linux**: `scripts/linux/run_predictions.sh`

The launcher runs the model, opens the newest workbook, and (if present) uses the
VSIN CSV at `~/.mlb_engine/vsin_today.csv`.

### Ledger desktop shortcut

A dedicated shortcut opens the running audit **ledger** workbook
(`~/.mlb_engine/output/ledger.xlsx`) directly, and carries a ledger-book icon
(green cover, ruled rows, red debit/credit column rules — `assets/ledger.png`).
Run the installer for your OS once to drop the icon on your Desktop:

- **Windows**: `scripts\windows\install_ledger_shortcut.bat` → `Ledger.lnk`
- **macOS**: `scripts/macos/install_ledger_shortcut.command` → `Ledger.app`
- **Linux**: `scripts/linux/install_ledger_shortcut.sh` → `ledger.desktop`

Double-clicking it runs `open_ledger.*`, which opens `ledger.xlsx` (building it
via an audit first if it doesn't exist yet). Regenerate the icon art anytime with
`python scripts/make_ledger_icon.py`.

### Email the daily workbook

`scripts/email_results.py` emails the newest workbook as an attachment (Gmail
SMTP by default). Set these in your shell profile (`~/.zprofile` on macOS so the
launcher's non-interactive shell sees them):

```bash
export GMAIL_USER=you@gmail.com
export GMAIL_APP_PASSWORD=xxxxxxxxxxxxxxxx   # a Gmail App Password, not your login
export MLB_EMAIL_TO=you@gmail.com            # optional; defaults to GMAIL_USER
```

Then append `python scripts/email_results.py` to your `run_predictions` launcher
(or run it standalone). Non-Gmail servers: override `SMTP_HOST`/`SMTP_PORT`.

## Market data (VSIN / DraftKings / Circa)

`mlb-engine run` writes `output/vsin_template_<date>.csv` listing every priced
selection. Fill in odds + handle/bets from VSIN and re-run with `--vsin-csv`:

```csv
matchup,market,selection,book,american,handle_pct,bets_pct
AZ @ CHC,game_ml,CHC ML,draftkings,132,70,45
AZ @ CHC,game_total,Over 9.5,circa,-105,58,61
```

Without market prices the engine still outputs model probabilities (all Pass).

## FanGraphs tail metrics (SIERA / Stuff+ / wRC+ / xSLG)

xSLG is pulled automatically from the public Baseball Savant leaderboard. The
FanGraphs-owned metrics come from a **custom-report CSV drop-in** (no scraping):
in FanGraphs → Leaders, build a date-ranged report matching the rolling windows
and hit *Export Data* (CSV or XLSX) — one hitters report (wRC+, xSLG) and one
pitchers report (SIERA, Stuff+).

Drop both files in **`~/.mlb_engine/fangraphs/`** and they're picked up
automatically (the one-click launcher needs no flags). Or point anywhere:

```bash
mlb-engine run --fangraphs-csv /path/to/folder
```

Rows are matched to the slate by MLBAM id (FanGraphs exports include an
`MLBAMID` column) and fall back to player name; unmatched players (and any
missing file/metric) simply stay neutral. These feed the ≥2 SD tail layer.
Re-export daily — the engine reuses whatever files are in the folder, so stale
files feed stale numbers.

## Rest-of-season projections (the batter prior)

Every hitter's rate vector is shrunk toward a rest-of-season projection rather
than toward the league mean, so the prior is what keeps a slugger and a backup
catcher apart in a thin window. By default the engine builds its own Marcel off
the free official season lines, but a subscriber's projection is better: drop a
**Standard-view rest-of-season export** (FanGraphs → Projections → *system* →
Rest of Season → Batters → *Export Data*) in **`~/.mlb_engine/projections/`** and
it becomes the prior, with the Marcel covering the hitters it omits.

```bash
MLBE_PROJECTION_SOURCE=atc   # match on the file name; batx, steamer, ... also work
MLBE_PROJECTIONS_DIR=~/Downloads   # or read them where the browser already puts them
mlb-engine ros-prior         # or just run the slate: a new export is picked up at once
```

The export must carry `MLBAMID` (it is the join to Statcast) plus `PA`, `H`,
`2B`, `3B`, `HR`, `BB`, `SO` — all present in the Standard view. Hitters
projected for fewer than 25 PA are left to the Marcel, since a system that
rounds its counting stats to integers turns a two-PA bench line into a .000
hitter.

**Name the files by system** (`atc_ros.csv`, `batx_ros.csv`): the folder is
resolved by finding `MLBE_PROJECTION_SOURCE` as a *word* in the file name — set
off by punctuation, so `Statcast_leaderboard.csv` and `Match_History.csv` are not
`atc` — rather than by taking the newest CSV, so pointing `MLBE_PROJECTIONS_DIR`
at a download folder full of unrelated CSVs is safe. A named system that isn't
there logs a warning and prices off the Marcel rather than guessing.

## Credentials

Subscription logins are read from environment variables and never committed:
`FANGRAPHS_USER/PASS`, `ROTOWIRE_USER/PASS`, `VSIN_USER/PASS`,
`TEAMRANKINGS_EMAIL/PASSWORD`. FanGraphs/Rotowire projections and VSIN quotes can
also be imported from CSV (see `data/`).

TeamRankings is the outside benchmark on the game markets, and it is the one login
without which a run captures *nothing* rather than less: signed out, their grid
serves the last slate already played, with the results filled in. The daemon reads
these from `/etc/engine.env` like every other key.

## Development

```bash
pip install -e ".[dev]"
ruff check mlb_engine tests
mypy mlb_engine
pytest -q
```

## Historical backtest (accuracy / calibration)

`mlb_engine/backtest.py` replays past slates to measure whether the model is
*predictive*, independent of odds. For each historical date it rebuilds every
feature strictly as-of the day before (Statcast rolling windows sliced from a
preloaded season frame; season-to-date Savant leaderboards disabled via
`Pipeline.run(..., enrich_leaderboards=False)` to avoid look-ahead), emits the
model's probability for each market at standard sportsbook lines, and grades it
against the official box score.

```bash
python scripts/run_backtest.py       # downloads one season Statcast frame, replays, pickles picks
python scripts/analyze_backtest.py   # -> calibration, PPV/NPV, FP/FN bias, output/backtest_2024.xlsx
```

Reported per market group: win%, Brier score, reliability/calibration curve,
and a confusion matrix (PPV, NPV, sensitivity, specificity, false-positive vs
false-negative rate) keyed on the model's own 0.5 probability boundary. This is
an accuracy/calibration test only -- a profitability (ROI) backtest additionally
requires paid historical odds.

### Probability calibration

The backtest exposed systematic over-confidence (worst on pitcher props), which
inflates EV and manufactures false positives. `mlb_engine/calibration.py` fits a
per-market isotonic map from the backtest (`scripts/fit_calibration.py` ->
`mlb_engine/data/calibration_2024.json`) that rescales each raw probability onto
its historically realized win rate *before* EV/tier classification. Isotonic is
monotone, so it never flips which side the model prefers -- it only shrinks
over-confident edges (demoting phantom-edge false positives) and lifts
systematically under-rated bands (reclaiming false negatives). Out-of-sample
(train < Aug 1, test Aug-Sep 2024) it cut the mean over-confidence gap from ~7.0
to ~1.2 pts and lifted favored-pick PPV from .554 to .580. The raw probability is
kept on each recommendation (`raw_prob`) for audit. Disable with
`MLBE_CALIBRATE=0`.

### Refitting from your own ledger

The packaged map is a 2024 fit and does not cover every market the engine now
prices -- `batter_tb` had no map at all, so total bases priced off the flatter
pooled curve and ran 15.5 pts over-confident. `mlb-engine calibrate` refits from
`~/.mlb_engine/audit/ledger.csv` (the `raw_prob` column, since the map applies to
raw simulation output), holds out the most recent slates, and adopts the refit
**per market** -- only where it beats the packaged fit out of sample:

```bash
mlb-engine calibrate --holdout 2
```

The result goes to `~/.mlb_engine/calibration_live.json`, which the pipeline
prefers over the packaged file; markets that did not improve keep the packaged
map. Point somewhere else with `MLBE_CALIBRATION_FILE`.

### Confidence shrink

Isotonic only corrects a tail once that tail has graded history. As a standing
guard, everything past `MLBE_SHRINK_PIVOT` (.60) is compressed by
`MLBE_SHRINK_SLOPE` (.55) after calibration: on the eight-slate ledger the .70+
bucket predicted 75.7% and won 59.3%, and that bucket is exactly what trips a
Strong Buy. The shrink is continuous, monotone and never crosses .5, so it
cannot flip a side. It is one-sided: mirroring it about .5 lifts genuinely rare
events (a .10 home-run over becomes .24) and cost Brier on the graded slates.
Disable with `MLBE_SHRINK_TAILS=0`.

### Barrel rate on the singles line

A singles over needs a *single* -- an extra-base hit loses it. Across 323
qualified batters (95k PA) singles per PA falls from .175 under 2% barrel to
.124 at 10-12%, while total hits per PA barely move (.224 vs .219): power
hitters convert hits to extra bases and strike out more. The engine used to
miss this entirely, and the distribution-tail bonus actively made it worse by
lifting 1B, 2B, 3B and HR with the same multiplier.

Two changes, both on by default:

- The singles multiplier carries a barrel term centred on the .080 league
  baseline, `MLBE_SINGLES_BARREL_SLOPE` (1.5, clipped to +-6%). The raw league
  slope is roughly 3.5; half of the effect is strikeouts the simulator already
  carries in each batter's own K rate, so pricing the full slope would
  double-count. Disable with `MLBE_SINGLES_BARREL=0`.
- `TailAdjuster` holds barrel, hard-hit% and xSLG out of the 1B multiplier, so
  an elite-power tail no longer lifts the singles line. Disable with
  `MLBE_TAIL_POWER_SPLIT=0`.

The mirror image -- ground balls being the singles-producing batted ball -- ships
**off**. Enable with `MLBE_SINGLES_GB=1` (slope `MLBE_SINGLES_GB_SLOPE`, 0.5,
clipped to +-6%). Leave-one-slate-out wants a larger coefficient than that, but
it is not separable from zero: over the same eight slates the term moved
`batter_1b` PPV .4861 -> .4880 on seven fewer picks and left the engine flat, so
it is there to accumulate a graded counterfactual rather than to be trusted yet.

## College football: efficiency and talent (`cfb_engine`)

The CFB engine's spine is an SP+/public-model consensus plus situational point
adjustments. Two objective layers now sit underneath it, both measured on
2014-2025 before being wired in (6,513 games with a consensus closing spread;
ratings for a week-*W* game are fit only on weeks before *W*, so nothing has seen
the game it prices).

**Opponent-adjusted per-play efficiency** (`cfb_engine/data/efficiency.py`).
CFBD game PPA (college EPA/play) fed into a ridge
`ppa = mu + off[team] - def[opponent] + hfa*site`. It is a real model -- held-out
margin r 0.56 / MAE 13.1, season-to-season repeatability r 0.69 versus 0.60 for a
ratings fit on scoring margin, correlation 0.88 with SP+ on live 2025 data, and a
fitted home field of 1.7 points.

It nevertheless **adds nothing the closing spread has not already priced**:
partial correlation with margin -0.001 (season-clustered 95% CI [-0.022,
+0.020]), held-out MAE 12.218 -> 12.220, and betting its disagreements goes 50.1%
ATS (-4.4% ROI, i.e. exactly the hold; the biggest disagreements are the worst, at
45.0% past 10 points). So `CFBE_EFFICIENCY_BLEND` defaults to **0** and the layer
earns its place as the *fallback* when SP+ is unavailable, where the engine
previously fell back to market-implied ratings -- i.e. to echoing the market back
at itself. Set `CFBE_EFFICIENCY_BLEND=0.3` to blend it into the SP+ net rating;
the measurement above does not support doing so.

Every other Tier-1/Tier-2 candidate was tested the same way and came back at
zero after the spread: four-year recruiting composite +0.008, line yards +0.017,
points per opportunity +0.013, havoc +0.007, success rate -0.004, explosiveness
-0.012. None of their intervals excludes zero.

**Returning production** (`cfb_engine/data/returning.py`) is the exception, and
the closest anything in either engine has come to beating a price:

    partial r +0.0389, p = 0.001, season-clustered 95% CI [+0.020, +0.058]

present in weeks 4-7 (+0.053) and weeks 8+ (+0.033) as well as September, so it
is not merely a slow-to-update opener. Fitted at **+2.5 points of margin per unit
of returning-production gap** (gap SD 0.34, so ~0.9 points in a typical game).
Betting it goes **51.96% ATS (2663-2462-89, -0.8% ROI)** against a 52.38%
break-even -- four fifths of the vig recovered, and still not through it. So it
ships **off**: `CFBE_RETURNING_PTS=2.5` enables it, capped by
`CFBE_RETURNING_MAX_PTS` (3.0). CLV, not outcomes, should decide whether it ever
ships on.

### Asking the money to agree with the moneyline

The moneyline is the one market where this family of engines has evidence against
its *own* number: in the MLB ledger, graded `game_ml` buys were won less often the
higher the model's EV said they should be (EV AUC 0.33, p=0.004, n=102), while
handle% minus tickets% on the side bet graded AUC 0.80 (p=0.027) -- winners
averaged +19.7 points of divergence, losers -2.6.

So `cfb_engine/data/vsin_splits.py` reads VSiN's public splits (Circa first, then
DraftKings, keyed to date *and* team because the page lists the whole season) and
`market/mlsharp.py` requires a moneyline buy's side to be taking at least as large
a share of the handle as of the tickets. Three limits are deliberate: only the
moneyline consults it, since that is the only market the inversion was measured
on; a side VSiN posts no split for is unaffected, so a data hole cannot become a
veto; and MLB's *upgrade* path (passes with divergence >= +5 won 62% of 32) is not
ported, because promoting a row the EV screen rejected is a new bet justified by
another sport's sample.

Nothing here is measured on college football -- there are no graded CFB rows yet
-- so the default is the weakest form of the finding (divergence >= 0) rather than
MLB's +19.7. Refusals are stamped `ml_no_sharp_money` and graded as a live screen
in the probation table, the stricter +5 bar accrues beside them as a candidate,
and every side's divergence lands in the ledger and on the card whatever market it
belongs to. `CFBE_ML_SHARP_GATE=0` turns the gate back into a measurement,
`CFBE_ML_MIN_DIVERGENCE` moves the bar, `CFBE_VSIN_SPLITS=0` stops the fetch.

### College-football shortcuts and daily schedule

One-click desktop launchers mirror the MLB set. Run the installer for your OS
once to drop **CFB Predictions** (`cfb-engine run` -> emails the Excel card +
article PDF + MP3), **CFB Audit** (`cfb-engine audit` -> grades + emails the
recap), and **CFB Ledger** (opens the ledger workbook) on your Desktop:

- **macOS**: `scripts/cfb/macos/install_shortcuts.command`
- **Linux**: `scripts/cfb/linux/install_shortcuts.sh`

For hands-off operation, install the daily schedule so the card, closing-line
snapshots, and audit run by themselves:

- **macOS** (launchd): `scripts/cfb/macos/install_schedule.command`
- **Linux** (cron): `scripts/cfb/linux/install_schedule.sh`

Default local times: `run` 09:00, `close` 11:00/15:00/19:00/23:00 (repeat-safe
CLV snapshots across the game day), `audit` 03:00. Override the card/audit hours
with `CFB_RUN_HOUR` / `CFB_AUDIT_HOUR`. Both installers are idempotent (re-running
replaces the previous jobs) and shells out through `autorun.{command,sh}`, which
loads credentials from `/etc/engine.env` or `~/.cfb_engine/engine.env` -- neither
launchd nor cron reads your shell profile, so API keys and the Gmail app password
must live in one of those files. The shortcut and schedule installers seed
`~/.cfb_engine/engine.env` from `scripts/cfb/engine.env.example` (chmod 600, never
overwriting an existing file) on first run; fill in `CFBD_API_KEY`,
`THE_ODDS_API_KEY`, `GMAIL_USER`, `GMAIL_APP_PASSWORD`, and `CFBE_EMAIL_TO`. Logs
go to `~/.cfb_engine/schedule.log`.

Remove the schedule with `launchctl unload ~/Library/LaunchAgents/com.payoffpitch.cfb.*.plist`
(macOS) or `crontab -l | grep -vF '# payoff-pitch-cfb-schedule' | crontab -` (Linux).

## Notes / limitations

- xSLG is derived from launch-based expected stats (no clean per-pitch column);
  sprint speed / Stuff+ / Location+ come from separate leaderboards or the
  FanGraphs subscription feed.
- Open-Meteo forecast covers today/near future; older dates (backtests) use the
  historical archive API automatically.
- Baseball is high-variance: an optimized F5 model tops out ~57–62% accuracy.
