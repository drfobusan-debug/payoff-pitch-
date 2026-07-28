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
9. **Market + EV** — no-vig fair prob, EV per $1, Strong/Moderate/Pass tiers
   (`MLBE_MIN_EDGE` thin-edge guard, `MLBE_STRONG_ONLY` for strict selection).
9a. **Run-line NPV gates** — veto (not nudge) selections whose failure to cover
    is highly predictable. All off by default; see below.
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

Calibrate the thresholds against this engine's own scales, not FanGraphs/Savant
leaderboards: hard-hit% and GB% are computed off the tracked-batted-ball slice
(same convention as `build_pitcher_regression`) and read a few points lower than
the public versions, and WHIP is the PA-derived proxy. On a full 15-game slate
the published defaults (.45 hard-hit, .50 GB) fired on 0 of 60 run lines, while
.25 / .40 fired on 12 — start loose enough to get a sample, then tighten on what
the `VETO` rows show.

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

# grade yesterday, update the scorecard, and append to the running ledger
mlb-engine audit

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

### Market anchoring

`MLBE_MARKET_ANCHOR` (default `0`, off) blends the devigged market price into the
probability the EV screen bets on: `0` bets the model alone, `1` bets the market,
`0.4` moves the market 40% of the way toward the model. The model's own
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
shrinks a loss rather than earning a profit. It ships off; pick a weight on CLV,
not on ROI.

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

## Credentials

Subscription logins are read from environment variables and never committed:
`FANGRAPHS_USER/PASS`, `ROTOWIRE_USER/PASS`, `VSIN_USER/PASS`. FanGraphs/Rotowire
projections and VSIN quotes can also be imported from CSV (see `data/`).

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

## Notes / limitations

- xSLG is derived from launch-based expected stats (no clean per-pitch column);
  sprint speed / Stuff+ / Location+ come from separate leaderboards or the
  FanGraphs subscription feed.
- Open-Meteo forecast covers today/near future; older dates (backtests) use the
  historical archive API automatically.
- Baseball is high-variance: an optimized F5 model tops out ~57–62% accuracy.
