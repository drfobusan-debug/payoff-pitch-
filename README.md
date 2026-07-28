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
10. **Excel output** + **nightly audit** (sensitivity / specificity / PPV / NPV
    per tier from final box scores).

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

# grade yesterday, update the scorecard, and append to the running ledger
mlb-engine audit

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

- **Overall** — cumulative sensitivity / specificity / PPV / NPV / win% / ROI /
  net units. The first row is the **whole engine** (`ENGINE (p>=.5)`): PPV/NPV
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
- **Daily** — per-date Buy (S+M) rollup.
- **Bets** — every graded pick, win/loss/push shaded.

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

## Notes / limitations

- xSLG is derived from launch-based expected stats (no clean per-pitch column);
  sprint speed / Stuff+ / Location+ come from separate leaderboards or the
  FanGraphs subscription feed.
- Open-Meteo forecast covers today/near future; older dates (backtests) use the
  historical archive API automatically.
- Baseball is high-variance: an optimized F5 model tops out ~57–62% accuracy.
