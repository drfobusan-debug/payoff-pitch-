---
name: testing-mlb-engine
description: How to run and verify the payoff-pitch MLB engine end-to-end (live priced slate, Excel invariants, threshold env overrides, audit artifacts) when testing tier/pricing/output changes.
---

# Testing the payoff-pitch MLB engine end-to-end

## Environment
- Venv at repo root: `source .venv/bin/activate`.
- Odds pricing reads `THE_ODDS_API_KEY`, falling back to `ODDS_API_KEY`. If neither is set the
  engine silently prices everything off a synthetic -110, which makes any pricing/tier test
  meaningless. Always assert the produced workbook has many distinct `Book Odds` and several
  distinct `Book` names before drawing conclusions (a healthy 15-game slate gave ~780 priced rows,
  ~200 distinct prices, 11-12 books).
- The app's "today" may be a simulated future date; check `date` on the box and use a slate date
  the engine has data for.
- Check remaining Odds API credits before planning multiple full runs:
  `curl -s -D - -o /dev/null "https://api.the-odds-api.com/v4/sports/?apiKey=$ODDS_API_KEY" | grep x-requests`
  A full `run` costs ~135 credits (15 events x 9 markets).

## Commands
- Full slate: `python -m mlb_engine.cli run --date YYYY-MM-DD --sims 800`
  (~2 min; writes `~/.mlb_engine/output/mlb_recommendations_<date>.xlsx` and pushes state to an
  `engine-state` git branch as a normal side effect — don't fight it).
- `python -m mlb_engine.cli audit --date YYYY-MM-DD` only prints metrics and refreshes
  `ledger.xlsx`. **The md/HTML/PDF audit report is only written when you pass `--report`**
  (`--email` also triggers it). Do not conclude the PDF is broken because timestamps did not change.
- Gates: `python -m pytest -q`, `ruff check mlb_engine cfb_engine tests`, `mypy mlb_engine cfb_engine`.

## Verifying tier / threshold behaviour
- Tier config lives in `mlb_engine/config.py` (`EVThresholds`), classification in
  `mlb_engine/market/tiers.py:classify`. Env knobs: `MLBE_MIN_EV`, `MLBE_MIN_EDGE`,
  `MLBE_EDGE_STRONG_GAP`, `MLBE_MAX_EDGE`, `MLBE_STRONG_ONLY`, each with a `_<MARKET>` suffix
  override (e.g. `MLBE_MAX_EDGE_GAME_ML`). Exercise these against `EVThresholds().for_market(...)`
  and `classify()` in a subprocess per env combination — env is read at dataclass construction,
  so setting `os.environ` in an already-imported process is unreliable.
- Cheap way to prove an env knob changes real selections without paying for another slate: read
  the `All` sheet of an existing workbook, rebuild synthetic `EVResult`s from its EV/Edge columns,
  and re-run `classify` under different env in a subprocess.
- Rows that legitimately sit outside the classify gate when auditing a workbook:
  - `Market == comeback` — informational resilience flags tiered in
    `pipeline.py:_comeback_recs`, no EV/edge/odds at all. Exclude them from gate invariants.
  - `Market == game_ml` with `ml-upgrade` in Notes — `MLSharpGate.upgrades`
    (`mlb_engine/features/ml_gate.py`) promotes a vetoed PASS row to Moderate on VSIN sharp money,
    after strong_only is applied, so these can sit outside the edge band. Since the upgrade also
    requires `evres.ev > thr.min_ev`, they must still be positive-EV.
  - Tiers can also be bumped ±1 by run-line signals (`runline_adjustment`), so a Strong row can sit
    below `min_edge + strong_edge_gap` and a Moderate row above it. Don't assert a hard
    Strong/Moderate edge split; assert the gate band instead.

## Which markets are actually priced (and how to reach the gated ones)
- Only `DEFAULT_PROP_MARKETS` in `mlb_engine/data/oddsapi.py` are paid for: batter hits, batter
  singles, pitcher K/outs/hits/walks. **`batter_home_runs`, doubles, runs and RBIs are NOT priced
  by default**, so any code path that only runs on a priced row of those markets (e.g. the
  `HRPowerGate` in `features/hr_gate.py`, which is only consulted when a `batter_hr` row already
  survived classification) is completely inert in a default run and cannot be tested by it.
- To exercise them, override the market list on a live run, e.g.
  `MLBE_ODDS_PROPS="batter_hits,batter_singles,batter_home_runs,pitcher_strikeouts,pitcher_outs,pitcher_hits_allowed,pitcher_walks"`.
  That raises the cost to ~150 credits and, on a 15-game slate, produced ~242 priced `batter_hr`
  rows, 22 of which reached the HR gate (5 kept, 17 vetoed) — enough to test both gate branches.
- `batter_1b` is in `PRICE_ONLY_MARKETS`: it is priced to capture the under quote and its over is
  always hard-passed after classification. Never expect a `batter_1b` buy.
- Post-gate reason strings are the cheapest way to prove which branch ran: `ml-gate: OK` /
  `ml-gate: neutral` / `ml-gate: PASS`, `hr-gate: OK` / `hr-gate: neutral` / `hr-gate: PASS`,
  `ml-upgrade: BUY`. Assert on these in the `Notes` column rather than inferring from counts.
- Watch for local-variable shadowing between gates in `pipeline._mk`: the batter contact-quality
  veto is a `_mk` **parameter** named `gate_reason` and is applied at the very end
  (`if gate_reason is not None and rec.tier != Tier.PASS: rec.tier = Tier.PASS`). A gate that
  assigns its own reason into a local of that same name silently hard-passes every row it
  approved. Symptom to look for: a market whose only surviving buys come from an alternate
  promotion path, or a gate whose `OK` reason string never appears in any workbook.

## Reading the Excel
- `openpyxl` is available. Sheets: `Strong Buys`, `Moderate Buys`, `Fades`, four family tabs, `All`.
- `All` has full columns (`Market`, `EV`, `Edge`, `Book Odds`, `Tier`, `Notes`); the grid sheets have
  a `Best` column and format `Odds` as strings like `+107` / `n/a`, so parse defensively.
- Row shading interpolates a per-class light->neon pair (see `_SCHEMES` in `output/excel.py`) with
  `t = 0.15 + 0.7 * (conv - lo) / (hi - lo)`. To test the gradient, read `cell.fill.fgColor.rgb`
  and solve for `t` by least squares across all three channels — the green channel alone is nearly
  flat in the yellow (Moderate) scheme. Expect inversions of ~0.005 in `t` between rows whose
  displayed edge is equal to 3 dp; that is rounding, not a bug.
- To view a workbook visually: `sudo apt-get install -y --no-install-recommends libreoffice-calc`
  then `libreoffice --calc <path>` (dismiss the "Tip of the Day" dialog and the notification bar
  before screenshotting). PDFs render fine in Chrome via a `file:///` URL.

## Outside-model benchmark columns (VSiN VOLT/JOLT, Opta, THE BAT X, TeamRankings)
These are display-only second opinions. Several live in the same code paths, so read the imports
before testing one: `data/propicks.py` and `data/teamrankings.py` both export
`load_picks`/`merge_picks`/`save_picks`, and `cli.py` disambiguates by aliasing the propicks ones
(`load_propicks`/`merge_propicks`/`save_propicks`). When testing either, assert the capture landed in
its **own** `<audit dir>/<name>_<date>.json` and that the sibling's file was not written.
- Capture command: `mlb-engine propicks [--date] [--league MLB]`. The VSiN pages
  (`https://data.vsin.com/propicks/volt/` and `/jolt/`) are public, need no key, and hold **today
  only** — each card carries its own `fp-date`. `cmd_propicks` refuses (exit 1,
  "VSiN is publishing X, not Y; nothing captured") when `--date` disagrees with the cards, so the
  slate date you test with must equal the live `fp-date`; check it first with
  `curl -s .../propicks/volt/ | grep -o 'fp-date">[^<]*' | sort -u`.
- Captures merge on `date|model|subject|raw_market|side`, so re-running the same day is a no-op and
  the JSON should come out byte-identical. `run` prefers the saved capture and only fetches if it is
  absent; both paths filter by slate date.
- An unrecognised VSiN market label is deliberately kept as a pick with `market == ""`, listed under
  `markets not mapped to an engine bet:`, and matches nothing. To test, save the live HTML and
  re-serve a copy with a renamed `class="fp-market">` label via a monkeypatched
  `mlb_engine.data.http.get`.

## Proving a display-only change did not touch pricing
**Do not diff two full runs of the same slate — they are not comparable.** `hours_to_first_pitch`
advances between runs and cascades into `xrd` -> `model_prob` -> `edge`/`ev`/`tier`; two runs 15
minutes apart differed on `model_prob` for 4752 of 6705 rows and flipped 64 tiers, with none of it
caused by the feature under test. The live VSiN splits / Opta feeds also move, changing
`handle_pct`/`bets_pct`/`opta_prob`, and `line` moves shift the join key on ~60 rows.
Instead, prove it in-process and deterministically:
1. Load a real run's `predictions_<date>.json` with `recommendations.load_json` (it is `asdict(rec)`,
   so full precision, all fields).
2. Snapshot every dataclass field, call the exact function `cmd_run` calls
   (e.g. `cli._annotate_propicks(cfg, recs, slate_date)`), then diff every field.
3. Assert the only mutated fields are the benchmark's own, and that the no-picks leg mutates nothing.
Note the annotation runs **after** `pipe.run()` and before `write_workbook`, so ordering already
guarantees pricing is settled; the field diff is what proves nothing else is written.
Also useful: blocking a host to simulate an outage must be scoped to the **URL path**, not the host —
`data.vsin.com` also serves the betting splits and Opta projections that genuinely feed the model, so
blocking the whole host changes pricing and invalidates the comparison. Block `"/propicks/" in url`.
A second full run within 30 min is free and price-stable: the Odds API responses are disk-cached with
`cache_ttl=1800` (`data/oddsapi.py`), so `x-requests-remaining` does not move.

## Card and Excel checks for a new column
- `run` writes `card_<date>.{md,html,pdf}` **only with `--card`** (or `--email`).
- The ★/✗ legend sentence is emitted only when some play carries a mark, so test both directions:
  with the capture present the legend must appear exactly once, and with the benchmark unavailable
  the card must contain zero marks, no legend, and no doubled space where the clause was.
- Header/value alignment is worth asserting by **type**, not by eye: read each sheet's header row and
  compare to `output/excel.COLUMNS` (the `All` tab) and `GRID_COLUMNS` (grid tabs), then check that a
  `VOLT `/`JOLT `-prefixed string never appears under any header other than `VSiN Pick` and that
  marks are only ever in {"", "★", "✗"}. A shifted header still "looks fine" in a screenshot.
- Known cosmetic drift: the `All` sheet's `widths` list in `output/excel.py` is hard-coded and
  shorter than `COLUMNS` (21 vs 27), so the trailing widths land on the wrong columns (`Notes` ends
  up narrow, `Opta %` 40 wide). Values are written by header name and stay correct; check whether the
  list has been extended before reporting it as new.

## Verifying a devig / fair-price change on a real board
- `fair_prob` is persisted per selection in `predictions_<date>.json`, so the board-wide invariant
  is cheap: group `game_ml` by matchup (and `game_rl` by pairing home `-1.5` with away `+1.5`,
  `game_total` by over/under) and assert each pair's `fair_prob` sums to 1.000 +- 0.001. A summed
  over-round (e.g. ~1.0055) means some quote in the consensus still carries its vig.
- To show the *delta* a devig change makes without a second paid run, rebuild the identical live
  board in-process — `MLBStatsClient().get_slate(d)`, `VSINClient(cfg.creds).fetch(slate)`,
  `cli._odds_client(cfg).fetch(slate, include_props=False)`, `pipeline._merge_quotes(odds, vsin)` —
  then per key compare `evaluate(0.5, qs).fair_prob` against the old formula recomputed on the same
  quotes with `opposite_american` stripped from the VSIN books (`{"circa", "draftkings"}`). The
  Odds call is served from the disk cache, so it is free.
- Useful companions on the same harness: `evaluate(...).best_quote.american` must equal
  `max(qs, key=american_to_decimal)` even when that quote has `devigged == False` (line shopping),
  and `devig_coverage` must equal the book-weighted devigged share of the **whole** quote list, not
  of the consensus subset.

## Exercising a threshold gate the live slate does not reach
- A shipped threshold can be completely inert on a given day. On 2026-08-16 the highest batter-prop
  OVER buy was `model_prob` 0.505, so a 0.62 ceiling fired zero times — "no buy above the ceiling"
  passes vacuously. Before concluding, print the max `model_prob` among the buys the gate targets.
- The fix is to run the same live slate again with the knob moved into the populated part of the
  distribution (e.g. `MLBE_BATTER_MAX_BUY_PROB=0.40`) and a third time with it disabled (`=1`).
  Odds responses are cached for 30 min, so the extra runs cost no credits.
- `run` overwrites `~/.mlb_engine/audit/predictions_<date>.json` every time — copy it to a scratch
  path immediately after each run, and pass `--out /tmp/runX.xlsx` so the workbook is not clobbered.
- Joining two runs on `(matchup, market, selection, line, side)` is stable, and with the Monte Carlo
  seeded (`Pipeline(..., seed=7)`) back-to-back runs within the cache window gave **identical
  `model_prob` on all 6705 rows**, which is what makes "the screen moves no probability" provable.
  Expect a couple of unrelated `pass_gate` flips on `game_ml` rows anyway: the VSIN handle/bets
  splits and `hours_to_first_pitch` move between runs and change which ML gate claims the PASS.
- `pass_gate` reaches the ledger only via `~/.mlb_engine/audit/ledger.csv` (the `Bets` sheet of
  `ledger.xlsx` has no such column), and only for **graded** dates — today's refusals cannot be in
  it before the games finish. To prove gradeability the same day, feed the refused recs to
  `audit.ledger.entries_from_graded` and `gate_metrics` in-process and check a
  `GATE <name>` bucket appears.
- A new `CANDIDATE_SCREENS` entry in `audit/probation.py` shows up in `mlb-engine audit --report`
  output under the probation block (`WATCHING <name> n=… ROI=…`); grep for its name there.

## Before/after on the same live board when the change alters pricing
- Some changes are *supposed* to move probabilities (fitted priors, model coefficients), so the
  "prove nothing moved" recipe above is the wrong shape. Run the same slate twice — old code first,
  new code second — and quantify the move. The Odds API cache (`cache_ttl=1800`) makes the second
  run free and prices it off the identical board, and the Monte Carlo is seeded, so the only
  difference is the code.
- If the PR is already merged, the "before" is the merge commit's first parent. Use a second
  worktree rather than switching branches (which would disturb the checkout the lead is using):
  `git worktree add /tmp/pp-before $(git rev-parse origin/main^1)`. The venv's editable install
  points at the main checkout, but **cwd wins**: `cd /tmp/pp-before && <repo>/.venv/bin/python -m
  mlb_engine.cli run ...` imports the worktree's code. Assert it before trusting the run, e.g.
  print `mlb_engine.__file__` and the constants under test from both cwds.
- Run the "before" leg with `--out /tmp/before.xlsx` and **without** `--card` so the shipped
  workbook/card in `~/.mlb_engine/output` come from the new code; copy
  `audit/predictions_<date>.json` after each leg (it is overwritten every run).
- Joining the two runs on `(matchup, market, selection, line, side)` leaves ~80 unmatched batter
  rows when a lineup changes between legs (all confined to the affected game). Report the
  `only_before`/`only_after` counts and check which game they belong to before blaming the change.
- Cheapest sanity anchor for a probability that moved a lot: compare it with the book's own price
  on the same row (`market_american`). A prop that moved from 5% to 42% on a +109 board moved
  *toward* the market, which is evidence the old number was the broken one.

## Pitcher-prior style changes (xK% / xBB%)
- The priors live in `features/regression.py` (`XK_*`, `XBB_*` constants, `XK_INTERCEPT` derived at
  import from the anchors) and are consumed in `pipeline._team_offense`:
  `build_pitcher_regression(statcast[statcast.pitcher == id], shrink=w.starter_contact_shrink)` then
  `blend_k_rate(pit_prof.allowed, reg.expected_k_pct())` / `blend_bb_rate(..., expected_bb_pct())`
  at a 150-PA prior weight. Mirror exactly that to dump per-starter values.
- Both slope sets can be evaluated in one process: monkeypatch the module constants and
  **recompute `R.XK_INTERCEPT`** the same way the module does, otherwise the old numbers are wrong.
- `OutcomeRates` fields are `p_k`/`p_bb` (not `k`/`bb`).
- Clip saturation is the thing to check: `expected_k_pct` clips to [0.08, 0.42] and
  `expected_bb_pct` to [0.02, 0.20]. Print how many starters sit within 1e-9 of a bound under each
  slope set — on the 2026-08-16 slate the old slopes pinned 5 of 30 arms, the fitted ones 0.
- A starter with 0 pitches in the window falls back to the league baselines and is identical under
  any slopes (`Edward Cabrera` on that slate) — expect a few no-change rows.
- K-total props are threshold bets on a count distribution, so a ~10pp move in the blended allowed
  K rate can swing an over/under probability by 25-37pp. Judge the prop move by the underlying rate
  move, not by the prop delta alone.

## Baseline choice when the branch is behind main
- `git merge-base origin/main HEAD` first. If `git log --oneline HEAD..origin/main` is non-empty, main
  carries other merged PRs and a main-vs-branch comparison conflates them. Use the **merge-base**
  commit as the "before" worktree instead and say so in the report; `FEATURE_BASIS` is a quick tell
  that main has moved on (each pricing PR rewrites it).

## Multiplier-style changes (`k_multiplier`, bullpen NPV)
- Starter path: `pipeline.py` `build_pitcher_regression(pit_rows, shrink=w.starter_contact_shrink)`
  -> `k_multiplier()`, applied as `apply_multipliers(vs_start, {"K": k_mult})`.
  Pen path: `build_bullpen_profile(statcast, abbrev, date, w.bullpen_days, w.bullpen_min_inning,
  skill_days=w.bullpen_skill_days, xwoba_shrink=w.bullpen_xwoba_shrink,
  prior_strength=PEN_PRIOR_STRENGTH if cfg.pen_shrink else PRIOR_STRENGTH)` ->
  `build_pitcher_regression(bpen.skill_frame, bullpen=cfg.pen_contact_level)` -> `k_multiplier()`,
  and `bpen.npv_multipliers(avail)["BB"]`. Mirror those calls exactly in a harness.
- Old vs new in one process: monkeypatch only the baseline constants (e.g.
  `BL_TWO_STRIKE_WHIFF = 0.280`, `BL_PEN_K_MINUS_BB = BL_K_MINUS_BB`) — that reproduces the previous
  code path exactly without a second worktree, since the arithmetic is unchanged.
- Check **term-level** clipping, not just the product: recompute the term the module computes
  (e.g. `(two_strike_whiff - BL) * 0.8 <= -0.06 + 1e-12`) and count how many arms sit on it under
  each constant set. Also count arms on the product clip `[0.75, 1.30]` — a baseline correction can
  push an elite arm onto the *upper* product bound (Skubal, 2026-08-16).
- Arms with `pitches < 100` (and a 0-pitch probable) return a flat 1.0 and never move; expect a few
  no-change rows and exclude them when comparing means to an offline study.
- A slate's ~30 probable starters is a different population from an offline "201 arms with 400+
  pitches" study: compare the *shift* (mean delta) rather than the absolute mean level.
- `ruff check` at repo root also lints untracked scratch scripts under `scripts/`; lint
  `git ls-files '*.py'` instead, and re-run on the baseline worktree to prove any hit is pre-existing.

## Run-to-run noise floor (mostly fixed as of the weather cache, but still check)
- History: before the weather cache (PR #196), two **identical** runs off the same frozen odds board
  differed on 6,050 of 6,705 rows, mean |Δ| 1.23pp, max 6.88pp, because `WeatherProvider.fetch` was
  called live every run and `wx_hr_mult` / `wx_summary` / `xrd` / `xrd_sd` moved with it.
- **Now**: `WeatherProvider` caches the Open-Meteo hourly payload under
  `~/.mlb_engine/cache/weather/` (`MLBE_WEATHER_CACHE_TTL`, default 1800s; past dates are immutable
  and never refetch). Two back-to-back runs are byte-identical on `model_prob`, `raw_prob`,
  `bet_prob`, `ev`, `edge`, `tier`, `xrd`, `xrd_sd`, `wx_*`. So the **same-code control leg is no
  longer mandatory** for before/after work — but do keep both legs inside the TTL, or set
  `MLBE_WEATHER_CACHE_TTL=86400` alongside `MLBE_ODDS_CACHE_TTL=86400`, otherwise the forecast
  refreshes mid-comparison and the old noise is back.
- **Residual nondeterminism that is NOT weather**: `fair_prob` / `edge` on `game_ml` (~14 rows) can
  still move between runs because the VSIN moneyline scrape is live ("170 vs 171 handle entries").
  Model-side fields do not move. Don't report that as a regression, and don't assert 0 diffs on
  `fair_prob` for `game_ml`.
- Freeze the board: `MLBE_ODDS_CACHE_TTL=86400` makes every leg reuse the same cached Odds API
  responses regardless of how long the legs take (the default TTL is 1800s and a 3-leg comparison
  easily exceeds it), and costs no extra credits.
- If you still need a hard freeze (e.g. an exactness proof spanning hours):
  `WeatherProvider.fetch = lambda self, park, dt: WeatherEffect(None, 1.0, note="frozen")` before
  `from mlb_engine.cli import main; main(["run", "--date", ...])`.

## Proving a code path is *gone* (not just changed)
- The strongest evidence is a **call counter plus a poison run**: monkeypatch the retired function
  (e.g. `PitcherRegression.k_multiplier`) to increment a counter and return an absurd value, run the
  full slate, and assert (a) the counter is 0 and (b) with weather frozen, every `model_prob`,
  `raw_prob`, `ev`, `edge` and `tier` is identical to a clean run. A plain before/after diff cannot
  distinguish "no longer applied" from "applied with a different value".
- Grep for surviving callers to back it up: `grep -rn "k_multiplier" --include=*.py mlb_engine scripts`
  should show only the reporting script.

## Proving a cache is actually used (and load-bearing)
- Counting is not enough on its own: wrap `requests.Session.get` and filter on the host
  (`"open-meteo" in url`) to count real HTTP, and wrap the provider method (`WeatherProvider._hourly`)
  to label each call HIT/MISS by whether the HTTP counter moved. Cold run = N misses + N files;
  warm run = N hits + 0 HTTP.
- Then **poison the network path** (return an absurd payload, e.g. 110F / 35mph straight out) and
  re-run: with the cache warm nothing changes and the poison is never served.
- Crucially, also run the **negative control** — same poison with the cache bypassed
  (`MLBE_WEATHER_CACHE_TTL=0`) — to prove the poison *would* have shown through. On 2026-08-16 that
  moved 6,147 of 6,705 rows (mean 3.3pp, max 18.5pp) and `wx_hr_mult` on 6,258 rows. Without this
  control, "nothing changed" is indistinguishable from "the poison was inert".
- Beware the timing trap: late in the day the real forecast has settled, so two uncached runs can
  agree and a naive before/after control is a null result. The poison + negative control pair is
  what makes the verdict robust.

## Devin Secrets Needed
- `ODDS_API_KEY` or `THE_ODDS_API_KEY` — required for real market prices.
- `GMAIL_USER` / `GMAIL_APP_PASSWORD` — only needed for `--email`; do not send email while testing.
