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

## Replaying a past slate off the odds cache (deterministic A/B for a pricing/gate change)
When a change alters *pricing or tiering*, the in-process field-diff above does not apply and two
live runs are not comparable. Replay one past slate instead: identical real prices, zero credits, and
a **zero** noise floor (verified: `main` vs the branch with the new screen lifted differed on 0 of
6705 rows, every field).
1. Pick the slate by evidence, not by date: read `~/.mlb_engine/audit/predictions_<date>.json`
   (`asdict`, all fields incl. `tier`/`pass_gate`/`market_american`) and count the rows the change
   should move. Prop-rich slates are the ones run near first pitch; a run made ~18h out prices almost
   no props at all (lineups unposted), so **today's early run is usually the wrong testbed**.
2. `MLBE_ODDS_CACHE_TTL=99999999` makes the disk cache authoritative. Per-event prop responses are
   keyed by event id and survive for days, but the **bulk game board (`{BASE}/odds`) is keyed on the
   query alone**, so today's run has overwritten the one for the replay day and the run will resolve
   0 events and price nothing. Rebuild it: scan `~/.mlb_engine/cache/oddsapi/*.json` for dicts whose
   `commence_time` starts with the replay date, emit a list of their
   `{id, sport_key, commence_time, home_team, away_team, bookmakers: []}`, and write it to
   `sha256(json.dumps({"url": f"{BASE}/odds", "regions": "us", "oddsFormat": "american",
   "markets": ",".join(_GAME_MARKETS)}, sort_keys=True))[:20] + ".json"`. Caveat to disclose:
   game-market (h2h/spread/total) prices are then absent; **prop prices are the real captured ones**.
   `run` uses `pregame_only=False`, so a finished slate still prices.
3. Isolate state or the runs are not repeatable: `MLBE_DATA_DIR=/tmp/mlbe_replay` (symlink `cache`
   contents, `projections`, `ros_hitters.csv`, `calibration_live.json`; copy `output/vsin_template_*`),
   `MLBE_STATE_SYNC=0` so nothing is pushed to `engine-state`, and **re-copy the audit dir before
   every variant** — `board_<date>.json` is written by each run and feeds the CLV/drift gate.
4. Freeze the clock: patch `mlb_engine.pipeline.hours_to_first_pitch` to pass a fixed `now=` before
   importing `cli.main`. Without this the variants are not comparable (see the section above).
5. Run `main` from a `git worktree add /tmp/main_wt HEAD^` so the branch checkout is left alone.
6. Run the variants and diff by `(matchup, market, selection, line, side)`. Always include a
   **gate-lifted** variant (e.g. `MLBE_<X>_MAX_BUY_ODDS=100000`): its diff against `main` is the noise
   floor and should be empty, which is also the proof the change is inert when disabled.
7. Set the threshold to a value that *splits the real rows* (e.g. a ceiling of 500 when the slate has
   buys at +485, +500 and +502) — that tests an exclusive bound on live data rather than in a unit
   test.

## First-five (F5) markets, and testing an opt-in model swap (`MLBE_F5_FROM_SIM`)
- Structure is fixed at **9 F5 rows per game** (`pipeline.py`): 3 `f5_ml` (home/away/tie), 4
  `f5_total` (4.5/5.5 x over/under), 2 `f5_rl` (+/-0.5). A slate of 15 games therefore has exactly
  135 F5 rows — assert the per-game count, not just the total.
- **F5 book quotes come from the per-event request** (`_F5_MARKETS` =
  `h2h/spreads/totals_1st_5_innings`), *not* the bulk board. A board pulled early in the day very
  often has **zero** F5 quotes, so every F5 row lands `Tier=Pass`, `Odds=n/a`, blank EV/Edge, and the
  card shows no F5 pick. That is legitimate, not a bug — but it means a fresh live run may be unable
  to prove any *priced* F5 behaviour. Reprice a **cached prop-rich past slate** (rebuild the bulk
  board per the section above) to get priced F5 rows; a 2026-08-16 reprice gave 66 priced F5 rows
  across 9 books, including F5 Strong buys, at 0 credits.
- For an opt-in model-swap flag, prove the flag *actually switches implementations* before trusting
  any diff: wrap both functions (e.g. `pipeline.f5_from_lineups` / `pipeline.f5_from_sim`) in the
  runner and assert one is called once per game and the other **zero** times in each variant. A no-op
  flag otherwise produces a clean-looking "no unintended rows moved" result.
- Isolation diffs must join on `(matchup, market, selection, line)` and compare `model_prob`,
  `raw_prob`, `bet_prob`, `ev`, `edge`, `tier`, `pass_gate`, `veto_gate`, `market_american`, `book`.
  Expect the derived fields to move only where the market is priced: swapping the F5 model moved
  `model_prob`/`raw_prob` on all 135 F5 rows but `bet_prob`/`ev`/`edge` on only the 66 priced ones.
- Sanity band for F5 projected totals is roughly 3-8 runs, but **check the baseline model too before
  calling an outlier a bug**: the highest-total park produced a mean F5 total of 10.2 with the flag
  off and 9.5 with it on, i.e. the outlier is a property of the slate, not of the new model.
- Excel: F5 rows live in the `First-5 (F5)` sheet (and `All`). The probability column is
  **`Model %`** (there is no `Prob` column) and unpriced rows carry `Odds = "n/a"` with a
  `no market price` note — parse defensively. Audit md/HTML/PDF label them `First-5 moneyline` /
  `First-5 run line` / `First-5 total`, and only `audit --report` writes those files.

## Where a `pass_gate` shows up (and where it does not)
- Predictions JSON: yes, `rec.pass_gate` verbatim. This is the primary evidence.
- Excel: there is **no** `pass_gate` column. The evidence is `Tier == "Pass"` plus the reason string
  appended to `Notes` (e.g. `doubles-price-ceiling: PASS (+340 at or beyond +300)`); assert the price
  inside the note equals that row's own `Book Odds`.
- `ledger.csv` is **not** written by `run` — it stays byte-identical. Rows (and `pass_gate`) are
  written by `mlb-engine audit --date <date>`, which grades `predictions_<date>.json` from the same
  data dir. Run it in the scratch dir to prove the `screen_probation` dependency; `screen_probation`
  itself needs a graded window and returns nothing for a single day, so assert on the ledger rows.
- A new gate placed early will **re-attribute** rows that a later screen would also have refused
  (`contact_floor`, `clv_drift` …). Expect the gate's row count to exceed the number of buys it
  actually removed, and separate the two in the report.

## Testing state publication / pull (`engine-state`) without touching production
The `engine-state` branch is the production record — never push test predictions to it. Build a
throwaway origin instead; the state code only ever talks to `origin` of the checkout it is run from
(`state.repo_root()` = `git rev-parse --show-toplevel` of the **cwd**), so:
1. `git init --bare /tmp/x/origin.git`; `git clone <repo> /tmp/x/boxA` then
   `git -C /tmp/x/boxA remote set-url origin /tmp/x/origin.git`. Assert
   `git remote get-url origin` before every push. A second clone (`boxB`) with the same fake origin
   is a second machine; the state worktree path is `repo.parent/.<repo-name>-engine-state`, so
   distinct clone names keep concurrent boxes from colliding.
2. One scratch data dir per box: `MLBE_DATA_DIR=/tmp/x/dataA` (symlink `cache` entries,
   `projections`, `fangraphs`, `batx`, `evanalytics`, `calibration_live.json`, `ros_hitters.csv`;
   copy `output/vsin_template_*.csv`; copy the real `audit/*` non-prediction files if the run needs
   boards/closes). Delete `ledger.csv`/`scorecard.csv` when you want to count a single date's rows.
3. `mlb-engine state push|pull [--date]` drives sync explicitly — much cheaper than a full `run`.
   `run` pushes (cli.py), `audit` pulls first and pushes after grading.
4. Published path on the branch: `mlb/predictions/predictions_<date>.json.gz`. Inspect it by
   `git clone --branch engine-state /tmp/x/origin.git`, then compare `state.card_lead_hours()`,
   row counts and json equality against the local card; md5 the `.gz` to prove "unchanged".
5. Real cards are the best fixtures: `~/.mlb_engine/audit/predictions_<date>.json` (the run's own
   card) and `predictions_<date>.pregame.json` (the copy pulled from the branch) often have very
   different lead hours, including a **negative** lead when the local file is the audit's
   after-the-fact re-price. Copy them into a scratch data dir instead of generating new ones.
6. To make a genuinely later pregame card of *today's* slate without waiting for first pitch: run
   once live, then re-run off the cache (`MLBE_ODDS_CACHE_TTL` huge, 0 credits) with
   `hours_to_first_pitch` frozen a couple of hours before first pitch. The lead hours differ, which
   is what the publication rule keys on.
7. Adversarial cases worth including: push an *earlier* card after the later one (must not change
   the branch), push a card with `hours_to_first_pitch` stripped (untimed → must not change it), and
   write an earlier card onto the branch directly and pull (a box must not regress to it).

## Auditing the priced ("money") section of the Audit Desk report
The money section (`mlb_engine/audit/priced.py` + `output/audit_insight.py`) is the only part of the
report that claims betting P/L; everything else (PPV/NPV, discriminants) is classification. Test them
separately, and never let a good PPV number stand in as evidence the money math is right.
- It is driven by `history=all_entries` (the **cumulative** ledger) passed from `cli.py:cmd_audit`,
  not by the graded slate, so a single-day ledger produces an almost empty money table while the PPV
  tables look full. Seed the scratch data dir with a **multi-slate** `audit/ledger.csv` or the section
  is untested.
- The filter is `source == engine` **and** `tier in {"Strong buy", "Moderate buy"}` **and**
  `odds is not None` **and** `result != "push"`. The tier strings come from `market/tiers.py:Tier` and
  are `"Strong buy"`/`"Moderate buy"` — filtering on `"Strong"`/`"Moderate"` silently yields **zero**
  rows and a false pass. Check the enum values before writing any independent recomputation.
- Recompute independently straight off the CSV: `n`, wins, `mean(1/american_to_decimal(odds))` for the
  price-implied breakeven, `sum(pnl)` for units, one-way count (`under_odds` empty), and CLV /
  beat-close counts. Compare at the rendered precision (0.1 pt / 0.1u); the HTML is the easiest
  source to diff (`output/audit_insight_<date>.html`) and the PDF is the same content.
- Cheapest contamination proof, no fixtures needed: load the real ledger, `dataclasses.replace` one
  priced buy into four poison rows (`tier="Pass"`, `odds=None`, `result="push"`,
  `source="teamrankings"`) each with an absurd `pnl`, append them, and assert
  `engine_priced_stat(...)` returns an identical `n` and `units`. That catches an accidental widening
  of the filter far more directly than comparing pool sums.
- Beware two rounding traps when cross-checking probabilities: `ledger.csv` stores 4 dp and the Excel
  `Model %` column 3 dp, so exact equality fails on a few rows legitimately. Use a tolerance of half
  the last digit and, better, assert the value is **nearer the raw model prob than the shrunk one** —
  that is the assertion that actually distinguishes the two.
- Degenerate case to always run: a scratch data dir with **no** `ledger.csv` and a
  `predictions_<date>.json` whose `market_american`/`opposite_american` are all `null`. The section
  must print the "no graded row … carries both a buy tier and a real price" callout, exit 0 and emit
  no `nan`. When grepping the rendered text for `nan`, anchor the pattern — `correlations` contains
  `nan` and produces a false failure.

## Fitted market anchors (`Config.market_anchor_file`, `MLBE_MARKET_ANCHOR_*`)
- Precedence is env `MLBE_MARKET_ANCHOR_<MARKET>` > fitted JSON file > packaged
  `_MARKET_ANCHOR_BY_MARKET` (only `game_total`/`f5_total` are pinned) > global `market_anchor`.
  Test one **subprocess per case** — `Config` reads env at construction. A corrupt file must log
  `ignoring <path>: ...` at WARNING and fall back, never raise.
- The file default is `<data_dir>/market_anchor_live.json`, i.e. the real `~/.mlb_engine` in a default
  session. Always point `MLBE_MARKET_ANCHOR_FILE` at a temp path before running anything with
  `--write-anchors`, and assert the real one still does not exist afterwards.
- The anchor is applied in `pipeline.py` as `bet_prob = anchor_to_market(model_prob, fair_prob,
  anchor)`; `rec.model_prob` is deliberately left as the **raw** model number. So the correct
  assertions are: `model_prob` bit-identical across variants, `bet_prob` equal to
  `fair + (1-anchor)*(model - fair)`, and no non-target market moving at all. Anchoring `game_ml`
  moved `bet_prob`/`ev`/`edge` on all 30 game_ml rows, `pass_gate` on 10 and `tier` on 2, and nothing
  else, on a 15-game slate.
- Always include the env-override-to-0.0 variant: its diff against baseline must be **empty**, which
  is simultaneously the noise floor of the harness and proof the feature is inert when disabled.
- Replays of a cached slate cost 0 credits — verify with `x-requests-remaining` before and after
  rather than trusting the log line, which prints an estimate (`~240 credits`) even on a pure cache
  hit and looks alarming.

## `scripts/market_shrink_study.py`
- Its `frame()` filter (engine, win/loss, two-sided, `fair_prob` present) is much narrower than the
  raw ledger: a 109k-row / 31-slate CSV reduced to 20,853 rows / **21** slates / 12 markets. Expect
  the slate count in the output to be lower than the ledger's date count; that is the filter, not a
  bug.
- To prove there is no lookahead, import the script as a module and wrap `fit_alpha`, `alphas_for`
  and `apply_alphas`, recording `max(train.date)` against the day being graded; assert strict `<` on
  every call, and that the number of refits equals `len(dates) - max(5, int(0.4*len(dates)))` minus
  days with <20 rows. When importing by path, register the module in `sys.modules` **before**
  `exec_module` or its `@dataclass` definitions fail with `'NoneType' object has no attribute
  '__dict__'`.
- `--write-anchors` writes `1 - alpha`; cross-check each entry against the alphas printed in the
  single-split table (alpha 0.23 → 0.767, alpha 0 → 1.0). Pass `--no-walk-forward` to keep the run
  to about a minute.

## Devin Secrets Needed
- `ODDS_API_KEY` or `THE_ODDS_API_KEY` — required for real market prices.
- `GMAIL_USER` / `GMAIL_APP_PASSWORD` — only needed for `--email`; do not send email while testing.
