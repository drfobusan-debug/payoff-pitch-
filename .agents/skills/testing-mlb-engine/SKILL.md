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

## Testing a new pricing *term* that is already merged into main
When the PR adds a term to a model (e.g. `XK_SHAPE_COEF * shape_plus` in `expected_k_pct`) and is
already on `main`, you do not need a merge-base worktree. Neutralise the term in-process instead and
run the same slate twice off one frozen board — it reproduces the pre-PR arithmetic exactly:

    from mlb_engine.features import stuff
    stuff.shape_plus = lambda pdf: 0.0        # also patch the alias the caller holds:
    import mlb_engine.features.regression as R; R.stuff.shape_plus = lambda pdf: 0.0
    from mlb_engine.cli import main; main(["run", "--date", ..., "--sims", "800"])

Always run all legs with `MLBE_ODDS_CACHE_TTL=86400` **and** `MLBE_WEATHER_CACHE_TTL=86400`; with
both frozen the noise floor is exactly 0 rows, so any diff is the term.

To prove the feature is populating rather than silently returning its "no opinion" value, wrap the
builder during the real run rather than trusting an offline snippet — `build_pitcher_regression` is
imported by name into `pipeline`, so patch **both** `regression.build_pitcher_regression` and
`pipeline.build_pitcher_regression`. Record per call: `pdf["pitcher"]` ids (1 id = starter,
many = the bullpen frame, which is graded too), `len(pdf)`, the feature value, and the target metric
recomputed with the new term removed. Then assert the arithmetic exactly
(`xk - xk_no_term == coef * clip(value)` to 1e-12) and that the value is non-default for every arm
above the feature's own floor (`stuff.MIN_PITCHES = 100` graded pitches). Statcast caches have no
`player_name` column: map ids via `MLBStatsClient().get_slate(date)` probables, falling back to
`https://statsapi.mlb.com/api/v1/people/<id>` for arms swapped after the run.

A data-file-backed feature gives you a free third leg: move the JSON aside
(`mlb_engine/data/*.json`), re-run, and assert the output is **byte-identical to the neutralised
leg** — that proves the fallback is genuinely "no opinion" and not merely non-crashing. Restore the
file and check its sha256 afterwards.

**Check `FEATURE_BASIS` on any pricing PR.** `mlb_engine/calibration.py` keys the calibration map to
that string; if a PR moves prices (e.g. 3,906 of 4,023 rows) without bumping it, a map refit on the
older engine still matches the basis and would be applied to prices it was not fit on. Report it —
the run-time stale-basis warning will *not* fire in that case.

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

## Devin Secrets Needed
- `ODDS_API_KEY` or `THE_ODDS_API_KEY` — required for real market prices.
- `GMAIL_USER` / `GMAIL_APP_PASSWORD` — only needed for `--email`; do not send email while testing.
