# MLB engine → shared runtime: migration plan (offseason 2026–27)

Status: **deferred** (decision 2026-10-12, option b — see `docs/integration_master_plan.md`):
the runtime is built for CBB alone; MLB moves onto it only by a later explicit ask, after CBB has
run a season on it. This document is the plan for that later move. Until then MLB's only change
is the shared credit-ledger `debit()` call and the audit-email reorder described there. `engine_common` stays pure math. Nothing here changes how MLB
*prices* a bet; every phase is proven by replaying the 2026 season and matching the old output.

## 0. What MLB is today (verified in code)

| Piece | Where | Size | Problem |
|---|---|---|---|
| Odds client | `mlb_engine/data/oddsapi.py` | 635 lines | own `credits_remaining` + `min_credits` floor; four other clients do the same against the same key |
| Quote archive | board CSV per slate, collapsed close JSON (`audit/clv.py`) | — | close keyed on `matchup\|market\|selection` text, not event id; the double-header bug NBA fixed (L8) never propagated |
| Ledger | `audit/ledger.py`, 28 fields | 635 | no event id, no pass tag, no parameter version, no refused rows, no `close_status`; `update_ledger` appends by date |
| State sync | `state.py` (`engine-state` orphan branch) | 960 | predictions pruned at 35 days because a slate is ~5 MB; eight bespoke `merge_*_files` functions |
| Pipeline | `pipeline.py` | 2,589 | capture, pricing, gates, board recording and output are one class |
| Audit lead | `audit/scorecard.py` → `output/report.py` | — | page one is PPV/NPV + Play/Fade dot; `market_isolation.py` (model-vs-close Brier, paired bootstrap) exists but is offline |
| Clock | `scripts/macos/run_predictions.command`, `run_audit.command`, plists | — | no pass id, no shared failure alert |
| Tests | 126 of 144 top-level test files import `mlb_engine` | — | the regression net for the whole migration |

MLB's graded record (3,618 buys, −9.3% ROI; model Brier 0.2198 vs close 0.2149 on 140,630
rows) is the fixture: after migration every number in the cumulative audit must reproduce
to the cent from the same ledger.

## 1. Target: the NBA ledger row is the schema

`nba_engine/audit/ledger.py::LedgerRow` (verified): `slate_date tip_utc event_id espn_id
matchup market ot_rule side entity line book american opposite_american exec_books exec_prices
books paired_books fair model_prob edge ev tier gates priced_at pass_tag versions mode outcome
void_reason pnl away_score home_score close_status close_captured_at close_american
close_opposite close_fair clv clv_ev pre_pull_american pre_pull_captured_at pre_pull_clv
graded_at`.

Generalised into `engine_runtime.ledger.Row` with: `sport`, `event_id` (Odds API) + `official_id`
(ESPN for CBB/NBA, `game_pk` for MLB), the CBB §4.a decision-snapshot groups (provenance,
decision incl. `edge_by_rung`, context, execution) as *optional* columns so MLB can leave them
blank in year one, `mode ∈ {live, shadow, paper, research}`, `grading_status`.
MLB-specific fields (`lineup_status`, `hours_to_first_pitch`, `batx_prob`, `category`,
`under_odds`) ride in a `sport_extra` JSON column — never promoted to the shared row.

## 2. Phases (each one a PR, each one replay-verified, in this order)

### Phase A — `engine_runtime` skeleton, used by nobody (1 session)
* `engine_runtime/odds_client.py`: one Odds API client; `/odds` featured + `/events/{id}/odds`
  per-event; returns raw JSON + headers. Copied from the NHL/NBA client, not MLB's (MLB's has
  prop-market pricing logic baked in — that stays in `mlb_engine`).
* `engine_runtime/credits.py`: one **persisted** daily credit ledger (`engine-state/runtime/credits.csv`,
  one row per call: sport, endpoint, markets×regions cost, remaining-header, pass id). Budget per
  sport per day declared in one YAML; a call that would breach its sport's slice **or** the key's
  floor is refused and logged. CBB's T-3h featured passes are a reserved slice.
* `engine_runtime/quotes/`: per-pass Parquet shard writer (content fingerprint skip, as
  `nhl_engine.data.capture.fingerprint` does), daily manifest with hashes committed to
  `engine-state`, DuckDB view over the shard directory, `rclone crypt` push.
* `engine_runtime/odds_math.py`: **power de-vig** and `fair_median(excluding=exec_books)`
  moved here from NBA's local code; `engine_common.odds.devig` (multiplicative) kept untouched for
  the engines that still call it, marked deprecated in its docstring.
* `engine_runtime/ledger.py`: the Row above, write-once pass files, `corrections.csv` append-only,
  hash-chained daily files.
* `engine_runtime/pass_runner.py`: `capture → price → write_once → grade`, each step stamps
  `pass_id`, `code_sha`, `env_lock_hash`, credit spend, snapshot hash; a missing step/close raises
  an alert (email) instead of leaving a blank CLV cell.
* Gate: ruff / mypy / pytest green; no engine imports it yet.

### Phase B — MLB quote tape and close onto the runtime (1 session)
* `mlb_engine` capture writes the board through `engine_runtime.quotes` (Parquet shards) **and**
  still writes the old board CSV for one phase (dual-write); close = last pre-first-pitch quote
  from the tape, keyed on `event_id`/`game_pk`, computed by the runtime.
* Backfill: convert the existing 2026 board CSVs and collapsed-close JSONs into shards so the
  140k close rows are queryable in DuckDB; **matchup-text keys are mapped to `game_pk` through
  the saved slate JSON**, double-headers resolved by first-pitch time; rows that cannot be mapped
  are kept with `official_id = NULL` and counted in the integrity block (never dropped, never
  guessed).
* Verification: `clv.py` recomputed from the tape must equal the stored CLV on every graded row
  (diff report, tolerance 0); mapped-row count published.

### Phase C — MLB ledger onto the runtime row (1 session)
* One-shot migration script `scripts/mlb/migrate_ledger.py`: 28-field CSV → runtime Row
  (`sport=mlb`, `mode=live`, `versions` = the config hash that was live on that date if
  recoverable from `engine-state` history, else `legacy`), MLB extras to `sport_extra`.
  Old file retained read-only as `ledger_legacy.csv`.
* `audit/ledger.py` metric functions re-pointed at the runtime loader; **every function in
  `overall_metrics` / `engine_metrics` / `daily_rollup` must return identical numbers on the
  migrated ledger** (golden-file test on the full 3,618-row record).
* Refused rows: from this point gate refusals are written as rows (`mode=live`,
  `outcome` graded counterfactually), as NBA does — this is additive; the audit gains the
  gate false-negative table for 2027.

### Phase D — Pipeline split and pass runner (1–2 sessions)
* `pipeline.py` split along existing seams: `capture` (`_record_board`, `_merge_quotes`) moves
  behind the runner's capture step; pricing (`run`, `_team_offense`, sims) is untouched and
  called by the price step; write-once and grade steps call the runtime ledger.
* `run_predictions.command` / `run_audit.command` become `mlb-engine pass --step …` invocations
  of the runner; launchd only fires the clock. `state.py` keeps syncing ledger/params/manifests
  only — board and prediction blobs leave the orphan branch (prune logic deleted, not tuned).
* Verification: replay five archived 2026 slates offline (per `testing-mlb-engine` skill) — the
  priced card, Excel invariants and emailed PDF must be byte-identical except for timestamps and
  the new footer (pass id, code sha, credits used).

### Phase E — Page one of the MLB audit (1 session, output only)
* Lead with: **fill-verified CLV in cents** (rows with no captured price are *excluded*, not
  paid at −110), game-clustered bootstrap interval, share of bets beating the close; then
  model-vs-close Brier/log-loss from `market_isolation.py` (now a nightly section, not a script),
  calibration bins; then PPV/NPV/ROI next to base rate and break-even. Play/Fade dot removed.
* The six standard tables (accuracy, market comparison, selection, gates, price band, ROI) in
  the NBA scorecard's column order, so Monday review is one shape across sports.
* Per-gate standing attribution: CLV of refused vs kept rows, clustered on slate date, with
  interval; `n` and "underpowered" printed where the interval covers zero.
* Kill switch / SD-scaled drawdown pause encoded for 2027 as `mode` flips (`live → research`),
  thresholds in `mlb/params/preregistered.yaml`, same rule text as CBB §4.

### Phase F — Cut-over and cleanup (½ session)
* Remove dual-write, delete `mlb_engine/data/oddsapi.py` credit logic (client stays only for
  prop-market shaping), delete `merge_closing_files` / `merge_board_files` from `state.py`.
* Blueprint/README: MLB section updated; `docs/mlb/` gets the runtime row dictionary.

## 3. What does not change
Pricing: Statcast windows, F5 Markov chain, prop sim, calibration map, tiers, gates, every
threshold in `config.py`. Outputs: worksheet, totals sheet, power screen, regression article.
`engine_common` stays math-only. CFB/NFL/NHL/NBA untouched except that they *may* opt into
`engine_runtime.credits` (Phase A) — that is the one cross-engine change and is additive.

## 4. Order and timing
A → B → C → D → E → F, ~6 sessions of work; external waits: none (all replay is offline, no
Odds API spend). Hard deadline: spring training boards (~late Feb 2027) — Phase D must be live
on the Mac before the first exhibition board is captured so 2027 starts with event-id-keyed
closes. CBB (Nov 2026) is built on Phase A directly, so Phase A comes first regardless.

## 5. Risks
* **Matchup→`game_pk` mapping** on old closes may leave a tail unmapped; it is reported, not
  imputed. If > 2% of rows fail, Phase B stops and the gap is shown before Phase C.
* **Golden-file drift**: any metric that does not reproduce blocks the phase; no "close enough".
* **Mac disk/off-site**: Parquet shards need `rclone` configured on the Mac — one paste block,
  done in Phase B before dual-write starts.
* **Shared credit ledger** changes behaviour of the other four clients only if they opt in;
  default is observe-only (log, never refuse) for one month.
