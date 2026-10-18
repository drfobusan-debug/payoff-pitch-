# MLB recode — build plan for Phases 1–3

This is the build sheet for `docs/mlb/recode_master_plan.md`. The master plan
says what and why. This file says which module changes, what each new file
holds, which tests gate each phase, and what "done" means. Nothing here opens a
buy, changes a tier, a stake, a card cell, or `engine_common`.

Labels follow the master plan: **[MEASURED]** · **[DEFAULT]** (a starting
value the review can move) · **[CHECK]** (verify in code or with the vendor
before writing the line that depends on it).

---

## 0. Ground rules for every phase

- **Card neutrality is the first test.** Replay one archived slate before and
  after the change:
  - `predictions_<date>.json` and the card workbook's cell values hash
    identical (sha256);
  - ledger rows compare field by field, **excluding the columns this build
    adds**. A byte-identical ledger file would fail on the first replay just
    because a header gained a column, which is not a behaviour change. Use the `testing-mlb-engine` replay (archived board, isolated
  data dir, no Odds API spend, no state push, no email).
- **No new Odds API request.** Every new file is written from quotes or sim
  arrays already in memory. A test fails if the shadow command constructs an
  `OddsAPIClient`.
- **Best-effort writes.** Each new hook follows `_record_board`: an `OSError`
  logs a warning and the run carries on. A tape or histogram failure never
  stops a card.
- **Local gate:** `ruff`, `mypy`, `pytest` green before each phase is shown.
  There is no CI.
- Each phase is its own branch and PR, and is opened only when asked.

---

## Phase 1 — Quote tape and close quarantine

Ships alone. Its value does not depend on the tilt.

### 1.1 What exists today [MEASURED, `origin/main`]

- `pipeline.run` merges VSIN and Odds API quotes into
  `Quotes = dict[(matchup, market, selection), list[MarketQuote]]`. It then
  calls `_record_board`, which collapses each key to one consensus
  `ClosingQuote` with `audit/clv.py` `closing_quotes` (the same `evaluate()`
  consensus the live screen uses). Across the day's runs, `merge_board` keeps
  the **first** consensus seen per key, which is the opening board.
- `cmd_close` fetches with `pregame_only=True` and merges with
  `merge_closing` (last price wins).
- `MarketQuote` already carries `book`, `american` and `opposite_american`. The
  per-book two-sided pair is therefore in memory, and is thrown away when it is
  collapsed to consensus.
- `quote_key` is `matchup|market|selection`. The line is inside the selection
  string (`Over 8.5`, `LAD -1.5`, `Shohei Ohtani BB o0.5`). `event_id` lives on
  `oddsapi._Event` (`_SlateIndex` holds only abbreviations and start times)
  and never reaches the quotes.
- `_parse_game` and `_parse_props` write each event's quotes straight into
  `Quotes` under `ev.matchup`. Two events with the same matchup string land in
  one list, and the event is gone by the time `fetch` returns.
- `_game_selection` drops any outcome that is neither team (a first-five draw).
  `_opposite_prices` pairs only exact two-way sides and leaves a three-outcome
  market unpaired. So the stored F5 fair is a two-way devig where the book
  priced it two-way, and a raw price where it didn't.
- `merge_dated_csv` keeps whichever field list is **longer**
  (`fields = local_fields if len(local_fields) >= len(fields) else fields`).
  It does not union headers.
- The closing JSON carries no `hours_to_first_pitch`. An in-play prop at an
  ordinary price passes `is_plausible_close`, which is the case that
  function's comment already describes.
- `predictions_<date>.json` carries no run id. `cmd_run` calls
  `write_workbook` and then `save_json`.
- `is_plausible_close` (added 2026-08-14, after the 08-08 ATH row) refuses a
  team-market close at −1000 or shorter, or at +1000 or longer. The ATH
  +242/−2000 join would now be refused. Trea Turner H o1.5 +183→−185 (08-08)
  would not.

### 1.2 New: `mlb_engine/audit/tape.py`

```python
TAPE_VERSION = 1

@dataclass(frozen=True)
class TapeRow:
    tape_version: int
    captured_at: str          # UTC ISO, seconds
    source: str               # "run" | "close"
    slate_date: str
    event_id: str | None      # None for VSIN-only keys
    game_pk: int | None       # stats-api game, mapped in the pipeline
    commence_utc: str | None  # the event's start, from the vendor
    matchup: str
    market: str               # engine market, e.g. "pitcher_k"
    selection: str            # exactly as keyed in Quotes
    canonical: str            # keys.canonical(selection)
    line: float | None        # parsed from selection; None for ML
    side: str                 # "over" | "under" | team abbrev
    book: str
    american: float
    opposite_american: float | None
    hours_to_first_pitch: float | None
```

- File: `audit/tape_<date>.jsonl.gz`. It is append-only, one row per book per
  selection per capture.
- **Dedup key:** `(source, captured_at, game_key, market, selection, book)`,
  where `game_key` is defined in §1.3. Re-running a capture in the same second
  writes nothing new. A run and a close in the same second stay separate rows.
- `append_tape(audit_dir, slate_date, captures, source, now, game_pks)`
  returns the number of rows written. `captures` are the client's
  `QuoteCapture` records (§1.3). `append_tape` builds the frozen `TapeRow` at
  write time. It stamps `source`, `captured_at = now`,
  `hours_to_first_pitch = (commence − captured_at)` in hours, and `game_pk`
  from the pipeline's match.
- Hours are **stored, never recomputed**. Recomputing them during the
  overnight audit would make every row look in play.
- `load_tape(path) -> list[TapeRow]` skips unknown `tape_version` values with a
  warning rather than raising.
- `parse_line(market, selection)` lives here and is unit-tested against every
  selection format in `market/keys.py`.
- Size estimate (not measured): ~400 selections × ~5 books × 4–6 captures a day
  is about 10k rows/day, roughly 150 KB gzipped.
- **Size check.** `cmd_audit` logs the tape's gzipped bytes per day. If a week
  averages above 1 MB/day [DEFAULT], a later change splits the tape into a
  static selection registry plus integer-keyed ticks. Until then it stays one
  file, so the two-Mac merge stays a plain set union.

### 1.3 Event ids and doubleheaders — a Phase 1 prerequisite

The hook after `_merge_quotes` cannot do this: by then two same-matchup events
are already one list in `Quotes`. The rows have to be copied while the event
is still in hand.

- **Side list in the client.** The client can't build a `TapeRow`: it does
  not know `source`, `game_pk` or the capture time. So `_parse_game(raw, ev,
  out, *, f5, tape)` and `_parse_props(raw, ev, out, tape)` take a
  `tape: list[QuoteCapture]`. Next to each
  `out.setdefault(...).append(MarketQuote(...))` they append a capture record:
  `event_id`, `commence`, `matchup`, `market`, `selection`, `book`, `american`
  and `opposite_american`. `fetch` exposes the list as `client.last_captures`.
  The returned `Quotes` dict is not touched, so card neutrality holds.
- **VSIN rows** come from the merged `Quotes` with `event_id=None`, written
  only for keys the Odds API side did not supply.
- **`game_pk`.** The pipeline fills it by matching each event to a slate
  `Game` on `(away, home)` and the nearest `game_datetime_utc` within
  `_SAME_GAME_WINDOW` (8 h). Two events for one matchup inside that window
  are a doubleheader, and nearest commence separates them (13:05 vs 19:10).
  `game_pk` is blank **only** when two games are equally near.
- **`game_key`** = `event_id`; for a VSIN row,
  `f"{matchup}|{commence_utc or scheduled start:%Y-%m-%dT%H:%MZ}"`.
- **Known limit, not fixed in Phase 1.** The live `Quotes` dict still blends
  a doubleheader's two games under one key, so the live card can price game 2
  off game 1's quotes. The tape keeps them apart, but the card does not. A
  test pins this down so the limit stays visible (§1.8).

**CLV join rule**, for ledger rows (which have no `event_id`):

| Events for this matchup on the date | Join |
|---|---|
| One | Joins on matchup text, as today. |
| Two or more, `event_id` on both sides | Joins on `event_id`. |
| Two or more, no id on one side | `event_mismatch`. Not guessed. |

### 1.4 Hooks

- `pipeline.run`: right after `quotes = _merge_quotes(odds, quotes)` and before
  `_record_board`, call
  `append_tape(..., captures=client.last_captures + vsin_captures,
  source="run")` inside `try/except OSError`.
- `cli.cmd_close`: after `client.fetch(...)`, call
  `append_tape(..., captures=client.last_captures, source="close",
  game_pks=match_game_pks(client.last_captures, slate.games))`. `cmd_close`
  already loads the slate, so it runs the same nearest-commence match the
  pipeline runs. Unmatched, or two games equally near, stays null. CLV does
  not need `game_pk` (the close joins on `game_key`); the Phase 2 worksheet
  join does.
- Neither hook reads the tape back. `_record_board`, `closing_quotes` and the
  drift gate are untouched.

### 1.5 Quarantine: replaces the "other side moved with it" clause

The master plan's clause never fires on no-vig numbers, because the two sides
always sum to 1. The rule as built:

A close is **quarantined** (left out of every CLV line, counted in the report)
when any of these hold:

| Reason code | Rule |
|---|---|
| `implausible` | Existing `is_plausible_close` fails. |
| `no_pregame` | Taped date, and every taped row for the key is in play (`hours_to_first_pitch ≤ 0`). |
| `no_pair` | The tape has no book at the close with both sides of this selection at the same `captured_at`. Applies only once the tape exists for that date. |
| `event_mismatch` | The bet's `event_id` and the close's `event_id` both exist and differ, or the matchup is ambiguous (§1.3). |
| `overround` | **Log only until frozen.** The band starts unset: the first week of tape prints the overround distribution by market (a −110 game line and a +2600 HR book are not one number). Once frozen, a pair outside the band is **dropped from that capture's consensus**, and the selection is quarantined only when no pair remains. One junk book can't stay inside an otherwise good close. |
| `move_gt_15pp` | The absolute value of the close no-vig minus the bet `fair_prob` is above 0.15 [DEFAULT]. |

- **Order:** the first matching code wins, in the order above.
- A quarantined close is never turned back into a usable close. The review can
  move the bands; it cannot remove a code.
- **Which close.** For a date with a tape, the close is a **consensus over
  every book at one capture**, not the single row appended last:
  1. Take the latest `source="close"` capture (`captured_at`) whose rows for
     the key have `hours_to_first_pitch > 0`.
  2. If there is none, take the latest pregame `source="run"` capture.
  3. Devig that capture's rows with the card's existing `evaluate()`
     consensus (same book weights, two-sided rows only).
  4. If every taped row for the key is in play, the selection is quarantined
     `no_pregame`.

  Dates without a tape keep the closing JSON. The two close definitions are
  different prices, so for the first two weeks of tape the audit prints
  taped-date CLV under both side by side. They are not pooled into one series
  without that comparison.
- `audit/ledger.py`: new `LedgerEntry.clv_quarantine: str | None`, added to
  both field lists. Its values are a reason code, `clean` (judged and kept),
  or blank (**never judged**). Blank is not clean: every summer row is blank.
- **`state.merge_dated_csv` changes to union headers:** local fields in order,
  then any remote-only fields. This is needed because today the longer list
  wins, and two ledgers of equal width with different new columns lose one.
  The change is strictly more preserving for every file in `_MERGED_CSVS`.
- `attach_clv` sets `clv_quarantine` on every row it judges and leaves
  `close_odds`, `close_prob`, `clv` and `clv_ev` blank on quarantined rows.
- `cmd_audit` prints
  `CLV: n priced, j judged, k quarantined (by reason), u never judged`.
  The live mean still includes never-judged rows (the August 8 Turner join
  stays in it), so the line says so. The replay script (§1.7) prints the
  judged number beside it.

### 1.6 State sync

- `state.push_state` and `pull_state`: add `audit/tape_*.jsonl.gz` ↔
  `mlb/tape/`. Two machines' tapes merge as a set union on the dedup key, so
  the merge is idempotent and order-free.
- `merge_tape_files(remote, local)` follows the pattern of the other
  `merge_*_files`.

### 1.7 History: report only

- `scripts/clv_quarantine_replay.py` reads the `engine-state` ledger and the
  closing JSONs, applies the rules that need no tape (`implausible`,
  `move_gt_15pp`, and `event_mismatch` where the date had two same-matchup
  events), and prints counts plus the before/after mean CLV and ROI.
- It does not rewrite the ledger.
- Expected from the summer book [MEASURED]: 2 of 1,925 CLV rows, mean `clv_ev`
  −3.56¢ → −3.71¢.

### 1.8 Tests

- `tests/test_tape.py`:
  - round-trip;
  - dedup on re-append; a run and a close in the same second are both kept;
  - `hours_to_first_pitch` is stamped at write time from `commence −
    captured_at`, and `load_tape` returns the stored value (it is not
    recomputed from a later clock);
  - `parse_line` for every market format, including `+1.5`/`-1.5` sign
    survival and integer lines;
  - unknown version skipped;
  - a VSIN-only key carries `event_id=None`;
  - `OSError` swallowed by the hook.
- `tests/test_oddsapi.py` (doubleheader fixture: same matchup, starts 13:05Z
  and 19:10Z, two event_ids):
  - `last_captures` holds each game's rows under its own `event_id`;
  - the pipeline assigns each game its own `game_pk` by nearest commence;
    two equally near games leave it blank;
  - the returned `Quotes` equals today's output, byte for byte;
  - **known limit:** that `Quotes` key holds both games' prices, asserted so
    a later fix has to change the test on purpose.
- `tests/test_clv.py`:
  - one test per reason code;
  - code precedence;
  - a 14.9pp move kept and a 15.1pp move quarantined;
  - the ATH +242/−2000 fixture quarantined as `implausible`;
  - the Turner +183/−185 fixture quarantined as `move_gt_15pp`;
  - quarantined rows leave the CLV columns blank;
  - `clv_rows` and `summarize` exclude quarantined rows;
  - judged-and-kept rows read `clean`, unjudged rows stay blank;
  - the join table in §1.3 (one event, two with ids, two without);
  - close selection:
    - three books at the last pregame close capture give the `evaluate()`
      consensus devig, not the last-appended book's price;
    - an in-play capture (`hours_to_first_pitch < 0`) at an ordinary price is
      skipped for the last pregame one;
    - with no pregame close capture, the last pregame run capture is used;
    - all rows in play gives `no_pregame`;
  - `overround` logs and does not quarantine while the band is unset;
  - with a frozen band, one out-of-band book is dropped from the capture's
    consensus and the close is kept; with every pair out of band, the
    selection is quarantined.
- `tests/test_state.py`:
  - tape union merge;
  - ledger merge where each side has a **different** extra column at the
    **same** width: both columns survive (today's longer-list rule fails
    this);
  - an older ledger without `clv_quarantine` merges without dropping it.
- **Neutrality:** the replayed slate's predictions and card are unchanged.

**Done when:** tests are green, the replay is neutral, one live slate writes a
tape on its run and close, and the audit prints the judged/quarantine line.

---

## Phase 2 — Joint histograms, the tilt, `mlb-engine shadow`

### 2.1 Histograms: `mlb_engine/models/histograms.py`

These are built from the `GameSimResult` arrays `_price_game` already holds,
right after `res = mc.simulate(...)`, and **held in memory** on the pipeline
(`pipeline.last_histograms`). Nothing is written there. `montecarlo.py` does
not change.

| Histogram | Source arrays | Key |
|---|---|---|
| Full game | `home_runs_full × away_runs_full` | `(h, a)` |
| First five | `home_runs_f5 × away_runs_f5` | `(h, a)` |
| Each priced prop | `bat[team][stat][:, slot]` or `pit[team][stat]` × full-game total | `(k, t)` |

- Only props with a quote on the board get a histogram.
- Stored as sparse integer counts with `n_sims`, `seed`, `mc_sims` and the
  engine git SHA.
- File: `audit/hist_<date>.json.gz`, keyed by `game_key` (§1.3: the Odds API
  id, else `matchup|commence`), then by histogram name. Bare matchup is never
  a key, so a doubleheader can't overwrite itself.
- **Decided: the card run.** `cmd_run` writes the file after
  `write_workbook` and `save_json` both succeed, and **before** the
  `_state_push` that follows `save_json`, so the push carries it. A run that
  fails before the card writes nothing.
- **Identity is per game.** There is no run id in the predictions file, and a
  late pass (`--within-hours`) merges the file but reprices only the games
  near first pitch. So each game's histogram stores `pred_hash`: the sha256 of
  that game's prediction rows (by `game_pk`, canonical JSON, sorted), as
  written. A run replaces histograms **only for the games it priced**.
- The shadow recomputes `pred_hash` from `predictions_<date>.json` and skips
  any game whose hash doesn't match (`pin_status=hist_stale`).
- These are the **raw** sim counts: before the calibrator, shrink, outs bias,
  HRR adjust and `run_env.apply_shift`. That is the "unshifted sim". The live
  `model_prob` is stored beside it for comparison, not recomputed.
- **Reproduction test:** the raw over probability read from a stored histogram
  equals the raw probability `_price_game` computed from the arrays, to 1e-12,
  on a fixture game. It uses **the same push exclusion `_price_game` uses** on
  an integer line. A reader that counts a push as a loss misses 1e-12 on a
  correct file.

### 2.2 The tilt: `mlb_engine/market/tilt.py`

**Devig target.** Take the tape rows from the capture the card used. For each
key, devig with the existing `evaluate()` consensus (same book weights as the
live screen). The **main line** is the line quoted two-sided by the most books;
ties go to the line closer to 50/50. One-sided means `unpinned`.

**Integer lines.** When the line can push, the target and the tilted read are
both `P(over) / (P(over) + P(under))`, with the push mass excluded.

**1D: props.**

```
tilt_1d(pmf, line, target) -> TiltResult(lam, pmf, ess, residual, status)
p_λ(k) ∝ p(k) · exp(λk)
```

- `P_λ(over)` is monotone in λ, so solve with bracketing plus `brentq` on λ in
  [−5, 5] [DEFAULT].
- If the target lies outside what the support can reach, the result is
  `unpinned`.

**2D: game and first five.**

```
tilt_2d(joint, total_line, target_over, target_home, three_way=False)
p(h, a) ∝ p_sim(h, a) · exp(α(h + a) + β(h − a))      # α=(λh+λa)/2, β=(λh−λa)/2
```

- **Primary solver:** damped Newton on (α, β), starting from (0, 0). At most
  50 iterations.
- **Jacobian for tail targets.** Let `s = h + a` and `m = h − a`. Under the
  tilted distribution, `∂P(E)/∂α = Cov(1{E}, s)` and `∂P(E)/∂β = Cov(1{E}, m)`,
  for `E = {s > line}` and the moneyline event. The two-way moneyline read is
  a ratio `W/(W+L)`, so its row is the quotient rule on `Cov(1{h>a}, ·)` and
  `Cov(1{a>h}, ·)`. `Cov(s, m)`-type feature covariances are the Jacobian for
  matching **means**. Pointed at tail residuals, they walk to a different
  `(α, β)`, and that changes the run-line number.
- **Fallback:** `scipy.optimize.least_squares` (trust-region reflective,
  bounds α, β ∈ [−5, 5] [DEFAULT]) on the same two residuals. It runs when
  Newton fails to converge, or when the Jacobian's condition number exceeds
  1e8 [DEFAULT]. `scipy>=1.10` is already a dependency.
- Overround never reaches the solver; its targets are devigged probabilities
  in (0, 1). The real failure is a target pair the support cannot reach.
- **Root selection: filter the misses first, then take the smallest step.**
  Targets are tail probabilities, not means, so more than one `(α, β)` can hit
  both.
  1. Newton and `least_squares` both run from (0, 0).
  2. If neither lands, `least_squares` also runs from the eight starts
     `{−1, 0, 1}² \ (0, 0)`.
  3. A candidate **lands** when both residuals are ≤ 1e-4 [DEFAULT].
  4. Among the landed candidates, the one used has the smallest `‖(α, β)‖`.
     The result is `unpinned` only if nothing lands.

  This keeps the hard-to-pin games in the run-line sample instead of quietly
  dropping them when the nearest root missed. The Newton and (0, 0)
  `least_squares` candidates are stored (`alpha_newton, beta_newton,
  alpha_lsq, beta_lsq`) with `n_landed`. The `solver` column names the one
  used. The numbered steps are the rule: both solvers run, a miss never beats
  a land, and the smaller landed norm is kept.
- Store both residuals for the chosen root.
- Full-game home win is the two-way formula `P(h>a)/(P(h>a)+P(a>h))`. The sim
  plays extra innings, so tie mass is near zero.
- **First five.** Pin on the same basis the stored fair used: the two-way
  devig, with the read `P(h>a)/(P(h>a)+P(a>h))`. A three-way read
  (`P(h>a)`, `P(tie)`, `P(a>h)`) sits beside it as a column only.
  **[CHECK]** how each book settles a first-five tie before the three-way
  column is tested or scored.

**Prop conditional on the game.**

1. From the tilted game joint, take the total marginal and form
   `w(t) = p_tilt(t) / p_sim(t)`.
   - A bin with zero sim mass gets `w(t) = 0`.
   - A bin with a single sim path is left out of the ratio (`w(t) = 0`)
     before `λ` is solved, or that one path sets the prop.
2. Reweight the prop's `(k, t)` histogram by `w(t)`.
3. Solve 1D λ on `k`.
4. Tilting `k` moves the total, because Ks and runs are correlated. Record
   `total_drift = E_final[t] − E_game[t]`.
5. Optionally alternate steps 2–3 up to 5 times (iterative proportional
   fitting) until the drift is below 0.01 runs. **Off by default for the pin;
   both versions are stored and scored:** `p_tilt_1pass`,
   `total_drift_1pass`, `p_tilt_ipf`, `total_drift_ipf`.

**Effective sample size.** Kish ESS on path weights:
`ESS = (Σ nᵢwᵢ)² / Σ nᵢwᵢ²`.

- The thin threshold is `0.1 · n_sims`, read from the histogram's `n_sims`,
  not a hard-coded 2,000. The tail-path count (raw paths beyond the read
  line) is stored too.
- **Measure, then freeze.** The first PR stores `ess` and `tail_paths` and
  prints their distribution. The `thin` flag is set only once the thresholds
  are frozen; until then it is blank.
- Thin reads are kept and labeled, not dropped.

### 2.3 Reads off the tilted distributions

- **Cross-book: its own rows, different lines only.** At the pin line itself,
  the tilted probability equals the consensus, so "tilt minus one book" is
  just consensus minus that book, not a test of the tilt. Moneylines have no
  second number at all. So the cross-book test runs only where a book posts a
  **different** line two-sided:
  - file `audit/crossbook_ledger.csv`, one row per
    `(date, game_key, market, player, pin_line, read_book, read_line)`;
  - columns: `p_tilt`, `book_devig`, `gap`, `ess`, `tail_paths`, `result`;
  - **`result` is the counting stat graded against `read_line`** (Ks against
    6.5 on a 6.5 row), never the card's line. A push at an integer read line
    is excluded, the same as in the pin. Graded against the wrong line, the
    Brier would be testing a different event.
  - same-line, cross-book differences are not written.
- **Run line:** the game-level tilt test. `P(margin ≥ 2)` and `P(margin ≤ −2)` off the tilted joint,
  against the posted run line's devig, stored as `rl_residual`
  (model minus market, on the home side's posted run line).
- **Holdout rotations.** The same `tilt_2d` solver is run on the other two
  pairs, from (0, 0), with the same root rule. Neither one replaces the price
  pin, and neither is a ticket.
  - **Moneyline holdout:** pin the total and the run line, then read the
    moneyline. `ml_holdout_gap` = tilted `P(home win)` minus the devigged
    moneyline (model minus market). With the total pinned, it has the
    **opposite sign** of `rl_residual` on every correct solve. Its size is
    `rl_residual` rescaled, so it is stored but never printed as a finding.
    A violation means the solver failed (the residuals show it) or the
    −1.5/+1.5 sides were swapped. The histogram cannot cause one.
  - **Total holdout:** pin the moneyline and the run line, then read the
    total. `total_holdout_gap` = tilted `P(over)` minus the devigged total
    (model minus market). `total_holdout_status` is `pinned` or `unpinned`
    with the same residual rule. Both knobs sit on the margin, so expect a
    real `unpinned` share. That share is counted.
  - Only a two-sided run line on the tape makes a holdout. Without one, both
    columns are blank and the status is `no_rl`.
- **Named hypotheses:** `anchor_to_market(raw, fair, w)`, with `w` the old
  anchor weight (not `α`, which is the sum tilt), for `game_total` 0.78,
  `pitcher_bb` 0.50, `pitcher_h` 0.81 and `batter_2b` 0.61. Each is a column
  and none is a price.

### 2.4 `mlb-engine shadow --date YYYY-MM-DD`

- Reads `tape_<date>`, `hist_<date>`, `predictions_<date>` (for live
  `model_prob`), `worksheet_ledger.csv` and `totals_ledger.csv`.
- Writes `audit/shadow_ledger.csv`.
- Does not call `classify`, the card builder, email, or the Odds API.
- **Columns:** `date, game_key, event_id, game_pk, matchup, market, selection, line,
  side, main_line, american, book, fair_prob, raw_sim_prob, live_model_prob,
  pin_status, solver, lam, alpha, beta, alpha_newton, beta_newton, alpha_lsq,
  beta_lsq, n_landed, residual_1, residual_2, ess, tail_paths, thin, p_tilt_1pass,
  total_drift_1pass, p_tilt_ipf, total_drift_ipf, rl_residual,
  ml_holdout_gap, total_holdout_gap, total_holdout_status, f5_3way_home,
  f5_3way_tie, hyp_total_078, hyp_pbb_050, hyp_ph_081, hyp_2b_061,
  worksheet_gap, worksheet_side_fair, totals_sum, sim_median_total,
  posted_total, result, close_prob, clv_quarantine`.
  - `american` and `book` are the best main-line price on the tape at the card
    capture. Without them the hypothesis `clv_ev` is undefined.
  - `game_pk` is the join to `worksheet_ledger.csv` and `totals_ledger.csv`,
    which are keyed by the stats-api game. The tape is keyed by the odds-api
    event.
- `state._MERGED_CSVS` gets `shadow_ledger.csv`, keyed by
  `(date, game_key, market, selection, line)`, with `game_key` from §1.3.
  `event_id` and `game_pk` are columns, not keys: a null `event_id` would fold
  every `Over 8.5` on a VSIN-only slate into one row, and an empty `game_pk`
  would collide two unmatched games. `crossbook_ledger.csv` is
  keyed as in §2.3.
- Scheduling: a step after the morning run's card is sent, on the Mac. It
  runs last and never blocks the card.

### 2.5 Tests

- `tests/test_tilt.py`:
  - **1D:**
    - λ=0 returns the input;
    - Poisson(6.9) tilted to `P(K>7.5)` = 0.45 hits the target to 1e-9;
    - monotone in the target;
    - all mass at 0 or at the cap returns `unpinned`;
    - integer-line push exclusion;
    - ESS equals N at λ=0 and falls as |λ| grows.
  - **2D:**
    - the sim joint's own probabilities as targets give α=β=0;
    - a known bivariate fixture recovers known (α, β);
    - the analytic tail Jacobian matches a finite-difference Jacobian to
      1e-6; a means-Jacobian solver on the same tail targets lands on a
      different `(α, β)` (guards against the swap);
    - an infeasible target pair returns `unpinned` with residuals;
    - a near-singular fixture (all mass on two adjacent totals): Newton fails
      and `least_squares` is used and recorded;
    - root selection:
      - with two landed roots, the smaller-norm one is used;
      - a fixture where the (0, 0) root misses but a multi-start root lands
        is pinned (not `unpinned`), with `n_landed` ≥ 1;
      - a landed root is never passed over for a closer root that missed;
    - first five: the two-way pin matches the stored fair's basis. The
      three-way test waits on the settlement **[CHECK]**.
  - **Holdouts:**
    - on fixture joints with targets the support can reach, `ml_holdout_gap`
      and `rl_residual` have opposite signs, model minus market, on every
      case (a seeded sweep of at least 200 targets);
    - the same sweep with the −1.5 and +1.5 sides swapped fails the
      opposite-sign assertion;
    - a target pair the support can't reach gives
      `total_holdout_status=unpinned` with residuals; no run line gives
      `no_rl` and blank gaps.
  - **Conditional:**
    - with `w(t)=1`, the result equals the 1D tilt;
    - a zero-mass total bin gets weight 0, and a one-path bin is excluded
      before `λ` is solved;
    - both `total_drift_1pass` and `total_drift_ipf` are written; the IPF
      one is below 0.01 on a positive fixture.
  - **ESS:** the threshold scales with `n_sims` (a 10,000-path fixture flags
    at 1,000); `thin` stays blank while thresholds are unfrozen.
- `tests/test_histograms.py`:
  - sparse round-trip; only quoted props stored;
  - the reproduction test in §2.1, including an integer-line push fixture;
  - nothing written when the card write fails, and the file exists before
    `_state_push` is called;
  - a late pass replaces only the games it priced, and each game's
    `pred_hash` equals the hash of its rows;
  - the shadow skips a game with a changed `pred_hash` as `hist_stale`.
- `tests/test_crossbook.py`:
  - a same-line book writes no row, a different-line book writes one, and a
    moneyline writes none;
  - a player whose box-score name carries an accent the selection lacks
    still grades, through `canonical_index`;
  - grading: 6 Ks grade a 6.5 read row `under` and a 5.5 pin row `over` on
    the same game;
  - a 6.0 read line with exactly 6 Ks is a push and is excluded from the
    Brier.
- `tests/test_shadow_cli.py`:
  - runs on a fixture tape plus histograms;
  - no `OddsAPIClient` constructed;
  - no file written outside `shadow_ledger.csv` and `crossbook_ledger.csv`;
  - predictions and card files unchanged by mtime and hash.
- **Neutrality:** the replayed slate is unchanged.

**Done when:** tests are green, the replay is neutral, and one live slate
produces a `shadow_ledger.csv` with every quoted main line either pinned or
labeled `unpinned` with a reason.

---

## Phase 3 — Grade the shadow, and the morning block

### 3.1 Grading

- `cmd_audit`, after the live ledger is graded, grades the same date's
  `shadow_ledger.csv` and `crossbook_ledger.csv` rows with `audit/grade.py`.
- **Names resolve through the same `keys.canonical_index` the live close join
  uses**, to the prediction row's `player_id`. The stat is then read with
  `grade_batter` / `grade_pitcher` against the row's own line. Matching on the
  raw selection string would drop a prop whose accent differs from the box
  score. Both graders share this resolver.
- Closes and quarantine are filled from the tape.
- Every row is graded. None is a ticket. There is no P&L column: at the main
  line the shadow probability is the market's by construction.
- The named-hypothesis columns are scored on `clv_ev` at the row's `american`
  against the close.

### 3.2 Date-clustered intervals

- There is no cluster helper in `mlb_engine/audit` today [MEASURED, grep].
  Add `audit/cluster.py` with `clustered_mean_ci(values, dates, b=2000,
  seed=7)`: a bootstrap that resamples whole dates with replacement
  (`numpy.random.default_rng(seed)`), takes the row-weighted mean of each
  resample, and returns the 2.5 and 97.5 percentiles of the `b` means with
  `numpy.percentile`'s default linear interpolation.
- Tests:
  - one row per date matches an ordinary bootstrap;
  - duplicating every row within its date leaves the width unchanged;
  - fixed seed means fixed output.

### 3.3 Morning block: report and email, in this order

1. **Live CLV:** `clv_ev` with its clustered 95% interval, n, rows judged,
   rows never judged, and the quarantine count by reason. Plus the overround
   distribution by market while that band is unfrozen.
   - **Quarantine impact:** live-ledger ROI and mean `clv_ev`, each with and
     without the quarantined rows, side by side. This measures what the
     quarantine costs or protects. The shadow ledger has no P&L column: at the
     main line its probability is the market's by construction, so a P&L there
     is noise, not edge.
2. **Tilt tests:**
   - cross-book rows (different lines only): mean `gap` with its interval and
     n, by market, and the Brier of `p_tilt` vs `book_devig` against results;
   - run-line residual `P(margin ≥ 2)` after the moneyline and the total are
     pinned, against the posted run line's devig and against results;
   - **moneyline-holdout sign violations: a count only, expected 0.** The
     moneyline gap itself is not printed;
   - ESS and tail-path distributions; thin rows separately once the flag is
     frozen.
3. **Pin health:**
   - mean λ (or α, β) by market;
   - unpinned share and its reasons;
   - share of sim medians above the posted total;
   - mean `total_drift_1pass` and `total_drift_ipf`, and the Brier of each
     version's prop read.

The total holdout (`total_holdout_gap` mean with its interval, and the
`unpinned` share) goes in the stored audit, not on this screen.

Then worksheet gap vs the implied probability, totals
SUM vs the line, unshifted-sim Brier vs market Brier, and the named-hypothesis
columns scored on `clv_ev` against the pin.

- A fixed last line: `Gate shut: no alternate keys on the tape. 200-row rule
  is a 2027 rule.`
- The existing PPV report is not touched.

### 3.4 Tests

- An audit fixture with shadow rows checks grading, the interval call, the
  block order, the gate line, and an unchanged PPV report.
- **Neutrality**, not identity, for the live ledger. This audit can fill
  `clv_quarantine`, and a row it quarantines has its close columns blanked
  (`close_odds`, `close_prob`, `clv`, `clv_ev`). So the test asserts:
  - every field that existed before this build is unchanged on every row the
    audit did not judge;
  - on judged rows, only `clv_quarantine` and (if quarantined) those four
    close columns differ.

**Done when:** one real morning audit prints the block from a real shadow
ledger.

---

## 4. Order and size

| Step | Depends on | Rough size |
|---|---|---|
| Phase 1 (tape, event ids, quarantine, sync, replay script) | — | 1 session |
| Phase 2a (histograms and reproduction test) | — | ½ session |
| Phase 2b (tilt module and tests) | — | 1 session |
| Phase 2c (shadow CLI, schema, sync) | 1, 2a, 2b | ½–1 session |
| Phase 3 (grading, cluster CI, morning block) | 2c | 1 session |

2a and 2b can be built in parallel with Phase 1. Nothing in 2c or Phase 3 runs
on live data until Phase 1 has written a tape.

**Calendar.** The tape needs live slates. If Phase 1 lands during the
postseason, it collects a handful of games. Most of the cross-book sample
comes in April 2027.

## 5. Decisions for Franz before code

1. ~~Histograms: last run or card run~~. Decided: the card run (§2.1).
2. **Quarantine bands:** the 15pp move is a number to set now. The overround
   band is **measure, then freeze**: log-only until the first week of tape is
   printed (§1.5).
3. **ESS flag:** **measure, then freeze** at `0.1 · n_sims` and a tail-path
   floor, after the first week's distribution (§2.2).
4. **IPF on prop × total:** off by default, both scored (§2.2).
5. **History:** whether the summer ledger gets `clv_quarantine` written back,
   or stays report-only (§1.7). The default is report-only.

## 6. Not in this build

The master plan §10 list stands. In addition:

- No `*_alternate` keys.
- No change to `montecarlo.py`, the calibrator, the anchors, `min_edge`, tiers
  or `engine_common`.
- No Phase 4 code.
- No fix for the live card blending a doubleheader's quotes (§1.3). That is a
  separate, live-path change.
