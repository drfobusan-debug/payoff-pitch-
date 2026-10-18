# Integration master plan — shared runtime for CBB, one credit ledger, CLV-led audit emails

Decision (2026-10-12, option **b**): build `engine_runtime/` for CBB alone; the five live engines
keep their pipelines, ledgers and cards; the **only** cross-sport code change is a shared daily
Odds API credit ledger; the five audit emails are **reordered** to lead with the CLV they already
compute. MLB (or any other engine) moves onto the runtime later, one engine per explicit ask,
after CBB has run a season on it. `engine_common` stays math-only. No PR until asked.

## 1. Why (b)
* (a) CBB-only counter cannot see MLB's per-event prop loop spending the key between CBB passes —
  the T-3h featured-pass reserve is unenforceable.
* (c) rewrites live scoring and every morning report to prove an abstraction.
* Verified in code: the five `*_engine/data/oddsapi.py` clients each read `x-requests-remaining`
  into a per-process `credits_remaining` (MLB also keeps `min_credits=200`); nothing sums them.
  `nba_engine/data/capture.py` imports `nhl_engine.data.capture`; `engine_common.odds.devig` is
  multiplicative only; MLB closes key on `matchup|market|selection` text; `state.py` prunes
  predictions at 35 days.

## 2. `engine_runtime/` — CBB's substrate (new package, I/O allowed, no sport thresholds)

| Module | Does | Source to copy from |
|---|---|---|
| `odds_client.py` | one Odds API client: `/sports/{key}/odds` featured + `/events/{id}/odds` per-event; raw JSON + headers; every call goes through `credits.debit` | `nhl_engine/data/oddsapi.py::_get_json` (not MLB's — prop shaping is baked into that one) |
| `credits.py` | the shared ledger (§3) | new |
| `quotes/` | per-pass immutable Parquet shard writer with content-fingerprint skip, daily manifest (`sha256` per shard) committed to `engine-state/runtime/manifests/`, DuckDB view over the shard dir, `rclone crypt` push, OpenTimestamps weekly anchor of the chain head | fingerprint/skip logic from `nhl_engine/data/capture.py` |
| `odds_math.py` | **power de-vig** (`p_i = q_i^k`, k solved so Σp = 1) and `fair_median(quotes, exclude=exec_books)` — the CBB consensus. The review says NBA has a local power de-vig; grep did not find one, so this is written fresh and tested against the NFL hold study's favourite–longshot slope | new |
| `ledger.py` | Row generalised from `nba_engine.audit.ledger.LedgerRow` (`event_id`, `espn_id`, both sides, exec books/prices, `fair`, `model_prob`, `edge`, `ev`, `tier`, `gates`, `pass_tag`, `versions`, `mode`, close block, CLV block, pre-pull block) **plus** the CBB §4.a snapshot groups (`code_sha`, `env_lock_hash`, input hashes, `edge_by_rung`/`p_by_rung`, T-90/T-30 availability, `placed` Y/N/unknown + reason, executed price/stake/book, auto-stamped entry time); write-once pass files; `corrections.csv` append-only; daily files hash-chained | NBA |
| `pass_runner.py` | `capture → price → write_once → grade`, each step stamps `pass_id`, `code_sha`, `env_lock_hash`, credit spend, snapshot hash; a missed step or missed close sends an alert email instead of leaving a blank CLV | new |
| `scorecard.py` | the CBB six tables + page-one lights (CBB plan §4.d); **not** used by the five existing emails | NBA scorecard columns |

A sport package (`cbb_engine/`) supplies: market map, distribution object, gates, params YAMLs,
card templates. Nothing in `engine_runtime` imports a sport package.

## 3. Shared credit ledger — the one cross-sport change

**File:** `engine-state/runtime/credits/<YYYY-MM-DD>.csv` (append-only, one row per paid call:
`ts, sport, endpoint, markets, regions, events, cost, remaining_header, pass_id, allowed`), synced
through the existing engine-state push each sport already does. **Budget:** `runtime/credit_budget.yaml`
— daily key total, per-sport slice, and `cbb_featured_reserve` as a line item.

**API:** `engine_runtime.credits.debit(sport: str, cost: int) -> bool`.

```python
def debit(sport, cost):
    try:
        ledger = _load_today()                    # missing/corrupt → raise
        spent = ledger.spent(sport); total = ledger.spent()
        if spent + cost > budget[sport] or total + cost > key_total - reserve_unspent(): 
            ledger.append(..., allowed=False); return False
        ledger.append(..., allowed=True); return True
    except LedgerUnavailable:
        log.warning("credit ledger unavailable; falling through to client floor")
        return True                               # fail-open
```

**Wiring (each of the five `oddsapi.py`, inside the one `_get_json` each already has, before the
HTTP call):** `if not debit(SPORT, cost): log.info("skipped: credit budget"); return None` — the
callers already handle a `None`/empty board (that is today's behaviour when `min_credits` trips).
`cost` = markets × regions (featured) or per event × markets × regions (per-event), the same
arithmetic MLB's log line already prints. **No market list, parser, cache or price changes.**
Rule: a broken ledger never silences a live capture; a refused call is logged, never raised.
First month: `enforce: false` in the YAML — log only — so the five sports' real daily spend is
measured before any slice can refuse.

## 4. Audit emails — reorder only

Lead block, per sport, using numbers each report already computes; PPV / NPV / ROI stay directly
underneath, next to base rate and break-even **where those columns already exist**; the single-night
Play / Fade dot is removed from the lead (MLB `output/report.py` keeps its verdict table lower down).

| Sport | Lead line (definition printed on the line) | Interval | Where |
|---|---|---|---|
| MLB | "CLV vs consensus no-vig close, price taken" — mean ¢, n, % beat close | **none** (rows keyed on matchup text; mean + n only) | `mlb_engine/output/report.py` section order; `audit/email.py::send_audit_summary` body |
| NBA | "CLV at execution book, pre-tip close only; pulled props in `pre_pull_clv`" | game-clustered (ledger has `event_id`) — `clv_ev_lo/hi` already in `scorecard.py` | `nba_engine/audit/scorecard.py` render order |
| NHL | "CLV vs last archived pre-tip quote" — from `scorecard.py` `clv_sum/clv_n` | only if the row id is verified present; else mean + n | `nhl_engine/audit/scorecard.py::render` |
| CFB | CLV as computed today (verify column exists in `audit/scorecard.py`; if absent, lead with `edge_vs_be`, labelled, and note "CLV not captured") | none unless id present | `cfb_engine/audit/scorecard.py` |
| NFL | same rule as CFB (ratings/props research-only; label as such) | none | NFL audit renderer |

Untouched: probation rules (NHL still clears on sample size, ROI, Brier), thresholds, tier logic,
what later cards may buy. Not added: the six standard tables (CBB template only), game-id backfill
for MLB (that is the deferred MLB migration, `docs/mlb/runtime_migration_plan.md`).

## 5. Order
1. `engine_runtime/credits.py` + `credit_budget.yaml` + wiring in five clients, `enforce: false`
   (½ session; tests: refusal, fail-open on missing/corrupt file, reserve arithmetic, each
   client's `None` path).
2. Audit-email reorder, five renderers (½–1 session; tests: existing email/report tests plus
   ordering assertions; no numeric change — golden outputs identical except position).
3. `engine_runtime` skeleton: odds client, quotes store, odds_math, ledger row, pass runner
   (1–2 sessions) — CBB Phase 0 capture runs on it from ~Nov 3.
4. Flip `enforce: true` once a month of real spend is logged and the reserve is sized from it.
5. MLB migration: deferred; plan already written, re-opened only by explicit ask.

## 6. Checks before any of this is presented as code
ruff, mypy, pytest on the whole repo (no CI; local gate). For step 2, each sport's offline
replay per the `testing-*-engine` skills, confirming byte-identical numbers in the new order.
