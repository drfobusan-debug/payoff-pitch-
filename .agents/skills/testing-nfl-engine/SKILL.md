---
name: testing-nfl-engine
description: How to test the NFL engine (nfl_engine/) offline and for free — replay a season from nflverse, exercise the card/workbook/email delivery layer, price the research-only prop board, prove pricing/ledger neutrality, and avoid the traps that cause live Odds API spend or real emails. Use when testing anything under nfl_engine/, nfl-engine CLI commands, or scripts/macos/run_nfl_week.command.
---

# Testing the NFL engine

This is a **different engine** from the MLB one (see `testing-mlb-engine` for that). Different env
vars (`NFLE_*`), a different ledger, and a different launcher.

## Isolate every run — BOTH env vars, always

```bash
env -u THE_ODDS_API_KEY -u ODDS_API_KEY -u THE_ODDS_API_KEY_2 \
    NFLE_DATA_DIR=/tmp/nfl_A NFLE_OUTPUT_DIR=/tmp/nfl_A/out \
    .venv/bin/nfl-engine card --season 2024 --week 1
```

- `ledger_path()` = `<NFLE_DATA_DIR>/nfl_ledger.csv`; `output_dir()` = `NFLE_OUTPUT_DIR` or
  `<NFLE_DATA_DIR>/output`. **Setting only `NFLE_OUTPUT_DIR` silently falls back to `~/.nfl_engine`
  for data** — that will create/append the user's real ledger and board archives. Set both, every time.
- The `nfl-engine` console script may be declared in `pyproject.toml` but missing from an older venv.
  `.venv/bin/pip install -e . --no-deps` creates it; otherwise `python -m nfl_engine.cli` is the same
  entry point.
- nflverse data (`replay`, ratings, panels) is free public GitHub parquet, not the Odds API. Warm a
  cache once under a `/tmp` path and share it read-only across scratch data dirs; it is ~60-80 MB.

## DANGER: never PATH-shim the macOS launcher

`scripts/macos/run_nfl_week.command` line 48 is `nfl-engine job --card --email` — live Odds API **and**
a real email. The obvious way to neutralise it is a fake `nfl-engine` earlier on `PATH`. **This does
not work:** `scripts/macos/_repo.sh` sources `.venv/bin/activate`, which *prepends* `.venv/bin` to
`PATH` and defeats the shim. Doing this spends credits and sends mail for real.

Instead build a **fake checkout** and symlink into that:

```bash
R=/tmp/fake_repo
mkdir -p $R/mlb_engine $R/nfl_engine $R/.venv/bin $R/scripts/macos
cp scripts/macos/{_repo.sh,run_nfl_week.command} $R/scripts/macos/
cat > $R/.venv/bin/activate <<'EOF'
export PATH="/tmp/nfl_shim:$PATH"   # shim wins because activate is the stub
deactivate() { :; }
EOF
ln -s $R/scripts/macos/run_nfl_week.command "/tmp/desktop/NFL WEEK.command"
```

`_repo.sh` requires `mlb_engine/` **and** `.venv/bin/activate` to exist, hence both stubs. Use
`bash -x` to see `REPO_DIR` resolution, env-file sourcing and the DYLD loop. Note there may be **no
safe dry-run entry point** for the NFL launcher (a `nfl_dryrun.command` was deleted at one point), so
the fake checkout is the only safe way to exercise it.

Guard branches worth testing: a **copy** outside a checkout must exit 1 with link instructions; a
**symlink** must resolve into the checkout; a **half-checkout** (missing `.venv`) must exit 1 from
`_repo.sh` without reaching the CLI.

## A pure replay produces ZERO plays — this is correct

`replay` builds a board with exactly one book, and `screen()` vetoes any row with
`paired_books < 2`. So **every replayed row is a `Pass` with non-empty `screens`**, `build_card`
only makes a `Play` when `screens` is empty, and `_clv_sheet` filters `not e.screens`. A replay
therefore leaves the **Plays and CLV sheets empty by design** and the card reads
`0 plays over N games` / `No play: every price on this game was vetoed.`

Do not report that as a broken card, and do not try to prove play labels or CLV ordering from a
replay. Build a **small synthetic ledger** with the engine's own `save_ledger` (unvetoed rows with
`screens=""`, distinct `ev_fair`/`clv`, plus vetoed rows, a `benchmark_*` source row, and a row in a
different week) and drive it through the same `nfl-engine card` CLI.

Label shapes to assert: spread → `KC -2.5`, total → `OVER 47.5`, moneyline (`line=None`) → `KC ML`.

## The research-only prop layer (`nfl-engine props`)

`props` reads the **newest archived prop snapshot** and prices it. You never need `capture --props`
(which spends credits): write the snapshot yourself with the engine's own serializer, which
guarantees the exact schema `read_snapshot` expects.

```python
from nfl_engine.data import capture
capture.write_snapshot(rows, season=2024, week=5, kind=capture.PROP_KIND, root=root)
# -> <root>/captures/2024/wk05/prop-<ts>.csv ; latest_snapshot() globs "prop-*.csv" and takes the last
```

Key behaviours to assert, and the traps in each:

- **`--write` defaults to True**, so a bare `nfl-engine props` writes
  `<NFLE_DATA_DIR>/props/research_<season>_wk<nn>.csv` in **append** mode (header only when new).
  Re-running therefore grows the file — assert row growth + exactly one header, not idempotence.
- **Ledger isolation** is the headline guarantee: `write_research` never touches `nfl_ledger.csv`.
  Prove it with a *genuine* ledger (make one via offline `replay`, not by hand) and compare
  md5 + size + **mtime** before/after. Also grep the ledger for `research_only` (expect 0) and check
  `card`/`report` never surface prop rows — `card --week <w>` should still say `no priced rows` even
  when a research CSV exists for that week.
- **Every row must read as research, never as a bet**: `mode == "research"`, `research_only` present
  **and first** in the `;`-joined `screens`, `basis == "usage-shrunk-2016-2021"`.
- **Screens veto, they never drop** — vetoed rows stay in the CSV with reasons. But note a real
  asymmetry: an **unmapped market** (e.g. `player_anytime_td`) is *silently dropped* by `price_props`,
  while a `retired_market` keeps its row. Reconcile snapshot quotes → priced rows explicitly or your
  row counts will look wrong (best-price collapse also merges duplicate book/side quotes).
- **`ev_fair` (execution edge vs de-vigged consensus) and `edge_vs_fair` (model − market) are separate
  fields.** A row with `ev_fair` populated but `edge_vs_fair` empty (no paired model) is the clearest
  proof they are not collapsed.
- **`push_prob` needs a *projected* player on a *count* market.** `count_prob` only sets push when
  `line.is_integer()`; `yards_prob` is always 0. If the player has no projection the row's push is 0
  for an unrelated reason and the assertion is **vacuous** — pick players you have verified appear in
  `usage.projections(season, week)` (e.g. print the dict first), and use `player_rush_attempts`, not
  `player_carries`, as the carries market key.

## Leakage: the priced week must not inform its own projection

`usage.projections(season, week)` keeps only `week < week` of that season plus season−1 as an anchor,
and needs `MIN_GAMES` (4) prior games. Test it by monkeypatching **`nfl_engine.data.nflverse.player_week`**
(patch both the module attribute and `usage.nflverse.player_week`) with a synthetic DataFrame — no
network, fully deterministic.

Always include the **control**: inject the same monster game one week *earlier* and assert the mean
*moves*. Without it, an "unchanged" result could just mean your injected row never reached the loader,
and the leakage test proves nothing.

## Workbook invariants (openpyxl)

Sheets are exactly `["Plays", "Selections", "Record", "CLV"]`. Gotchas:

- The `Plays` sheet ends with a blank row **plus a paper-only note row**; filter it out or your
  "labels" list gets a `None` and sorting assertions blow up with a `TypeError`.
- `Selections` is week-scoped and includes vetoed **and** benchmark rows.
- `CLV` is sorted worst-move-first (non-decreasing) and excludes vetoed/non-engine rows, but is
  **not week-scoped** — rows from other weeks appear in a week's workbook by design (same as the
  season-to-date `Record` sheet). Don't file that as a leak without checking the docstring.

## Proving no credits and no mail

A wrapper that patches before importing the CLI is the cleanest proof:

- `socket.socket.connect`/`connect_ex` → raise on any address (for `card`, which needs no network at
  all), and assert the recorded attempt count is 0.
- Wrap `OddsAPIClient.__init__` to count constructions; `cmd_card` should construct **0** (the module
  is imported at `cli.py` top level, so import alone proves nothing).
- Replace `nfl_engine.output.email.smtplib.SMTP_SSL` with a capture stub recording host/port/from/to/
  subject/attachment filenames. **`GMAIL_APP_PASSWORD` is often set on the box**, so a bare
  `card --email` really sends. Expected attachments: `.md`, `.xlsx`, `.pdf`.
- Assert `THE_ODDS_API_KEY`, `ODDS_API_KEY` **and `THE_ODDS_API_KEY_2`** are all absent — it is easy
  to forget the third and have your own guard abort the run.

## Pricing-neutrality method

Replay twice into separate data dirs and diff the ledgers **by column group**, not by md5: group
pricing (`model_prob`, `fair_prob`, `ev_model`, `ev_fair`, `tier`, `screens`, `paired_books`, `odds`,
`opposite_odds`, `line`), grading (`result`, `pnl`, `clv`, `close_prob`, scores) and bookkeeping
(`captured_at`, `close_captured_at`, `date`, `mode`). Replay is deterministic (`DriveSim` is seeded)
in every pricing/grading column, but the two **`captured_at` stamps are wall-clock and always
differ**, so a raw md5 comparison shows a false diff. Then confirm `card` leaves the ledger
byte- and mtime-identical.

## Simulating the missing-PDF-libs failure

WeasyPrint may import fine here, so the "no renderer" case must be simulated with the real failure
shape — a `weasyprint.py` earlier on `PYTHONPATH` that raises at import:

```python
raise OSError("cannot load library 'libgobject-2.0-0': libgobject-2.0-0: "
              "cannot open shared object file: No such file or directory")
```

Expect exit 0 and `card PDF not rendered (…); markdown attached instead`.

## Known fragile spots (may already be fixed — verify before reporting)

`cmd_card` guards only the PDF and email steps. `out.mkdir()` and the md/html/xlsx writes are
unguarded, and `load_ledger` does a bare `csv.DictReader` + `int(row["season"])`. So these have each
produced an **uncaught traceback with no artifacts**: unwritable output dir, unwritable parent,
`NFLE_OUTPUT_DIR` pointing at a file, a `0x00` byte in the ledger, and non-numeric `season` or
`week`. `save_ledger` also raises `_csv.Error: need to escape…` on control characters. If you are
testing a fix, these are the exact repros; if you are testing something else, avoid them so they
don't mask your result.

By contrast, missing odds / missing `fair_prob` / empty matchup / malformed `kickoff_utc` / unicode
and emoji team names / truncated ledger / non-CSV garbage all degrade gracefully at exit 0.

**`capture.read_snapshot` has the same fragility** and it is on the `props` path: it skips a row whose
`american` is unusable, but does a bare `csv.DictReader` + `int(raw.get("season") or 0)`, so a
non-numeric `season`/`week` cell or a single `0x00` byte anywhere in the snapshot produces an uncaught
traceback (`capture.py` ~L256/L263) via `cmd_props`, with no research file written. A malformed
*archive file* can therefore kill `props` even though malformed *values* are handled.

The research **sink** is robust by contrast: `write_research` wraps everything in `except OSError` →
warn + `return None`, so root-is-a-file, CSV-path-is-a-directory, read-only CSV and read-only `props/`
all degrade to `research rows not written (see log)` at exit 0.

## Gates

Use the blueprint's own commands (path-based `mypy scripts` trips an unrelated module-mapping error
in `scripts/cfb/`):

```bash
.venv/bin/python -m pytest -q                                     # ~1760 tests, ~2 min
.venv/bin/ruff check mlb_engine cfb_engine nfl_engine tests
.venv/bin/mypy mlb_engine cfb_engine nfl_engine
```

`ruff format --check nfl_engine/cli.py` **fails on `main` already** (the `build_card(...)` call in
`cmd_card` is hand-wrapped). Scope a format gate to the files the branch actually changed, and check
whether `main` fails too before reporting it as a regression.

`scripts/nfl/props_study.py --cutoff 2021` is free (nflverse only) and finishes in a few minutes. Its
`worse than the base rate: retired` verdicts should map exactly onto `props.RETIRED_MARKETS` — compare
via `props.MARKET_STATS`, since the study prints *stat* names and the constant holds *market* keys.

## Devin Secrets Needed

- `THE_ODDS_API_KEY` (or `ODDS_API_KEY`) — **unset it** for offline testing; only `capture`/`price`/
  `close`/`job` spend credits.
- `GMAIL_APP_PASSWORD` — present on the box; stub `SMTP_SSL` or clear it so no mail is sent.
