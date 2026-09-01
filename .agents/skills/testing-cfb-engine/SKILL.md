---
name: testing-cfb-engine
description: How to run and verify the payoff-pitch college-football (cfb_engine) CLI end-to-end — priced slate, first-seen board / drift baselines, closing snapshots, CLV, ledger columns and the drift gate — including how to work around exhausted CFBD quota.
---

# Testing the CFB engine (`cfb-engine`)

All testing is shell-only (no UI), so do not screen-record it. Collect command
output and file diffs as evidence instead.

## Setup

- venv: `/home/ubuntu/repos/payoff-pitch-/.venv`; entry point `.venv/bin/cfb-engine`.
- Credentials are already exported in the session shell (`CFBD_API_KEY`,
  `THE_ODDS_API_KEY`, `GMAIL_USER`, `GMAIL_APP_PASSWORD`). There may be **no**
  `~/.cfb_engine/engine.env` file — do not assume one exists; check `env` first.
- State lives in `~/.cfb_engine/`: `audit/` (predictions, `board_<date>.json`,
  `closing_<date>.json`, `ledger.csv`, `scorecard.csv`), `output/` (xlsx/pdf/mp3),
  `cache/oddsapi/`.
- Commands: `cfb-engine {run,card,close,audit,report,calibrate,scorecard} --date YYYY-MM-DD [--no-email]`.
  Always pass `--no-email` in tests.
- A `run` takes ~4 minutes. Budget for it; launch in background and poll.

## Making runs deterministic and credit-free

Export `CFBE_ODDS_CACHE_TTL=999999` so the cached Odds API board under
`~/.cfb_engine/cache/oddsapi/` is replayed instead of refetched. With the same
board replayed, `drift` is a pure function of the saved baseline file, which is
what makes baseline-perturbation tests exact. Note `cfb-engine close` always
fetches fresh (cache_ttl=0) and does spend credits.

## Pre-existing fail-soft noise (not regressions)

CFBD weather 401s, `sagarin.com` SSL verification failures, and CFBD 429s all
log warnings and the run still completes.

## CFBD quota is the usual blocker for `audit`

If CFBD returns `429 {"message":"Monthly call quota exceeded."}`, `run` still
works (fail-soft) but `audit` grades nothing, because grading needs
`CFBDClient.fetch_results` → `/games`. Check with:

```
curl -s -H "Authorization: Bearer $CFBD_API_KEY" \
  "https://api.collegefootballdata.com/games?year=2024&seasonType=regular&week=1" | head -c 200
```

Workaround that still exercises the real audit code path — stub only that one
method with real final scores from ESPN's free scoreboard API:

```python
# runner.py: .venv/bin/python runner.py audit --date 2024-08-31 --no-email
from cfb_engine.data.cfbd import CFBDClient, GameResult
import cfb_engine.cli as cli, sys
CFBDClient.fetch_results = lambda self, season, day: [...]  # from ESPN
sys.exit(cli.main(sys.argv[1:]))
```

ESPN source (real, no key):
`https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates=20240831&groups=80&limit=400`.
Always label this substitution explicitly in the test report.

To get graded rows at all you need a **past-season** date with real results;
Odds API will not price a past date, so hand-build
`audit/predictions_<date>.json` (a list of `Recommendation` dicts — `game_date`,
`matchup`, `market`, `selection`, `line`, `side`, `team_side`, `home_abbrev`,
`away_abbrev`, `market_american`, `fair_prob`, `tier`, plus `drift`/`pass_gate`)
and `audit/closing_<date>.json`. Use real matchups so CFBD/ESPN grading matches
by normalized team name.

## Snapshot / drift / CLV invariants worth asserting

- Snapshot keys are `"<matchup>|<market>|<side>"` with the handicap in the value
  (`line`). A slate of N games must show exactly 2N distinct `game_total` keys —
  if it collapses to 2, matchup dropped out of the key.
- `board_<date>.json` is **write-once**: a second `run` on the same date must
  leave the file byte-identical (`md5sum` before/after). Sides absent from the
  file are added without redefining existing ones.
- Drift signs (`market/linevalue.py`): positive = market moved **toward** the
  side bet. ATS `held - other`; over `other - held`; under `held - other`; ML is
  pure price (`to_prob - from_prob`). Handicap points convert at
  `1/(sd*sqrt(2*pi))` — 0.024934/pt at `margin_sd=16`, 0.030688/pt at `total_sd=13`.
- Test drift by editing the *saved baseline* file (lines and/or `no_vig_prob`),
  never the live board, then re-running with the odds cache pinned. Setting two
  different games' baselines to the same number is the sharpest cross-game
  contamination check: their drifts must differ.
- Drift reason strings are only attached to recs whose tier is not Pass
  (`pipeline._rate` skips the gate for Pass rows), so an empty `reasons` on a
  Pass row is expected, not a bug.
- Gate: `CFBE_DRIFT_GATE=1` plus a small `CFBE_DRIFT_MAX_ADVERSE` (e.g. 0.001)
  demotes adverse rows to `tier="Pass"` with `pass_gate="clv_drift"` and a reason
  ending `-> PASS`. With the gate off, identical drifts must demote nothing.
- Fail-soft: `chmod 444 ~/.cfb_engine/audit/board_<date>.json` after deleting an
  entry (so a write is attempted) — the run must still print
  "Wrote N recommendations" and only log
  `WARNING cfb_engine.pipeline: could not write first-seen board`.
- CLV: `compute_clv` must populate `clv/clv_ev/close_odds/close_prob/clv_pts`
  for ATS/totals rows **whose number moved**. A good pre-fix control is to
  rewrite `closing_<date>.json` with legacy 2-part keys
  (`game_total|Over 47.5`) — CLV then goes empty for every moved row, which is
  the bug the 3-part key fixes. `clv_pts` is empty for moneylines.
- Ledger: `~/.cfb_engine/audit/ledger.csv` carries `clv_pts`, `drift`,
  `pass_gate`. Because `update_ledger` reloads and rewrites the whole file,
  always audit **two** dates and re-check that the first date's values survive
  (a `load_ledger` that does not read a column back blanks it silently).

## Cleanup

Testing writes into the shared `~/.cfb_engine/audit`. Afterwards restore the
original `board_<date>.json`, and delete any synthetic
`predictions_*/closing_*/ledger.csv/scorecard.csv/PayoffPitch_CFB_Ledger_*.xlsx`
you created so later sessions do not grade fixtures as real history.

## Devin Secrets Needed

- `CFBD_API_KEY` (free, https://collegefootballdata.com/key) — monthly quota is
  small; grading is blocked once it is exhausted.
- `THE_ODDS_API_KEY` — live board pricing.
- `GMAIL_USER` / `GMAIL_APP_PASSWORD` — only if testing email delivery
  (otherwise always use `--no-email`).
