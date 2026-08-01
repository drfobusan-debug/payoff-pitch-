Payoff Pitch — slate audit bundle (generated on Devin VM)
Slates: 2026-07-28, 07-29, 07-30, 07-31

Contents (mirrors your ~/.mlb_engine layout):
  audit/predictions_YYYY-MM-DD.json  — the priced slate (needed to grade/audit)
  audit/previews_YYYY-MM-DD.json     — per-game preview context (for the report)
  cache/boxscores/<gamePk>.json      — final box scores used for grading
  audit/ledger.csv / graded_metrics.csv / scorecard.csv
                                     — THIS VM's ledger, included ONLY so you can
                                       diff/merge vs the Mac's source-of-truth
                                       ledger. Do NOT copy these over the Mac's.

INSTALL on the Mac (non-destructive — does NOT touch your ledger.csv):
  1) cd ~/.mlb_engine
  2) cp -n /path/to/bundle/audit/predictions_*.json ~/.mlb_engine/audit/
     cp -n /path/to/bundle/audit/previews_*.json     ~/.mlb_engine/audit/
     cp -n /path/to/bundle/cache/boxscores/*.json    ~/.mlb_engine/cache/boxscores/
     (use -n so nothing you already have is overwritten; do NOT copy ledger.csv)

THEN audit each slate (appends to your ledger.csv, so only grade dates your
Mac has NOT already graded — you already got 07-28 and 07-29 by email, so
grading those again would double-count):
  mlb-engine audit --date 2026-07-30 --report
  mlb-engine audit --date 2026-07-31 --report   # after tonight's games are final

NOTE: predictions carry the PR #48 'opposite_american' field, so pull/merge
PR #48 first or the loader will reject the extra key.
