#!/bin/bash
# One-click daily MLB package: builds today's slate, the slate-preview article
# (+audio), and the pitcher/batter regression articles (+narration), emails them
# all in a single message, then opens the Excel bet card.
# Individual report/email steps only warn on failure so the rest still runs.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

VSIN="$HOME/.mlb_engine/vsin_today.csv"
if [ -f "$VSIN" ]; then
    mlb-engine run --vsin-csv "$VSIN"
else
    mlb-engine run
fi

OUT="$HOME/.mlb_engine/output"
xlsx=$(ls -t "$OUT"/mlb_recommendations_*.xlsx 2>/dev/null | head -1)
if [ -z "$xlsx" ]; then
    echo "No bet card produced; aborting package." >&2
    exit 1
fi
day=$(basename "$xlsx" .xlsx)
day=${day#mlb_recommendations_}

# Slate preview article + audio.
python -m scripts.regen_slate "$day" || echo "WARN: slate article failed" >&2

# Pitcher + batter regression articles + narration (need the trailing Statcast pkl).
pkl=$(ls -t "$HOME/.mlb_engine/cache/"statcast_*.pkl 2>/dev/null | head -1)
if [ -n "$pkl" ]; then
    python -m scripts.regen_regression "$day" "$(basename "$pkl")" \
        || echo "WARN: regression articles failed" >&2
else
    echo "WARN: no Statcast cache pkl found; skipping regression articles" >&2
fi

# Email the whole package as one message (Gmail App Password from the engine env).
python -m scripts.email_daily_package "$day" || echo "WARN: email step failed" >&2

# Open the bet card locally.
[ -n "$xlsx" ] && open "$xlsx"
