#!/bin/bash
# Open the latest college-football audit ledger workbook (builds one via an
# audit of yesterday's slate if none exists yet).
cd "$(dirname "$0")/../../.." || exit 1
DIR="$HOME/.cfb_engine/audit"
latest=$(ls -t "$DIR/"PayoffPitch_CFB_Ledger_*.xlsx 2>/dev/null | head -1)
if [ -z "$latest" ]; then
    echo "No ledger yet — running an audit to build it..."
    # shellcheck disable=SC1091
    source .venv/bin/activate
    DAY="$(date -d 'yesterday' +%F 2>/dev/null || date -v-1d +%F)"
    cfb-engine audit --date "$DAY" --no-email
    latest=$(ls -t "$DIR/"PayoffPitch_CFB_Ledger_*.xlsx 2>/dev/null | head -1)
fi
if [ -n "$latest" ]; then
    xdg-open "$latest"
else
    echo "Still no ledger — audit a slate first (run_audit.sh)."
fi
