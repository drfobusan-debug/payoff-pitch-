#!/bin/bash
# One-click daily MLB predictions -> opens the Excel workbook.
cd "$(dirname "$0")/../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

VSIN="$HOME/.mlb_engine/vsin_today.csv"
if [ -f "$VSIN" ]; then
    mlb-engine run --vsin-csv "$VSIN"
else
    mlb-engine run
fi

# Today's Opta calls, taken on the day: the page keeps no archive.
mlb-engine opta || echo "WARN: Opta benchmark capture failed" >&2

latest=$(ls -t "$HOME/.mlb_engine/output/"*.xlsx 2>/dev/null | head -1)
[ -n "$latest" ] && xdg-open "$latest"
