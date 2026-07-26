#!/bin/bash
# Daily PayoffPitch run: slate predictions, master Excel sheet, and emailed PDF card.
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR" || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

OUTPUT_DIR="$HOME/.mlb_engine/output"
mkdir -p "$OUTPUT_DIR"

VSIN="$HOME/.mlb_engine/vsin_today.csv"
if [ -f "$VSIN" ]; then
    mlb-engine run --vsin-csv "$VSIN" --card --email
else
    mlb-engine run --card --email
fi

# Open the generated workbook for review.
latest=$(ls -t "$OUTPUT_DIR"/*.xlsx 2>/dev/null | head -1)
[ -n "$latest" ] && open "$latest"
