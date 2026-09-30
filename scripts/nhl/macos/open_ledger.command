#!/bin/bash
# Open the NHL data folder (price archive; ledger workbook once Phase 2 ships).
set -uo pipefail
DATA="${NHLE_DATA_DIR:-$HOME/.nhl_engine}"
ledger=$(ls -t "$DATA/audit/"PayoffPitch_NHL_Ledger_*.xlsx 2>/dev/null | head -1)
if [ -n "$ledger" ]; then
    open "$ledger"
else
    mkdir -p "$DATA/prices"
    echo "No NHL ledger workbook yet (Phase 0 captures prices only); opening $DATA/prices"
    open "$DATA/prices"
fi
