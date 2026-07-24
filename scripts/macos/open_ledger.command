#!/bin/bash
# Open the audit ledger workbook (builds it via an audit if it doesn't exist yet).
cd "$(dirname "$0")/../.." || exit 1
LEDGER="$HOME/.mlb_engine/output/ledger.xlsx"
if [ ! -f "$LEDGER" ]; then
    echo "No ledger yet — running an audit to build it..."
    # shellcheck disable=SC1091
    source .venv/bin/activate
    mlb-engine audit
fi
if [ -f "$LEDGER" ]; then
    open "$LEDGER"
else
    echo "Still no ledger — audit a slate first (run_audit.command)."
fi
