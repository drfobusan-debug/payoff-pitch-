#!/bin/bash
# Nightly college-football self-audit: grade the given day's recommendations
# (default: yesterday), update the ledger, and email the audit package
# (ledger workbook + recap PDF + MP3).
cd "$(dirname "$0")/../../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

DAY="${1:-$(date -d 'yesterday' +%F 2>/dev/null || date -v-1d +%F)}"
cfb-engine audit --date "$DAY"
echo "Ledger: $HOME/.cfb_engine/audit/ledger.csv"
