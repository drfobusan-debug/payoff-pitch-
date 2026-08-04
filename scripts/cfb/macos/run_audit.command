#!/bin/bash
# One-click college-football self-audit: grade the given day's slate (default:
# yesterday), update the ledger, and email the audit package (ledger workbook +
# recap PDF + MP3) in a single message, then open the ledger workbook locally.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

for _envf in /etc/engine.env "$HOME/.cfb_engine/engine.env"; do
    if [ -f "$_envf" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$_envf"
        set +a
    fi
done

for _libdir in /opt/homebrew/lib /usr/local/lib; do
    if [ -d "$_libdir" ]; then
        export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:+$DYLD_FALLBACK_LIBRARY_PATH:}$_libdir"
    fi
done

DAY="${1:-$(date -v-1d +%F 2>/dev/null || date -d 'yesterday' +%F)}"
cfb-engine audit --date "$DAY" || echo "WARN: audit step failed" >&2
echo "Ledger: $HOME/.cfb_engine/audit/ledger.csv"

ledger=$(ls -t "$HOME/.cfb_engine/audit/"PayoffPitch_CFB_Ledger_*.xlsx 2>/dev/null | head -1)
[ -n "$ledger" ] && open "$ledger"
