#!/bin/bash
# One-click NHL audit (Phase 2): grade yesterday's ledger against official
# finals (per-period, OT/SO, dual-rule flag, CLV) and print the running
# scorecard / probation table.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"

DAY="${1:-$(date -v-1d +%F 2>/dev/null || date -d 'yesterday' +%F)}"
nhl-engine results --date "$DAY" --compact
echo
nhl-engine audit --date "$DAY"
