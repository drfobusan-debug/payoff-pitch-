#!/bin/bash
# One-click NHL audit. Phase 0 has no ledger to grade; this pulls yesterday's
# official finals and summarises yesterday's price archive so the capture can
# be eyeballed against results. Becomes `nhl-engine audit` in Phase 2.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"

DAY="${1:-$(date -v-1d +%F 2>/dev/null || date -d 'yesterday' +%F)}"
nhl-engine results --date "$DAY" --compact
echo
nhl-engine archive --date "$DAY"
echo
echo "NOTE: no graded ledger yet -- the NHL engine is in Phase 0 (capture)."
