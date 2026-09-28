#!/bin/bash
# One-click NHL card. Phase 0 has no pricing model yet, so this refreshes the
# price archive and prints what has been captured for today; once the Phase 2
# pricer lands this becomes `nhl-engine run`.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"

nhl-engine capture
echo
nhl-engine archive
echo
echo "NOTE: no priced card yet -- the NHL engine is in Phase 0 (capture)."
echo "See docs/nhl/master_plan.md section 6 for what each phase ships."
