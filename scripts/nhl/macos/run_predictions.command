#!/bin/bash
# One-click NHL card. No pricing model yet (Phase 2), so this refreshes the
# price archive, refreshes the preseason prior on its weekly schedule and
# prints today's as-of team-strength posteriors (Phase 1, features only).
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"

nhl-engine capture
echo
nhl-engine archive
echo
nhl-engine strength --metrics xgf60_5v5,xga60_5v5,pp_xgf60,pk_xga60
echo
echo "NOTE: no priced card yet -- the NHL engine is in Phase 1 (features, no prices)."
echo "See docs/nhl/master_plan.md section 6 for what each phase ships."
