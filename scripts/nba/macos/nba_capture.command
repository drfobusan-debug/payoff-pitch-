#!/bin/bash
# One-click NBA price capture: archive today's featured board plus every game's
# first-half and five player-prop markets under ~/.nba_engine/prices/<date>/.
# Idempotent -- an unchanged board writes nothing. Arguments go straight to
# `nba-engine capture` (e.g. --board-only, --date YYYY-MM-DD).
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"

nba-engine capture "$@"
echo
echo "Archive: ${NBAE_DATA_DIR:-$HOME/.nba_engine}/prices"
