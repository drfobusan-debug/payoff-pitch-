#!/bin/bash
# One-click NHL price capture: archive today's featured board plus every
# period, team-total and player-prop market per game under
# ~/.nhl_engine/prices/<date>/. Idempotent -- an unchanged board writes nothing.
# Arguments are passed straight to `nhl-engine capture` (e.g. --board-only,
# --date YYYY-MM-DD); with none it captures today's slate (ET) in full.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"

nhl-engine capture "$@"
echo
echo "Archive: ${NHLE_DATA_DIR:-$HOME/.nhl_engine}/prices"
