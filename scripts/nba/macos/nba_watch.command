#!/bin/bash
# One news-alarm tick: compare the board and both injury feeds with the last
# look, re-capture any game that moved, archive the alert, take any T-5 close.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"

nba-engine watch "$@"
