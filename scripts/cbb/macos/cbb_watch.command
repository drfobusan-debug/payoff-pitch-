#!/bin/bash
# The 5-minute watch: ESPN + RotoWire only. Spends no credits, never pulls a board.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"
cbb-engine watch "$@"
