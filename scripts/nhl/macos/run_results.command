#!/bin/bash
# One-click official results: per-period scores, OT/SO flag and starting
# goalies from the NHL API for the given day (default: yesterday).
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"

DAY="${1:-$(date -v-1d +%F 2>/dev/null || date -d 'yesterday' +%F)}"
nhl-engine results --date "$DAY"
