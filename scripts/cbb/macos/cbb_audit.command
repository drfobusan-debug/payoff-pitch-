#!/bin/bash
# 03:00: grade yesterday's bet-of-record rows on ESPN finals + close shards, email the six views.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"
cbb-engine audit --date "$(date -v-1d +%F 2>/dev/null || date -d yesterday +%F)" --email
