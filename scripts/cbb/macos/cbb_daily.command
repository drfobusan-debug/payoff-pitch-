#!/bin/bash
# Once a day: archive Torvik files (as-of replay inputs) and yesterday's ESPN finals.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"
cbb-engine torvik
cbb-engine results --date "$(date -v-1d +%F 2>/dev/null || date -d yesterday +%F)"
