#!/bin/bash
# One cadence tick: board (60/30/15 by time-to-tip), featured (10 min inside T-3h), closes (T-60..T-5).
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"
cbb-engine capture "$@"
