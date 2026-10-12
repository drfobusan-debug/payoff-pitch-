#!/bin/bash
# Card pass (decision 5): 11:30 ET on days with a tip before 16:00 ET, 16:30 ET otherwise.
# `cbb-engine price` decides from the ESPN schedule whether this tick writes anything.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"
cbb-engine price --email
