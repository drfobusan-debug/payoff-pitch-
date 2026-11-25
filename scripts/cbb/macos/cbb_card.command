#!/bin/bash
# Card pass (decision 5): 11:30 ET on days with a tip before 16:00 ET, 16:30 ET otherwise.
# `cbb-engine price` decides from the ESPN schedule whether this tick writes anything.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"
# New podcast episodes first, so the card prints this morning's picks (display only);
# the feed read stops after PODCAST_MAX_SECONDS so a stalled read never holds the card back.
cbb-engine podcast --max-seconds "${PODCAST_MAX_SECONDS:-3600}" || echo "podcast read failed; the card prints without new picks" >&2
cbb-engine price --email
