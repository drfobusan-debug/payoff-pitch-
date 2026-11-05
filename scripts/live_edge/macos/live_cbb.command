#!/bin/bash
# Saturday CBB live-total paper test: poll from before the first tip until the slate is final,
# then email this run's tickets and the running report. Paper only; at most --max-credits Odds API calls.
set -uo pipefail
# The CBB preamble: repo + .venv, and THE_ODDS_API_KEY / Gmail from ~/.cbb_engine/engine.env.
# shellcheck disable=SC1091
. "$(dirname "$0")/../../cbb/macos/_env.sh"
python -m live_edge.stoppage run --sport cbb --hours "${LIVE_CBB_HOURS:-15}" \
    --max-credits "${LIVE_CBB_MAX_CREDITS:-600}" --email
