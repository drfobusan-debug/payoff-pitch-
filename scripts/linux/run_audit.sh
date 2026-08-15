#!/bin/bash
# Nightly self-audit: grade yesterday's recommendations, update the scorecard.
cd "$(dirname "$0")/../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate
# Yesterday's Opta calls, now carrying results; VSIN drops them after that.
mlb-engine opta --day -1 || echo "WARN: Opta benchmark capture failed" >&2
mlb-engine audit
echo "Scorecard: $HOME/.mlb_engine/audit/scorecard.csv"
