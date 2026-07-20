#!/bin/bash
# Nightly self-audit: grade yesterday's recommendations, update the scorecard.
cd "$(dirname "$0")/../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate
mlb-engine audit
echo "Scorecard: $HOME/.mlb_engine/audit/scorecard.csv"
