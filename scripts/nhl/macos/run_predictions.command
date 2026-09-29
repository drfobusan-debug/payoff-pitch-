#!/bin/bash
# One-click NHL card (Phase 2): refresh the price archive, then price the
# board from one joint sim per game. The first run of the day writes the
# write-once ledger (--tag initial); later runs in the day re-price as a
# goalie/pre-drop pass and keep the morning ledger untouched.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"

TAG="${1:-initial}"
nhl-engine capture
echo
nhl-engine card --tag "$TAG"
echo
echo "Buys are the pass_gate rows with a Strong/Moderate tier. Period markets are"
echo "priced for the ledger only until they clear probation (nhl-engine audit)."
