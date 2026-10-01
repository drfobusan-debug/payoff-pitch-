#!/bin/bash
# One-click NHL card (Phase 2): refresh the price archive, pull RotoWire's
# expected/confirmed goalies + injuries, then price the board from one joint
# sim per game. RotoWire being down is not fatal -- the card runs on the
# projected goalie and keeps the goalie_unconfirmed gate. The first run of the
# day writes the write-once ledger (--tag initial); later runs in the day
# re-price as a goalie/pre-drop pass and keep the morning ledger untouched.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"

TAG="${1:-initial}"
nhl-engine capture
echo
nhl-engine lineups || echo "(rotowire skipped)"
echo
nhl-engine card --tag "$TAG" --email
echo
echo "Buys are the pass_gate rows with a Strong/Moderate tier. Period markets are"
echo "priced for the ledger only until they clear probation (nhl-engine audit)."
