#!/bin/bash
# One-click NHL card (Phase 2): refresh the price archive, pull RotoWire's
# expected/confirmed goalies + injuries, then price the board from one joint
# sim per game. RotoWire being down is not fatal -- the card runs on the
# projected goalie and keeps the goalie_unconfirmed gate. The first run of the
# day writes the write-once ledger (--tag initial); later runs in the day
# re-price as a goalie/pre-drop pass and keep the morning ledger untouched.
# A later pass (e.g. the scheduled pre-drop `goalie` run) captures the featured
# board only: ML/PL/totals are what it prices, and a full pass costs ~25 credits
# a game.
# The podcast step transcribes the Hockey Gambling Podcast episode for the
# slate (if one exists) so the PDF can show its read under each game. The audit
# step grades yesterday and rebuilds the ledger audit report, which the card
# email attaches alongside the PDF.
set -uo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/_env.sh"

TAG="${1:-initial}"
if [ "$TAG" = "initial" ]; then
    nhl-engine capture
else
    nhl-engine capture --board-only
fi
echo
nhl-engine lineups || echo "(rotowire skipped)"
echo
nhl-engine podcast || echo "(podcast skipped)"
echo
nhl-engine audit || echo "(audit skipped)"
echo
nhl-engine card --tag "$TAG" --email
echo
echo "Buys are the pass_gate rows with a Strong/Moderate tier. Period markets are"
echo "priced for the ledger only until they clear probation (nhl-engine audit)."
