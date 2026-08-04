#!/bin/bash
# Headless entry point for the scheduled college-football jobs (cron/systemd).
# Usage: autorun.sh <cfb-engine subcommand> [args...]
#   autorun.sh run     -> price today's slate and email the card
#   autorun.sh close   -> snapshot the closing market for CLV
#   autorun.sh audit   -> grade yesterday, update the ledger, email recap
#
# Unlike the double-click shortcuts this never opens a workbook, so it is safe
# to run from cron where there is no display.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

# Load engine credentials (Gmail app password, Odds/CFBD API keys). cron does not
# read your shell profile, so the schedule relies entirely on these files.
for _envf in /etc/engine.env "$HOME/.cfb_engine/engine.env"; do
    if [ -f "$_envf" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$_envf"
        set +a
    fi
done

if [ "$#" -eq 0 ]; then
    echo "usage: autorun.sh <run|close|audit> [args...]" >&2
    exit 2
fi

exec cfb-engine "$@"
