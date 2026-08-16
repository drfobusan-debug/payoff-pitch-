#!/bin/bash
# Archive the NFL board (and player-prop prices) and nothing else.
#
# Safe to run as often as you like: a snapshot is written only when a price has
# moved, so an hourly cron leaves one file per move rather than one per run.
# Cheap in credits and it stakes nothing. Run it from preseason onward -- game
# prices can be recovered from nflverse afterwards, prop prices cannot be
# recovered from anywhere.
#
#   0 * * * * /path/to/scripts/linux/nfl_capture.sh >> ~/.nfl_engine/logs/capture.log 2>&1
cd "$(dirname "$0")/../.." || exit 1
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate

mkdir -p "$HOME/.nfl_engine/logs"
echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) capture"
nfl-engine capture --props
