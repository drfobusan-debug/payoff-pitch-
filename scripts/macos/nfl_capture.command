#!/bin/zsh
# Double-click: archive the NFL board and player-prop prices. Stakes nothing.
#
# A snapshot is written only when a price has moved, so clicking this twice in a
# row leaves one file, not two.
cd "$(dirname "$0")/../.." || exit 1
[ -f .venv/bin/activate ] && source .venv/bin/activate
[ -f /etc/engine.env ] && set -a && source /etc/engine.env && set +a

mkdir -p "$HOME/.nfl_engine/logs"
echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) capture"
nfl-engine capture --props
