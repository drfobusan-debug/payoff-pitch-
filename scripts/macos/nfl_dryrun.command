#!/bin/zsh
# Double-click: the NFL paper dry run -- capture, price, close, grade, report.
#
# Nothing here can place a bet: there is no stake, bankroll or Kelly argument in
# the engine, and every ledger row is written mode=paper. Pricing appends only, so
# a second click does not double-count a position or rewrite the price of record.
cd "$(dirname "$0")/../.." || exit 1
[ -f .venv/bin/activate ] && source .venv/bin/activate
[ -f /etc/engine.env ] && set -a && source /etc/engine.env && set +a

mkdir -p "$HOME/.nfl_engine/logs"
echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) NFL paper dry run"
nfl-engine job --props
