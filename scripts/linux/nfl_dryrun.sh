#!/bin/bash
# The unattended NFL paper dry run: capture, price, close, grade, report.
#
# No money is staked and none can be: the engine has no stake, bankroll or Kelly
# argument anywhere in it, and every ledger row is written mode=paper.
#
# Pricing appends only, so running this twice in a week does not double-count a
# position or overwrite the price of record. In the off-season every step is a
# no-op and the exit status is still 0.
#
#   30 12 * * 4 /path/to/scripts/linux/nfl_dryrun.sh >> ~/.nfl_engine/logs/dryrun.log 2>&1
cd "$(dirname "$0")/../.." || exit 1
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate

mkdir -p "$HOME/.nfl_engine/logs"
echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) NFL paper dry run"
nfl-engine job --props
