#!/bin/bash
# Shared preamble for the CBB .command shortcuts. Sourced, not executed.
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
if [ ! -d "$REPO_DIR/cbb_engine" ] || [ ! -f "$REPO_DIR/.venv/bin/activate" ]; then
    echo "No payoff-pitch checkout with cbb_engine around this script; looked in $REPO_DIR." >&2
    echo "Run scripts/macos/setup.command in the checkout first." >&2
    exit 1
fi
cd "$REPO_DIR" || exit 1
# shellcheck disable=SC1091
. "$REPO_DIR/.venv/bin/activate"
for _envf in /etc/engine.env "$HOME/.cbb_engine/engine.env"; do
    if [ -f "$_envf" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$_envf"
        set +a
    fi
done
