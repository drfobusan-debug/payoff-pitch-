# Enter the checkout this file lives in, or say so and stop.
#
# Sourced by the .command scripts beside it, which find the engine by walking up
# from their own path. That holds exactly as long as they are IN the checkout --
# and a Desktop icon is usually made by dragging, which copies. A copy in
# ~/Desktop resolves ../.. to /Users, where activating the venv fails, and with
# no `set -e` the run carries on against whatever mlb-engine is on PATH: a full
# slate priced by a build months behind, quietly missing every column and fix
# since. So the checkout is verified rather than assumed.
#
# ``BASH_SOURCE`` rather than ``$0``: this file is only ever read from inside the
# checkout, while the thing being double-clicked may be a symlink to it -- which
# is the right way to put one of these on the Desktop, since it cannot go stale.

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)

if [ ! -d "$REPO_DIR/mlb_engine" ] || [ ! -f "$REPO_DIR/.venv/bin/activate" ]; then
    echo "No payoff-pitch checkout around this script; it looked in $REPO_DIR." >&2
    echo "Run scripts/macos/setup.command in the checkout first." >&2
    exit 1
fi

cd "$REPO_DIR" || exit 1
# shellcheck disable=SC1091
. "$REPO_DIR/.venv/bin/activate"
