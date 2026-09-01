#!/bin/bash
# Headless entry point for the scheduled NFL jobs (launchd).
# Usage: autorun.command <nfl-engine subcommand> [args...]
#   autorun.command job --card --email  -> capture, price, close, grade, report
#   autorun.command close               -> re-stamp the closing number for CLV
#
# Unlike "NFL WEEK.command" this never opens a workbook, so it is safe to run
# from launchd where there is no foreground GUI session.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

# Credentials (Gmail app password, Odds API key). launchd does not read your
# shell profile, so the schedule relies entirely on these files -- the same list
# the double-click launcher reads, so one file serves both entry points.
for _envf in /etc/engine.env "$HOME/.mlb_engine/engine.env" "$HOME/.nfl_engine/engine.env"; do
    if [ -f "$_envf" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$_envf"
        set +a
    fi
done

# WeasyPrint (the PDF card) needs Homebrew's native libs (glib/pango/cairo),
# which a non-login launchd job does not inherit.
for _libdir in /opt/homebrew/lib /usr/local/lib; do
    if [ -d "$_libdir" ]; then
        export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:+$DYLD_FALLBACK_LIBRARY_PATH:}$_libdir"
    fi
done

if [ "$#" -eq 0 ]; then
    echo "usage: autorun.command <job|close|capture> [args...]" >&2
    exit 2
fi

echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) nfl-engine $*"
exec nfl-engine "$@"
