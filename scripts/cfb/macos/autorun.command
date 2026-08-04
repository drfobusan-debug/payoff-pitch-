#!/bin/bash
# Headless entry point for the scheduled college-football jobs (launchd).
# Usage: autorun.command <cfb-engine subcommand> [args...]
#   autorun.command run     -> price today's slate and email the card
#   autorun.command close   -> snapshot the closing market for CLV
#   autorun.command audit   -> grade yesterday, update the ledger, email recap
#
# Unlike the double-click shortcuts this never opens a workbook, so it is safe
# to run from launchd where there is no foreground GUI session.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

# Load engine credentials (Gmail app password, Odds/CFBD API keys). launchd does
# not read your shell profile, so the schedule relies entirely on these files.
for _envf in /etc/engine.env "$HOME/.cfb_engine/engine.env"; do
    if [ -f "$_envf" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$_envf"
        set +a
    fi
done

# WeasyPrint (PDF articles) needs Homebrew's native libs (glib/pango/cairo),
# which a non-login launchd job does not inherit.
for _libdir in /opt/homebrew/lib /usr/local/lib; do
    if [ -d "$_libdir" ]; then
        export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:+$DYLD_FALLBACK_LIBRARY_PATH:}$_libdir"
    fi
done

if [ "$#" -eq 0 ]; then
    echo "usage: autorun.command <run|close|audit> [args...]" >&2
    exit 2
fi

exec cfb-engine "$@"
