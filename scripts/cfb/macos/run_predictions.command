#!/bin/bash
# One-click daily college-football card: prices today's slate and emails the
# Excel bet sheet + article (PDF) + MP3 narration in a single message, then
# opens the workbook locally. `cfb-engine run` does the pricing, packaging, and
# emailing in one step.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

# Load engine credentials (Gmail app password, Odds/CFBD API keys) from the same
# env file(s) the scheduled autorun uses, so a manual double-click can email too.
for _envf in /etc/engine.env "$HOME/.cfb_engine/engine.env"; do
    if [ -f "$_envf" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$_envf"
        set +a
    fi
done

# WeasyPrint (PDF articles) needs Homebrew's native libs (glib/pango/cairo).
# A double-clicked shell doesn't inherit them, so point the dynamic loader at
# the Homebrew lib dir(s). Requires `brew install pango`.
for _libdir in /opt/homebrew/lib /usr/local/lib; do
    if [ -d "$_libdir" ]; then
        export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:+$DYLD_FALLBACK_LIBRARY_PATH:}$_libdir"
    fi
done

cfb-engine run || echo "WARN: cfb-engine run failed" >&2

xlsx=$(ls -t "$HOME/.cfb_engine/output/"*.xlsx 2>/dev/null | head -1)
[ -n "$xlsx" ] && open "$xlsx"
