#!/bin/bash
# One-click daily self-audit: grade yesterday's slate, build the audit report
# (Morningstar article + audio + Excel ledger), email it all in one message,
# then open the ledger workbook locally.
set -uo pipefail
# Enter the checkout, following symlinks: the Desktop icon should be a link to
# this file, not a copy of it. _repo.sh refuses to run outside a checkout.
_src=$0
while [ -L "$_src" ]; do
    _link=$(readlink "$_src")
    case $_link in
        /*) _src=$_link ;;
        *) _src=$(dirname "$_src")/$_link ;;
    esac
done
if [ ! -f "$(dirname "$_src")/_repo.sh" ]; then
    echo "A copy of this script cannot find the engine. Link to it instead:" >&2
    echo "  rm \"$0\" && ln -s \"\$HOME/payoff-pitch-/scripts/macos/$(basename "$_src")\" \"$0\"" >&2
    exit 1
fi
# shellcheck source=scripts/macos/_repo.sh
. "$(dirname "$_src")/_repo.sh"

# Load engine credentials (Gmail app password, Odds API key, etc.) from the same
# env file(s) the scheduled autorun uses, so a manual double-click can email too.
for _envf in /etc/engine.env "$HOME/.mlb_engine/engine.env"; do
    if [ -f "$_envf" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$_envf"
        set +a
    fi
done

# WeasyPrint (PDF report) needs Homebrew's native libs (glib/pango/cairo).
# A double-clicked shell doesn't inherit them, so point the dynamic loader at
# the Homebrew lib dir(s). Requires `brew install pango` (pulls glib/cairo).
for _libdir in /opt/homebrew/lib /usr/local/lib; do
    if [ -d "$_libdir" ]; then
        export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:+$DYLD_FALLBACK_LIBRARY_PATH:}$_libdir"
    fi
done

# Pull back yesterday's Opta calls now that they carry results. VSIN's day
# offset clamps at yesterday, so this is the last chance to have them at all.
mlb-engine opta --day -1 || echo "WARN: Opta benchmark capture failed" >&2

# Grade yesterday, write the report + audio + Excel ledger, and email them.
mlb-engine audit --report --email || echo "WARN: audit step failed" >&2
echo "Scorecard: $HOME/.mlb_engine/audit/scorecard.csv"
echo "Ledger:    $HOME/.mlb_engine/audit/ledger.csv"

# Open the ledger workbook locally.
[ -f "$HOME/.mlb_engine/output/ledger.xlsx" ] && open "$HOME/.mlb_engine/output/ledger.xlsx"
