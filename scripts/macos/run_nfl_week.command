#!/bin/bash
# One-click NFL week: archive the board, price it, re-stamp the close, grade what
# has finished, then write and email the card + workbook + PDF.
#
# Committed in the repository on purpose. The MLB launcher spent a season running
# from a copied Desktop folder that no `git pull` could reach, so this file refuses
# to run as a copy: put a *symlink* on the Desktop.
#   ln -sf "$HOME/payoff-pitch-/scripts/macos/run_nfl_week.command" ~/Desktop/"NFL WEEK.command"
set -uo pipefail
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

# Credentials (Gmail app password, Odds API key) from the same env files the MLB
# launcher reads, so one file serves both engines.
for _envf in /etc/engine.env "$HOME/.mlb_engine/engine.env" "$HOME/.nfl_engine/engine.env"; do
    if [ -f "$_envf" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$_envf"
        set +a
    fi
done

# WeasyPrint (the PDF card) needs Homebrew's native libs (glib/pango/cairo); a
# double-clicked shell does not inherit them. Requires `brew install pango`.
for _libdir in /opt/homebrew/lib /usr/local/lib; do
    if [ -d "$_libdir" ]; then
        export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:+$DYLD_FALLBACK_LIBRARY_PATH:}$_libdir"
    fi
done

# capture -> price -> close -> grade -> report -> card, with the close re-stamped
# on every run until kickoff. Safe to run repeatedly through the week; pricing
# appends only new positions and keeps the price of record.
nfl-engine job --card --email

OUT="${NFLE_OUTPUT_DIR:-$HOME/.nfl_engine/output}"
xlsx=$(ls -t "$OUT"/NFL_*.xlsx 2>/dev/null | head -1)
if [ -z "$xlsx" ]; then
    echo "No workbook produced: nothing was priced this week." >&2
    exit 1
fi
open "$xlsx"
