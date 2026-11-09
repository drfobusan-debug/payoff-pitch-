#!/bin/bash
# Shared preamble for the CBB .command shortcuts. Sourced, not executed.
# Sourced from a zsh prompt BASH_SOURCE is empty (it resolved to "/");
# zsh names the sourced file with %x.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _self=${BASH_SOURCE[0]}
elif [ -n "${ZSH_VERSION:-}" ]; then
    eval '_self=${(%):-%x}'
else
    _self=$0
fi
REPO_DIR=$(cd "$(dirname "$_self")/../../.." && pwd -P)
if [ ! -d "$REPO_DIR/cbb_engine" ] || [ ! -f "$REPO_DIR/.venv/bin/activate" ]; then
    echo "No payoff-pitch checkout with cbb_engine around this script; looked in $REPO_DIR." >&2
    echo "Run scripts/macos/setup.command in the checkout first." >&2
    case $- in *i*) return 1 ;; esac
    exit 1
fi
cd "$REPO_DIR" || { case $- in *i*) return 1 ;; esac; exit 1; }
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

# WeasyPrint (the card PDF) needs Homebrew's glib/pango/cairo; launchd jobs don't see them.
for _libdir in /opt/homebrew/lib /usr/local/lib; do
    if [ -d "$_libdir" ]; then
        export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:+$DYLD_FALLBACK_LIBRARY_PATH:}$_libdir"
    fi
done
