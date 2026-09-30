#!/bin/bash
# One-click CFB slate: price today's college football board, then write and
# email the article PDF + MP3 + workbook.
#
# Committed in the repository on purpose, like the NFL launcher beside it: this
# file refuses to run as a copy, so put a *symlink* on the Desktop.
#   ln -sf "$HOME/payoff-pitch-/scripts/macos/run_cfb_week.command" ~/Desktop/"CFB SLATE.command"
#
# The slate is one calendar day (US/Eastern): double-clicked on a Saturday it
# prices Saturday's board; pass --date YYYY-MM-DD for a Thursday or Friday card.
#
# Pass --check to resolve the checkout, credentials and libraries and print what
# would run, without pricing anything: the live path spends Odds API credits and
# sends mail.
set -uo pipefail
SELF_NAME=run_cfb_week.command
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
    echo "  rm \"$0\" && ln -s \"\$HOME/payoff-pitch-/scripts/macos/$SELF_NAME\" \"$0\"" >&2
    exit 1
fi
# shellcheck source=scripts/macos/_repo.sh
. "$(dirname "$_src")/_repo.sh"

# Credentials (Gmail app password, Odds API key, CFBD key) from the same env
# files the other launchers read, so one file serves every engine.
for _envf in /etc/engine.env "$HOME/.mlb_engine/engine.env" "$HOME/.cfb_engine/engine.env"; do
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

OUT="${CFBE_OUTPUT_DIR:-$HOME/.cfb_engine/output}"
if [ "${1:-}" = "--check" ]; then
    echo "checkout:    $REPO_DIR"
    echo "cfb-engine:  $(command -v cfb-engine || echo 'NOT FOUND -- run pip install -e .')"
    echo "data dir:    ${CFBE_DATA_DIR:-$HOME/.cfb_engine}"
    echo "output dir:  $OUT"
    echo "odds key:    $([ -n "${ODDS_API_KEY:-}${THE_ODDS_API_KEY:-}" ] && echo present || echo MISSING)"
    echo "cfbd key:    $([ -n "${CFBD_API_KEY:-}" ] && echo present || echo 'missing (ratings fall back to the ensemble)')"
    echo "email to:    ${CFBE_EMAIL_TO:-${CFB_EMAIL_TO:-${GMAIL_USER:-unset}}}"
    echo "gmail creds: $([ -n "${GMAIL_APP_PASSWORD:-}" ] && echo present || echo MISSING)"
    echo "would run:   cfb-engine run"
    echo "(--check spends no credits and sends no mail)"
    exit 0
fi

# Price the day, save predictions, write the workbook, then build and email the
# article PDF + MP3 with the workbook attached. Emails unless --no-email.
cfb-engine run "$@"

xlsx=$(ls -t "$OUT"/PayoffPitch_CFB_*.xlsx 2>/dev/null | head -1)
if [ -z "$xlsx" ]; then
    echo "No workbook produced: nothing was priced today." >&2
    exit 1
fi
open "$xlsx"
