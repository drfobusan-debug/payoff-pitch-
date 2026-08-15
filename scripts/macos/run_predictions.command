#!/bin/bash
# One-click daily MLB package: builds today's slate, the slate-preview article
# (+audio), and the pitcher/batter regression articles (+narration), emails them
# all in a single message, then opens the Excel bet card.
# Individual report/email steps only warn on failure so the rest still runs.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

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

# WeasyPrint (PDF articles) needs Homebrew's native libs (glib/pango/cairo).
# A double-clicked shell doesn't inherit them, so point the dynamic loader at
# the Homebrew lib dir(s). Requires `brew install pango` (pulls glib/cairo).
for _libdir in /opt/homebrew/lib /usr/local/lib; do
    if [ -d "$_libdir" ]; then
        export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:+$DYLD_FALLBACK_LIBRARY_PATH:}$_libdir"
    fi
done

# Archive today's saved propsheet prices before anything is priced. There is no
# historical archive of prop lines to buy, so a price not captured on the day is
# gone; the import finds the newest save in ~/Downloads by itself.
python -m scripts.propsheet_import || echo "WARN: propsheet price archive failed" >&2

VSIN="$HOME/.mlb_engine/vsin_today.csv"
if [ -f "$VSIN" ]; then
    mlb-engine run --vsin-csv "$VSIN"
else
    mlb-engine run
fi

# Today's Opta calls, taken while the page still offers them: the benchmark
# only exists if it is captured on the day.
mlb-engine opta || echo "WARN: Opta benchmark capture failed" >&2

OUT="$HOME/.mlb_engine/output"
xlsx=$(ls -t "$OUT"/mlb_recommendations_*.xlsx 2>/dev/null | head -1)
if [ -z "$xlsx" ]; then
    echo "No bet card produced; aborting package." >&2
    exit 1
fi
day=$(basename "$xlsx" .xlsx)
day=${day#mlb_recommendations_}

# Slate preview article + audio.
python -m scripts.regen_slate "$day" || echo "WARN: slate article failed" >&2

# Pitcher + batter regression articles + narration (need the trailing Statcast pkl).
pkl=$(ls -t "$HOME/.mlb_engine/cache/"statcast_*.pkl 2>/dev/null | head -1)
if [ -n "$pkl" ]; then
    python -m scripts.regen_regression "$day" "$(basename "$pkl")" \
        || echo "WARN: regression articles failed" >&2
else
    echo "WARN: no Statcast cache pkl found; skipping regression articles" >&2
fi

# Email the whole package as one message (Gmail App Password from the engine env).
python -m scripts.email_daily_package "$day" || echo "WARN: email step failed" >&2

# Open the bet card locally.
[ -n "$xlsx" ] && open "$xlsx"
