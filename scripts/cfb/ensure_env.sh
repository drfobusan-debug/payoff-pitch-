#!/bin/bash
# Seed ~/.cfb_engine/engine.env from engine.env.example on first install so the
# desktop shortcuts and the scheduled jobs have a private file to read
# credentials from (launchd/cron do not read your shell profile). Never
# overwrites an existing file. Called by the install_shortcuts / install_schedule
# scripts; safe to run standalone.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE="$HERE/engine.env.example"
DEST_DIR="$HOME/.cfb_engine"
DEST="$DEST_DIR/engine.env"

mkdir -p "$DEST_DIR"
if [ -f "$DEST" ]; then
    echo "Credentials file already present: $DEST (left untouched)"
    exit 0
fi

cp "$EXAMPLE" "$DEST"
chmod 600 "$DEST"
echo "Created $DEST (chmod 600) from the template."
echo ">> Edit it and fill in CFBD_API_KEY, THE_ODDS_API_KEY, GMAIL_USER,"
echo ">> GMAIL_APP_PASSWORD, and CFBE_EMAIL_TO before the jobs can email."
