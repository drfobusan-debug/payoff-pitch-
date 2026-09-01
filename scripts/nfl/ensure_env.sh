#!/bin/bash
# Seed ~/.nfl_engine/engine.env from engine.env.example on first install so the
# scheduled jobs have a private file to read credentials from (launchd does not
# read your shell profile). Never overwrites an existing file. Called by
# install_schedule.command; safe to run standalone.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE="$HERE/engine.env.example"
DEST_DIR="$HOME/.nfl_engine"
DEST="$DEST_DIR/engine.env"

mkdir -p "$DEST_DIR"
if [ -f "$DEST" ]; then
    echo "Credentials file already present: $DEST (left untouched)"
    exit 0
fi

cp "$EXAMPLE" "$DEST"
chmod 600 "$DEST"
echo "Created $DEST (chmod 600) from the template."
echo ">> Edit it and fill in THE_ODDS_API_KEY, GMAIL_USER, GMAIL_APP_PASSWORD"
echo ">> and NFLE_EMAIL_TO before the jobs can email."
