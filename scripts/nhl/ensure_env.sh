#!/bin/bash
# Seed ~/.nhl_engine/engine.env from engine.env.example on first install so the
# desktop shortcuts and the scheduled capture have a private file to read the
# Odds API key from. Never overwrites an existing file. Safe to run standalone.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE="$HERE/engine.env.example"
DEST_DIR="$HOME/.nhl_engine"
DEST="$DEST_DIR/engine.env"

mkdir -p "$DEST_DIR"
if [ -f "$DEST" ]; then
    echo "Credentials file already present: $DEST (left untouched)"
    exit 0
fi

cp "$EXAMPLE" "$DEST"
chmod 600 "$DEST"
echo "Created $DEST (chmod 600) from the template."
echo ">> Edit it and fill in THE_ODDS_API_KEY (or keep it in /etc/engine.env)."
