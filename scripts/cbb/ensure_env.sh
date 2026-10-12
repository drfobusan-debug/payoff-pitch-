#!/bin/bash
# Seed ~/.cbb_engine/engine.env from engine.env.example on first install. Never overwrites.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$HOME/.cbb_engine"
DEST="$DEST_DIR/engine.env"
mkdir -p "$DEST_DIR" "$HOME/.engine_runtime"
if [ ! -f "$HOME/.engine_runtime/credit_budget.yaml" ]; then
    cp "$HERE/../../engine_runtime/credit_budget.example.yaml" "$HOME/.engine_runtime/credit_budget.yaml"
    echo "Seeded $HOME/.engine_runtime/credit_budget.yaml (enforce: false — observe-only)."
fi
if [ -f "$DEST" ]; then
    echo "Credentials file already present: $DEST (left untouched)"
    exit 0
fi
cp "$HERE/engine.env.example" "$DEST"
chmod 600 "$DEST"
echo "Created $DEST (chmod 600). Fill in THE_ODDS_API_KEY (or keep it in /etc/engine.env)."
