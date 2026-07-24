#!/bin/bash
# Create a "Ledger" desktop launcher (.desktop) with the ledger icon.
set -e
cd "$(dirname "$0")/../.." || exit 1
REPO="$(pwd)"
ICON="$REPO/assets/ledger.png"
TARGET="$REPO/scripts/linux/open_ledger.sh"

if [ ! -f "$ICON" ]; then
    echo "Generating ledger icon..."
    # shellcheck disable=SC1091
    source .venv/bin/activate 2>/dev/null || true
    python scripts/make_ledger_icon.py assets
fi
chmod +x "$TARGET"

DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
mkdir -p "$DESKTOP_DIR"
LAUNCHER="$DESKTOP_DIR/ledger.desktop"

cat > "$LAUNCHER" <<DESK
[Desktop Entry]
Type=Application
Name=Ledger
Comment=Open the MLB engine audit ledger
Exec=$TARGET
Icon=$ICON
Terminal=false
Categories=Office;
DESK
chmod +x "$LAUNCHER"
# Mark trusted where the desktop supports it (GNOME).
gio set "$LAUNCHER" metadata::trusted true 2>/dev/null || true

echo "Created $LAUNCHER"
echo "A green Ledger icon is now on your Desktop — double-click it to open the ledger."
