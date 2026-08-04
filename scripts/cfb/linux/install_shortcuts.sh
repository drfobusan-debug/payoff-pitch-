#!/bin/bash
# Create desktop launchers for the college-football engine:
#   * "CFB Predictions" -> run_predictions.sh (email today's card)
#   * "CFB Audit"       -> run_audit.sh (grade + email the audit)
#   * "CFB Ledger"      -> open_ledger.sh (open the latest ledger workbook)
set -e
cd "$(dirname "$0")/../../.." || exit 1
REPO="$(pwd)"
ICON="$REPO/assets/ledger.png"
SCRIPTS="$REPO/scripts/cfb/linux"

if [ ! -f "$ICON" ]; then
    echo "Generating ledger icon..."
    # shellcheck disable=SC1091
    source .venv/bin/activate 2>/dev/null || true
    python scripts/make_ledger_icon.py assets
fi

chmod +x "$SCRIPTS"/run_predictions.sh "$SCRIPTS"/run_audit.sh "$SCRIPTS"/open_ledger.sh

DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
mkdir -p "$DESKTOP_DIR"

make_launcher() {
    local name="$1" comment="$2" target="$3" file="$4"
    local launcher="$DESKTOP_DIR/$file"
    cat > "$launcher" <<DESK
[Desktop Entry]
Type=Application
Name=$name
Comment=$comment
Exec=$target
Icon=$ICON
Terminal=true
Categories=Office;
DESK
    chmod +x "$launcher"
    gio set "$launcher" metadata::trusted true 2>/dev/null || true
    echo "Created $launcher"
}

make_launcher "CFB Predictions" "Email today's college-football card (Excel + article + MP3)" \
    "$SCRIPTS/run_predictions.sh" "cfb_predictions.desktop"
make_launcher "CFB Audit" "Grade the latest slate and email the audit package" \
    "$SCRIPTS/run_audit.sh" "cfb_audit.desktop"
make_launcher "CFB Ledger" "Open the college-football audit ledger workbook" \
    "$SCRIPTS/open_ledger.sh" "cfb_ledger.desktop"

echo "Done — three CFB launchers are on your Desktop."
