#!/bin/bash
# Create Desktop .app launchers for the college-football engine:
#   * "CFB Predictions.app" -> run_predictions.command (email today's card)
#   * "CFB Audit.app"       -> run_audit.command (grade + email the audit)
#   * "CFB Ledger.app"      -> open_ledger.command (open the latest ledger)
# Each bundle uses the ledger icon and just execs the matching .command script.
set -e
cd "$(dirname "$0")/../../.." || exit 1
REPO="$(pwd)"
ICON="$REPO/assets/ledger.icns"
SCRIPTS="$REPO/scripts/cfb/macos"

if [ ! -f "$ICON" ]; then
    echo "Generating ledger icon..."
    # shellcheck disable=SC1091
    source .venv/bin/activate 2>/dev/null || true
    python scripts/make_ledger_icon.py assets
fi

chmod +x "$SCRIPTS"/run_predictions.command "$SCRIPTS"/run_audit.command "$SCRIPTS"/open_ledger.command

make_app() {
    local name="$1" ident="$2" target="$3"
    local app="$HOME/Desktop/$name.app"
    rm -rf "$app"
    mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
    cp "$ICON" "$app/Contents/Resources/ledger.icns"
    cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$name</string>
    <key>CFBundleDisplayName</key><string>$name</string>
    <key>CFBundleExecutable</key><string>launch</string>
    <key>CFBundleIconFile</key><string>ledger</string>
    <key>CFBundleIdentifier</key><string>$ident</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>1.0</string>
</dict>
</plist>
PLIST
    cat > "$app/Contents/MacOS/launch" <<LAUNCH
#!/bin/bash
exec "$target"
LAUNCH
    chmod +x "$app/Contents/MacOS/launch"
    /usr/bin/touch "$app" "$app/Contents/Info.plist"
    echo "Created $app"
}

make_app "CFB Predictions" "com.payoffpitch.cfb.predictions" "$SCRIPTS/run_predictions.command"
make_app "CFB Audit" "com.payoffpitch.cfb.audit" "$SCRIPTS/run_audit.command"
make_app "CFB Ledger" "com.payoffpitch.cfb.ledger" "$SCRIPTS/open_ledger.command"

echo "Done — three CFB launchers are on your Desktop."
