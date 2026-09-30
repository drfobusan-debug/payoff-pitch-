#!/bin/bash
# Create .app launchers for the NHL engine in $NHLE_SHORTCUT_DIR
# (default ~/Desktop/engines, created if missing; older copies on the bare
# Desktop are removed):
#   * "NHL Capture.app"     -> nhl_capture.command   (archive today's board)
#   * "NHL Predictions.app" -> run_predictions.command
#   * "NHL Audit.app"       -> run_audit.command
#   * "NHL Results.app"     -> run_results.command   (official finals)
#   * "NHL Ledger.app"      -> open_ledger.command
# Each bundle just execs the matching .command script in this checkout, so a
# `git pull` updates what the shortcut does without reinstalling.
set -e
cd "$(dirname "$0")/../../.." || exit 1
REPO="$(pwd)"
ICON="$REPO/assets/ledger.icns"
SCRIPTS="$REPO/scripts/nhl/macos"

if [ ! -f "$ICON" ]; then
    echo "Generating ledger icon..."
    # shellcheck disable=SC1091
    source .venv/bin/activate 2>/dev/null || true
    python scripts/make_ledger_icon.py assets
fi

chmod +x "$SCRIPTS"/*.command
bash "$REPO/scripts/nhl/ensure_env.sh"

DEST="${NHLE_SHORTCUT_DIR:-$HOME/Desktop/engines}"
mkdir -p "$DEST"

make_app() {
    local name="$1" ident="$2" target="$3"
    local app="$DEST/$name.app"
    rm -rf "$app" "$HOME/Desktop/$name.app"
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
exec /usr/bin/open -a Terminal "$target"
LAUNCH
    chmod +x "$app/Contents/MacOS/launch"
    /usr/bin/touch "$app" "$app/Contents/Info.plist"
    echo "Created $app"
}

make_app "NHL Capture" "com.payoffpitch.nhl.capture" "$SCRIPTS/nhl_capture.command"
make_app "NHL Predictions" "com.payoffpitch.nhl.predictions" "$SCRIPTS/run_predictions.command"
make_app "NHL Audit" "com.payoffpitch.nhl.audit" "$SCRIPTS/run_audit.command"
make_app "NHL Results" "com.payoffpitch.nhl.results" "$SCRIPTS/run_results.command"
make_app "NHL Ledger" "com.payoffpitch.nhl.ledger" "$SCRIPTS/open_ledger.command"

echo "Done — five NHL launchers are in $DEST."
