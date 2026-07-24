#!/bin/bash
# Create a "Ledger" desktop shortcut with the ledger icon.
# Builds a tiny Ledger.app bundle on the Desktop whose icon is the ledger symbol
# (assets/ledger.icns) and whose action opens the ledger workbook.
set -e
cd "$(dirname "$0")/../.." || exit 1
REPO="$(pwd)"
ICON="$REPO/assets/ledger.icns"

if [ ! -f "$ICON" ]; then
    echo "Generating ledger icon..."
    # shellcheck disable=SC1091
    source .venv/bin/activate 2>/dev/null || true
    python scripts/make_ledger_icon.py assets
fi

APP="$HOME/Desktop/Ledger.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$ICON" "$APP/Contents/Resources/ledger.icns"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Ledger</string>
    <key>CFBundleDisplayName</key><string>Ledger</string>
    <key>CFBundleExecutable</key><string>ledger</string>
    <key>CFBundleIconFile</key><string>ledger</string>
    <key>CFBundleIdentifier</key><string>com.payoffpitch.ledger</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>1.0</string>
</dict>
</plist>
PLIST

# Launcher inside the bundle just opens the ledger workbook.
cat > "$APP/Contents/MacOS/ledger" <<LAUNCH
#!/bin/bash
exec "$REPO/scripts/macos/open_ledger.command"
LAUNCH
chmod +x "$APP/Contents/MacOS/ledger"

# Nudge Finder to pick up the new bundle icon.
touch "$APP"
/usr/bin/touch "$APP/Contents/Info.plist"
echo "Created $APP"
echo "A green Ledger icon is now on your Desktop — double-click it to open the ledger."
