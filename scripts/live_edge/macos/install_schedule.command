#!/bin/bash
# Install the CBB live-total paper test on macOS (launchd). Local clock, Eastern assumed.
#   * com.payoffpitch.live.cbb -> Saturdays at 10:00 (LIVE_CBB_WEEKDAYS, LIVE_CBB_HOUR):
#     `python -m live_edge.stoppage run --sport cbb --email`. It waits for the first tip, runs
#     until the slate is final and emails the report (GMAIL_* in ~/.cbb_engine/engine.env).
# Idempotent. Remove with: launchctl unload ~/Library/LaunchAgents/com.payoffpitch.live.cbb.plist
set -e
cd "$(dirname "$0")/../../.." || exit 1
REPO="$(pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/.live_edge"
# shellcheck source=scripts/macos/protected_dir.sh
. "$REPO/scripts/macos/protected_dir.sh"
refuse_protected_dir "$REPO" || exit 1
chmod +x "$REPO"/scripts/live_edge/macos/*.command
mkdir -p "$LAUNCH_AGENTS" "$LOG_DIR"

label="com.payoffpitch.live.cbb"
plist="$LAUNCH_AGENTS/$label.plist"
cal=""
for d in ${LIVE_CBB_WEEKDAYS:-6}; do
    cal="$cal        <dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>${LIVE_CBB_HOUR:-10}</integer><key>Minute</key><integer>0</integer></dict>
"
done
cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$label</string>
    <key>ProgramArguments</key><array><string>$REPO/scripts/live_edge/macos/live_cbb.command</string></array>
    <key>StartCalendarInterval</key>
    <array>
$cal    </array>
    <key>WorkingDirectory</key><string>$REPO</string>
    <key>StandardOutPath</key><string>$LOG_DIR/schedule.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/schedule_error.log</string>
</dict>
</plist>
PLIST
launchctl unload "$plist" 2>/dev/null || true
launchctl load "$plist"
echo "Installed $label -> $plist"
echo "Logs: $LOG_DIR/schedule.log. Report by hand: python -m live_edge.stoppage report --sport cbb"
