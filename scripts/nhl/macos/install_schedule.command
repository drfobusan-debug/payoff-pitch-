#!/bin/bash
# Install the hands-off NHL capture schedule on macOS (launchd).
#   * com.payoffpitch.nhl.board   -> every 30 min, 10:00-23:30 local: featured
#     board only (ML / puck line / total), 3 credits a pass.
#   * com.payoffpitch.nhl.capture -> at NHL_EVENT_HOURS (default 12 17 19):
#     the full per-event pass (periods, team totals, props), ~25 credits a game.
# Three full passes on a 15-game slate is ~1,100 credits a day; every-30-minute
# full passes would be ~10,000 and drain a 100k key in ten days.
# Idempotent: each agent is unloaded before being reloaded.
# Remove with: launchctl unload ~/Library/LaunchAgents/com.payoffpitch.nhl.*.plist
set -e
cd "$(dirname "$0")/../../.." || exit 1
REPO="$(pwd)"
CAPTURE="$REPO/scripts/nhl/macos/nhl_capture.command"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/.nhl_engine"

# shellcheck source=scripts/macos/protected_dir.sh
. "$REPO/scripts/macos/protected_dir.sh"
refuse_protected_dir "$REPO" || exit 1

chmod +x "$CAPTURE" "$REPO/scripts/nhl/macos/_env.sh"
mkdir -p "$LAUNCH_AGENTS" "$LOG_DIR"
bash "$REPO/scripts/nhl/ensure_env.sh"

FIRST_HOUR="${NHL_CAPTURE_FIRST_HOUR:-10}"
LAST_HOUR="${NHL_CAPTURE_LAST_HOUR:-23}"
EVENT_HOURS="${NHL_EVENT_HOURS:-12 17 19}"

install_agent() {
    # $1 label  $2 calendar-interval-XML  $3.. nhl_capture.command args
    local label="$1" calendar="$2"
    shift 2
    local plist="$LAUNCH_AGENTS/$label.plist"
    local args=""
    for a in "$@"; do
        args="$args        <string>$a</string>
"
    done
    cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>$CAPTURE</string>
$args    </array>
$calendar
    <key>WorkingDirectory</key>
    <string>$REPO</string>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/schedule.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/schedule_error.log</string>
</dict>
</plist>
PLIST
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load "$plist"
    echo "Installed $label -> $plist"
}

BOARD_CAL='    <key>StartCalendarInterval</key>
    <array>
'
for h in $(seq "$FIRST_HOUR" "$LAST_HOUR"); do
    for m in 0 30; do
        BOARD_CAL="$BOARD_CAL        <dict><key>Hour</key><integer>$h</integer><key>Minute</key><integer>$m</integer></dict>
"
    done
done
BOARD_CAL="$BOARD_CAL    </array>"

EVENT_CAL='    <key>StartCalendarInterval</key>
    <array>
'
for h in $EVENT_HOURS; do
    EVENT_CAL="$EVENT_CAL        <dict><key>Hour</key><integer>$h</integer><key>Minute</key><integer>5</integer></dict>
"
done
EVENT_CAL="$EVENT_CAL    </array>"

install_agent "com.payoffpitch.nhl.board" "$BOARD_CAL" --board-only
install_agent "com.payoffpitch.nhl.capture" "$EVENT_CAL"

echo
echo "Board every 30 min $FIRST_HOUR:00-$LAST_HOUR:30; full per-event pass at $EVENT_HOURS:05."
echo "Logs: $LOG_DIR/schedule.log (errors: schedule_error.log)"
echo "Credentials must live in /etc/engine.env or $LOG_DIR/engine.env."
echo "Test now: launchctl kickstart -k gui/\$(id -u)/com.payoffpitch.nhl.capture"
