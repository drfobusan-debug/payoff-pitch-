#!/bin/bash
# Install the hands-off weekly NFL schedule on macOS (launchd).
# Creates two LaunchAgents that run by themselves:
#   * com.payoffpitch.nfl.week  -> 09:00 daily  capture, price, grade, email card
#   * com.payoffpitch.nfl.close -> 12:00/16:00/20:00  re-stamp the close for CLV
#
# Daily rather than weekly on purpose. `nfl-engine job` appends only positions it
# has not already priced and keeps the price of record, so running it every
# morning costs nothing but picks up a line that only became a buy on Friday --
# and in the off-season the whole job is a no-op that exits 0.
#
# The close agent is the half that cannot be backfilled: CLV is measured against
# the number at kickoff, and a price seen on Wednesday is not that number. Its
# three times bracket the Thursday, Sunday and Monday windows.
#
# Times are local. Override the card's hour with NFL_RUN_HOUR before running.
# Re-running is idempotent: each agent is unloaded before being reloaded.
set -e
cd "$(dirname "$0")/../../.." || exit 1
REPO="$(pwd)"
AUTORUN="$REPO/scripts/nfl/macos/autorun.command"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/.nfl_engine"

# shellcheck source=scripts/macos/protected_dir.sh
. "$REPO/scripts/macos/protected_dir.sh"
refuse_protected_dir "$REPO" || exit 1

RUN_HOUR="${NFL_RUN_HOUR:-9}"

chmod +x "$AUTORUN"
mkdir -p "$LAUNCH_AGENTS" "$LOG_DIR"

# Seed the private credentials file the scheduled jobs read (no-op if present).
bash "$REPO/scripts/nfl/ensure_env.sh"

install_agent() {
    # $1 label  $2 calendar-interval-XML  $3.. nfl-engine args
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
        <string>$AUTORUN</string>
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

WEEK_CAL="    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key><integer>$RUN_HOUR</integer>
        <key>Minute</key><integer>0</integer>
    </dict>"
install_agent "com.payoffpitch.nfl.week" "$WEEK_CAL" job --card --email

CLOSE_CAL='    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Hour</key><integer>12</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>16</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
    </array>'
install_agent "com.payoffpitch.nfl.close" "$CLOSE_CAL" close

echo
echo "Done. The NFL card and the closing snapshots now run by themselves."
echo "Logs: $LOG_DIR/schedule.log (errors: schedule_error.log)"
echo "Credentials must live in /etc/engine.env, $HOME/.mlb_engine/engine.env or"
echo "$LOG_DIR/engine.env (launchd does not read your shell profile)."
echo "Test it now with:"
echo "  launchctl kickstart -k gui/\$(id -u)/com.payoffpitch.nfl.week"
echo "Remove with: launchctl unload ~/Library/LaunchAgents/com.payoffpitch.nfl.*.plist"
