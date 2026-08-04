#!/bin/bash
# Install the hands-off daily college-football schedule on macOS (launchd).
# Creates three LaunchAgents that run by themselves every day:
#   * com.payoffpitch.cfb.predictions -> 09:00  price + email today's card
#   * com.payoffpitch.cfb.close       -> 11:00/15:00/19:00/23:00  CLV snapshots
#   * com.payoffpitch.cfb.audit       -> 03:00  grade yesterday + email recap
#
# Times are local. Override the hours with CFB_RUN_HOUR / CFB_AUDIT_HOUR before
# running (minute is fixed at :00 for run/audit, close uses a fixed spread).
# Re-running is idempotent: each agent is unloaded before being reloaded.
set -e
cd "$(dirname "$0")/../../.." || exit 1
REPO="$(pwd)"
AUTORUN="$REPO/scripts/cfb/macos/autorun.command"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/.cfb_engine"

RUN_HOUR="${CFB_RUN_HOUR:-9}"
AUDIT_HOUR="${CFB_AUDIT_HOUR:-3}"

chmod +x "$AUTORUN"
mkdir -p "$LAUNCH_AGENTS" "$LOG_DIR"

# Seed the private credentials file the scheduled jobs read (no-op if present).
bash "$REPO/scripts/cfb/ensure_env.sh"

install_agent() {
    # $1 label  $2 calendar-interval-XML  $3.. cfb-engine args
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

single_time() {
    printf '    <key>StartCalendarInterval</key>\n    <dict>\n        <key>Hour</key><integer>%s</integer>\n        <key>Minute</key><integer>0</integer>\n    </dict>' "$1"
}

install_agent "com.payoffpitch.cfb.predictions" "$(single_time "$RUN_HOUR")" run
install_agent "com.payoffpitch.cfb.audit" "$(single_time "$AUDIT_HOUR")" audit

# Closing-line snapshots throughout the game day (repeat-safe since PR #69).
CLOSE_CAL='    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Hour</key><integer>11</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>15</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>19</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Hour</key><integer>23</integer><key>Minute</key><integer>0</integer></dict>
    </array>'
install_agent "com.payoffpitch.cfb.close" "$CLOSE_CAL" close

echo
echo "Done. The CFB card, CLV snapshots, and audit now run automatically every day."
echo "Logs: $LOG_DIR/schedule.log (errors: schedule_error.log)"
echo "Credentials must live in /etc/engine.env or $LOG_DIR/engine.env (launchd"
echo "does not read your shell profile)."
echo "Remove with: launchctl unload ~/Library/LaunchAgents/com.payoffpitch.cfb.*.plist"
