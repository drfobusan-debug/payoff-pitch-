#!/bin/bash
# Install the CBB Phase-0 capture schedule on macOS (launchd). Local clock, Eastern assumed.
#   * com.payoffpitch.cbb.capture -> every 5 min 09:00-23:55: `cbb-engine capture`, which
#     runs only the passes the §5 cadence table says are due (board 60/30/15 by time-to-tip,
#     featured every 10 min inside T-3h, close passes T-60..T-5 per game).
#   * com.payoffpitch.cbb.watch   -> every 5 min 11:00-23:55: ESPN + RotoWire, no credits.
#   * com.payoffpitch.cbb.daily   -> 06:00: Torvik archive + yesterday's finals.
#   * com.payoffpitch.cbb.card    -> 11:30 and 16:30: `cbb-engine price --email` (bet of record).
#   * com.payoffpitch.cbb.audit   -> 03:00: `cbb-engine audit --email` on yesterday's rows.
# Idempotent. Remove with: launchctl unload ~/Library/LaunchAgents/com.payoffpitch.cbb.*.plist
set -e
cd "$(dirname "$0")/../../.." || exit 1
REPO="$(pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/.cbb_engine"
# shellcheck source=scripts/macos/protected_dir.sh
. "$REPO/scripts/macos/protected_dir.sh"
refuse_protected_dir "$REPO" || exit 1
chmod +x "$REPO"/scripts/cbb/macos/*.command "$REPO/scripts/cbb/macos/_env.sh"
mkdir -p "$LAUNCH_AGENTS" "$LOG_DIR"
bash "$REPO/scripts/cbb/ensure_env.sh"

install_agent() {
    local label="$1" program="$2" calendar="$3"
    local plist="$LAUNCH_AGENTS/$label.plist"
    cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$label</string>
    <key>ProgramArguments</key><array><string>$program</string></array>
$calendar
    <key>WorkingDirectory</key><string>$REPO</string>
    <key>StandardOutPath</key><string>$LOG_DIR/schedule.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/schedule_error.log</string>
</dict>
</plist>
PLIST
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load "$plist"
    echo "Installed $label -> $plist"
}

every5() {
    local first="$1" last="$2" cal='    <key>StartCalendarInterval</key>
    <array>
'
    for h in $(seq "$first" "$last"); do
        for m in 0 5 10 15 20 25 30 35 40 45 50 55; do
            cal="$cal        <dict><key>Hour</key><integer>$h</integer><key>Minute</key><integer>$m</integer></dict>
"
        done
    done
    printf '%s    </array>' "$cal"
}

install_agent "com.payoffpitch.cbb.capture" "$REPO/scripts/cbb/macos/cbb_capture.command" "$(every5 "${CBB_CAPTURE_FIRST_HOUR:-9}" "${CBB_CAPTURE_LAST_HOUR:-23}")"
install_agent "com.payoffpitch.cbb.watch" "$REPO/scripts/cbb/macos/cbb_watch.command" "$(every5 "${CBB_WATCH_FIRST_HOUR:-11}" "${CBB_WATCH_LAST_HOUR:-23}")"
install_agent "com.payoffpitch.cbb.daily" "$REPO/scripts/cbb/macos/cbb_daily.command" "    <key>StartCalendarInterval</key>
    <dict><key>Hour</key><integer>${CBB_DAILY_HOUR:-6}</integer><key>Minute</key><integer>0</integer></dict>"
install_agent "com.payoffpitch.cbb.card" "$REPO/scripts/cbb/macos/cbb_card.command" "    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Hour</key><integer>11</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Hour</key><integer>16</integer><key>Minute</key><integer>30</integer></dict>
    </array>"
install_agent "com.payoffpitch.cbb.audit" "$REPO/scripts/cbb/macos/cbb_audit.command" "    <key>StartCalendarInterval</key>
    <dict><key>Hour</key><integer>3</integer><key>Minute</key><integer>0</integer></dict>"
echo "Logs: $LOG_DIR/schedule.log. Credit ledger: ~/.engine_runtime/credits/ (observe-only until enforce: true)."
