#!/bin/bash
# Install the hands-off NBA capture schedule on macOS (launchd). Hours are the
# Mac's local clock; the defaults assume Eastern time.
#   * com.payoffpitch.nba.watch   -> every 5 min, 11:00-23:55: the news alarm
#     (featured board 3 credits, ESPN injury feed and official report free;
#     ~8 credits per re-captured game) and the T-5 close per game.
#   * com.payoffpitch.nba.capture -> at NBA_EVENT_HOURS:NBA_EVENT_MINUTE (default
#     11 17 20 at :05, just after the 11 AM, 5 PM and 8 PM injury reports): the
#     full first-half and props pass, ~8 credits a game.
#   * com.payoffpitch.nba.card    -> at NBA_CARD_HOUR:NBA_CARD_MINUTE (default
#     17:15, after the 5 PM injury report and the 17:05 full pass): a board-only
#     pricing pass (3 credits), yesterday's grade (free), then the slate PDF +
#     workbook emailed. 17:15 is a DEFAULT, to be replaced by the fitted send
#     time (docs/nba/master_plan.md §10) once the audit can fit it.
# Also links "NBA CARD.command" on the Desktop to run the card by hand.
# Idempotent: each agent is unloaded before being reloaded.
# Remove with: launchctl unload ~/Library/LaunchAgents/com.payoffpitch.nba.*.plist
set -e
cd "$(dirname "$0")/../../.." || exit 1
REPO="$(pwd)"
CAPTURE="$REPO/scripts/nba/macos/nba_capture.command"
WATCH="$REPO/scripts/nba/macos/nba_watch.command"
CARD="$REPO/scripts/nba/macos/nba_card.command"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/.nba_engine"

# shellcheck source=scripts/macos/protected_dir.sh
. "$REPO/scripts/macos/protected_dir.sh"
refuse_protected_dir "$REPO" || exit 1

chmod +x "$CAPTURE" "$WATCH" "$CARD" "$REPO/scripts/nba/macos/_env.sh"
mkdir -p "$LAUNCH_AGENTS" "$LOG_DIR"
bash "$REPO/scripts/nba/ensure_env.sh"

FIRST_HOUR="${NBA_WATCH_FIRST_HOUR:-11}"
LAST_HOUR="${NBA_WATCH_LAST_HOUR:-23}"
EVENT_HOURS="${NBA_EVENT_HOURS:-11 17 20}"
EVENT_MINUTE="${NBA_EVENT_MINUTE:-5}"
# Default send time, not yet fitted (plan §10): replace with the fitted time.
CARD_HOUR="${NBA_CARD_HOUR:-17}"
CARD_MINUTE="${NBA_CARD_MINUTE:-15}"

install_agent() {
    # $1 label  $2 program  $3 calendar-interval-XML  $4.. program args
    local label="$1" program="$2" calendar="$3"
    shift 3
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
        <string>$program</string>
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

WATCH_CAL='    <key>StartCalendarInterval</key>
    <array>
'
for h in $(seq "$FIRST_HOUR" "$LAST_HOUR"); do
    for m in 0 5 10 15 20 25 30 35 40 45 50 55; do
        WATCH_CAL="$WATCH_CAL        <dict><key>Hour</key><integer>$h</integer><key>Minute</key><integer>$m</integer></dict>
"
    done
done
WATCH_CAL="$WATCH_CAL    </array>"

EVENT_CAL='    <key>StartCalendarInterval</key>
    <array>
'
for h in $EVENT_HOURS; do
    EVENT_CAL="$EVENT_CAL        <dict><key>Hour</key><integer>$h</integer><key>Minute</key><integer>$EVENT_MINUTE</integer></dict>
"
done
EVENT_CAL="$EVENT_CAL    </array>"

install_agent "com.payoffpitch.nba.watch" "$WATCH" "$WATCH_CAL"
install_agent "com.payoffpitch.nba.capture" "$CAPTURE" "$EVENT_CAL"

CARD_CAL="    <key>StartCalendarInterval</key>
    <dict><key>Hour</key><integer>$CARD_HOUR</integer><key>Minute</key><integer>$CARD_MINUTE</integer></dict>"
install_agent "com.payoffpitch.nba.card" "$CARD" "$CARD_CAL"

if [ -d "$HOME/Desktop" ]; then
    ln -sfn "$CARD" "$HOME/Desktop/NBA CARD.command"
    echo "Linked $HOME/Desktop/NBA CARD.command -> $CARD"
fi

echo
echo "News alarm every 5 min $FIRST_HOUR:00-$LAST_HOUR:55; full pass at $EVENT_HOURS (:$EVENT_MINUTE)."
printf 'Daily card + email at %02d:%02d (default send time, to be fitted).\n' "$CARD_HOUR" "$CARD_MINUTE"
echo "Logs: $LOG_DIR/schedule.log (errors: schedule_error.log)"
echo "Credentials must live in /etc/engine.env or $LOG_DIR/engine.env."
echo "Test now: launchctl kickstart -k gui/\$(id -u)/com.payoffpitch.nba.watch"
