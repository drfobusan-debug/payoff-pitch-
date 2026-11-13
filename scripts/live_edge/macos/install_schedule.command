#!/bin/bash
# Install the live-total paper tests on macOS (launchd), one agent per sport. Local clock,
# Eastern assumed. Each runs `live_total.command <sport>`: it waits for the first start, runs
# until the slate is final and emails that sport's report on its own (GMAIL_* in engine.env).
#   com.payoffpitch.live.cbb -> Sat 10:00            (LIVE_CBB_WEEKDAYS, LIVE_CBB_HOUR)
#   com.payoffpitch.live.nba -> daily 11:00          (LIVE_NBA_WEEKDAYS, LIVE_NBA_HOUR)
#   com.payoffpitch.live.nhl -> daily 11:00          (LIVE_NHL_WEEKDAYS, LIVE_NHL_HOUR)
#   com.payoffpitch.live.cfb -> Tue-Sat 11:00        (LIVE_CFB_WEEKDAYS, LIVE_CFB_HOUR)
#   com.payoffpitch.live.nfl -> Sun, Mon, Thu 12:00  (LIVE_NFL_WEEKDAYS, LIVE_NFL_HOUR)
# Weekdays are launchd's: 0 = Sunday ... 6 = Saturday. LIVE_SPORTS limits which agents install.
# A run polls the Odds API only within 45 minutes of a start and at the half (end of P2 in
# the NHL), so a day with no games costs nothing.
# Idempotent. Remove one with: launchctl unload ~/Library/LaunchAgents/com.payoffpitch.live.<sport>.plist
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

install_agent() {
    local sport="$1" days="$2" hour="$3" label plist cal="" d
    label="com.payoffpitch.live.$sport"
    plist="$LAUNCH_AGENTS/$label.plist"
    for d in $days; do
        cal="$cal        <dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>$hour</integer><key>Minute</key><integer>0</integer></dict>
"
    done
    cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$label</string>
    <key>ProgramArguments</key><array><string>$REPO/scripts/live_edge/macos/live_total.command</string><string>$sport</string></array>
    <key>StartCalendarInterval</key>
    <array>
$cal    </array>
    <key>WorkingDirectory</key><string>$REPO</string>
    <key>StandardOutPath</key><string>$LOG_DIR/schedule_$sport.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/schedule_${sport}_error.log</string>
</dict>
</plist>
PLIST
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load "$plist"
    echo "Installed $label -> $plist"
}

for sport in ${LIVE_SPORTS:-cbb nba nhl cfb nfl}; do
    case "$sport" in
        cbb) install_agent cbb "${LIVE_CBB_WEEKDAYS:-6}" "${LIVE_CBB_HOUR:-10}" ;;
        nba) install_agent nba "${LIVE_NBA_WEEKDAYS:-0 1 2 3 4 5 6}" "${LIVE_NBA_HOUR:-11}" ;;
        nhl) install_agent nhl "${LIVE_NHL_WEEKDAYS:-0 1 2 3 4 5 6}" "${LIVE_NHL_HOUR:-11}" ;;
        cfb) install_agent cfb "${LIVE_CFB_WEEKDAYS:-2 3 4 5 6}" "${LIVE_CFB_HOUR:-11}" ;;
        nfl) install_agent nfl "${LIVE_NFL_WEEKDAYS:-0 1 4}" "${LIVE_NFL_HOUR:-12}" ;;
        *) echo "unknown sport in LIVE_SPORTS: $sport" >&2; exit 2 ;;
    esac
done
echo "Logs: $LOG_DIR/schedule_<sport>.log. Report by hand: python -m live_edge.stoppage report --sport <sport>"
