#!/bin/zsh
# Install the daily 11:30 AM ET predictions + email launchd job on macOS.
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$REPO_DIR/scripts/com.payoffpitch.predictions.plist"
PLIST="com.payoffpitch.predictions.plist"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
PLIST_DST="$LAUNCH_AGENTS/$PLIST"

mkdir -p "$LAUNCH_AGENTS"

# Inject the absolute repo path into the plist.
sed "s|REPO_DIR|$REPO_DIR|g" "$PLIST_SRC" > "$PLIST_DST"

chmod +x "$REPO_DIR/scripts/run_card_email.sh"

# Unload first (idempotent) then load.
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "Installed and loaded $PLIST. Daily predictions + email card will run at 11:30 AM ET."
