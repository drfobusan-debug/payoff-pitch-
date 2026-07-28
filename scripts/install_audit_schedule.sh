#!/bin/zsh
# Install the daily 01:30 audit launchd job on macOS.
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="com.payoffpitch.audit.plist"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

mkdir -p "$LAUNCH_AGENTS"
cp "$REPO_DIR/scripts/$PLIST" "$LAUNCH_AGENTS/$PLIST"

# Unload first (idempotent) then load.
launchctl unload "$LAUNCH_AGENTS/$PLIST" 2>/dev/null || true
launchctl load "$LAUNCH_AGENTS/$PLIST"

echo "Installed and loaded $PLIST. Daily audit will run at 01:30."
