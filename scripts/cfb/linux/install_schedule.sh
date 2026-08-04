#!/bin/bash
# Install the hands-off daily college-football schedule on Linux (cron).
# Adds three cron jobs that run by themselves every day:
#   * 09:00              price + email today's card   (cfb-engine run)
#   * 11/15/19/23:00     CLV closing-line snapshots   (cfb-engine close)
#   * 03:00              grade yesterday + email recap (cfb-engine audit)
#
# Times are local to the machine's crontab. Override with CFB_RUN_HOUR /
# CFB_AUDIT_HOUR before running. Re-running is idempotent: the previous CFB
# block (tagged with the marker below) is replaced, not duplicated.
set -e
cd "$(dirname "$0")/../../.." || exit 1
REPO="$(pwd)"
AUTORUN="$REPO/scripts/cfb/linux/autorun.sh"
LOG="$HOME/.cfb_engine/schedule.log"
MARKER="# payoff-pitch-cfb-schedule"

RUN_HOUR="${CFB_RUN_HOUR:-9}"
AUDIT_HOUR="${CFB_AUDIT_HOUR:-3}"

chmod +x "$AUTORUN"
mkdir -p "$HOME/.cfb_engine"

# Seed the private credentials file the scheduled jobs read (no-op if present).
bash "$REPO/scripts/cfb/ensure_env.sh"

# Keep every existing crontab line except our previous CFB block.
existing="$(crontab -l 2>/dev/null | grep -vF "$MARKER" || true)"

{
    [ -n "$existing" ] && printf '%s\n' "$existing"
    echo "0 $RUN_HOUR * * * $AUTORUN run >> $LOG 2>&1 $MARKER"
    echo "0 11,15,19,23 * * * $AUTORUN close >> $LOG 2>&1 $MARKER"
    echo "0 $AUDIT_HOUR * * * $AUTORUN audit >> $LOG 2>&1 $MARKER"
} | crontab -

echo "Done. Installed the daily CFB cron jobs (run/close/audit)."
echo "Verify with: crontab -l | grep cfb"
echo "Logs: $LOG"
echo "Credentials must live in /etc/engine.env or $HOME/.cfb_engine/engine.env"
echo "(cron does not read your shell profile)."
echo "Remove with: crontab -l | grep -vF '$MARKER' | crontab -"
