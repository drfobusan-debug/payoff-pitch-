#!/bin/bash
#
# setup_engine_autorun.sh   (Apple Silicon / M1 Pro edition)
# ------------------------------------------------------------------
# Runs the payoff-pitch- engine automatically, even at the LOGIN SCREEN
# (LaunchDaemons run as root at boot/wake, no login):
#
#   * Night:   wake 23:00           (keep-awake only -- no engine work)
#   * Forced sleep at 03:00
#   * Morning: wake 10:00  ->  run 10:05  (before noon: THE daily job)
#
# The MORNING run (before noon) is the real job and does both, in order:
#   1) mlb-engine run --email -> today's slate. By mid-morning the VSIN public
#                           handle/bets splits have posted, so the picks use
#                           them -- that's why the slate runs in the morning.
#                           --email sends the Morningstar slate preview
#                           (PDF + audio) with the Excel bet sheet.
#   2) mlb-engine audit  -> grade YESTERDAY's finished games, write the report,
#                           and EMAIL the Excel ledger + article + audio.
#
# The night wake exists ONLY to keep the Mac up for the morning run: it re-arms
# the next 10:00 one-shot wake (pmset allows only ONE recurring wake), so the
# machine reliably wakes each morning while you're away. It runs no engine work.
#
# Run once:   sudo bash setup_engine_autorun.sh
# Uninstall:  sudo bash setup_engine_autorun.sh --uninstall
# ------------------------------------------------------------------
set -euo pipefail

# ================== EDIT THESE ====================================
RUN_AS_USER="jong"
REPO_DIR="/Users/jong/payoff-pitch-"          # confirm this path (see note in chat)
VENV_DIR="$REPO_DIR/.venv"

# >>> Real engine commands (already baked in -- nothing to edit) <<<
# The morning job runs these two in order: generate today's slate (with the
# morning handle/bets splits), then grade yesterday and email the 3 artifacts.
RUN_CMD="mlb-engine run --email"               # today's slate + email preview; defaults to today
AUDIT_CMD="mlb-engine audit --report --email"  # grade yesterday; defaults to yesterday

# Schedule (24h). Change if you like.
NIGHT_WAKE="23:00:00"; NIGHT_RUN_HOUR=23; NIGHT_RUN_MIN=5
FORCE_SLEEP="03:00:00"
MORNING_WAKE_HHMM="10:00"; MORNING_RUN_HOUR=10; MORNING_RUN_MIN=5

WAKE_DAYS="MTWRFSU"   # M T W R F S U = Mon..Sun
# ==================================================================

RUNNER="/usr/local/bin/run_engine.sh"
ENV_FILE="/etc/engine.env"
LOG_OUT="/var/log/engine.out.log"
LOG_ERR="/var/log/engine.err.log"
NIGHT_LABEL="com.franz.engine.night"
MORNING_LABEL="com.franz.engine.morning"
NIGHT_PLIST="/Library/LaunchDaemons/${NIGHT_LABEL}.plist"
MORNING_PLIST="/Library/LaunchDaemons/${MORNING_LABEL}.plist"

if [[ "${1:-}" == "--uninstall" ]]; then
  echo "Uninstalling..."
  for p in "$NIGHT_PLIST" "$MORNING_PLIST"; do
    launchctl bootout system "$p" 2>/dev/null || launchctl unload "$p" 2>/dev/null || true
    rm -f "$p"
  done
  rm -f "$RUNNER"
  pmset repeat cancel || true
  pmset schedule cancelall || true
  echo "Removed daemons, runner, and all wake/sleep schedules."
  echo "Left $ENV_FILE and logs in place."
  exit 0
fi

[[ $EUID -eq 0 ]] || { echo "Run with sudo:  sudo bash $0"; exit 1; }
command -v mlb-engine >/dev/null 2>&1 || \
  echo "NOTE: 'mlb-engine' not on PATH yet; it must be installed inside $VENV_DIR (pip install -e .)."
[[ -d "$REPO_DIR" ]] || { echo "REPO_DIR not found: $REPO_DIR (fix the path at top)."; exit 1; }
[[ -d "$VENV_DIR" ]] || echo "WARNING: venv not found at $VENV_DIR."

[[ "$(uname -m)" == "arm64" ]] && \
  echo "Apple Silicon confirmed: leave the Mac ASLEEP (not shut down), plugged in."

# --- 1. secrets file (root-only) ----------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
  printf '# root-only (chmod 600). e.g.:\n# ODDS_API_KEY=your_key_here\n' > "$ENV_FILE"
  chmod 600 "$ENV_FILE"; chown root:wheel "$ENV_FILE"
  echo "Created $ENV_FILE -- add your ODDS_API_KEY line."
fi

# --- 2. runner: runs engine; if invoked as 'night', arm next 10:00 wake
cat > "$RUNNER" <<EOF
#!/bin/bash
set -euo pipefail
MODE="\${1:-run}"
[[ -f "$ENV_FILE" ]] && set -a && source "$ENV_FILE" && set +a
cd "$REPO_DIR"
[[ -d "$VENV_DIR" ]] && source "$VENV_DIR/bin/activate"

# WeasyPrint (PDF export) loads pango/cairo/gdk-pixbuf via ctypes; on macOS
# those live in the Homebrew lib dir, which is NOT on the default dyld search
# path. Point at it so the slate-preview + audit PDFs render.
for _brew_lib in /opt/homebrew/lib /usr/local/lib; do
  [[ -d "\$_brew_lib" ]] && export DYLD_FALLBACK_LIBRARY_PATH="\$_brew_lib:\${DYLD_FALLBACK_LIBRARY_PATH:-}"
done

# caffeinate -i blocks *idle* sleep during the run. (The 03:00 forced
# sleep from pmset is intentional and will still fire.)
if [[ "\$MODE" == "night" ]]; then
  # Keep-awake only: re-arm a one-shot wake for the NEXT morning at
  # $MORNING_WAKE_HHMM so the machine wakes after the 03:00 sleep. No engine work.
  NEXT=\$(date -v+1d +"%m/%d/%Y")
  /usr/bin/pmset schedule wake "\$NEXT $MORNING_WAKE_HHMM:00" || true
  echo "[\$(date)] armed morning wake for \$NEXT $MORNING_WAKE_HHMM:00"
else
  # THE daily job (before noon): today's slate, then grade yesterday + email.
  caffeinate -i $RUN_CMD   || echo "[\$(date)] '$RUN_CMD' exited non-zero" >&2
  caffeinate -i $AUDIT_CMD || echo "[\$(date)] '$AUDIT_CMD' exited non-zero" >&2
fi
EOF
chmod 755 "$RUNNER"
echo "Wrote runner: $RUNNER"

# --- 3. LaunchDaemons ---------------------------------------------
write_plist() {  # <path> <label> <hour> <min> <mode>
  cat > "$1" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$2</string>
  <key>UserName</key><string>${RUN_AS_USER}</string>
  <key>ProgramArguments</key>
  <array><string>${RUNNER}</string><string>$5</string></array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>$3</integer><key>Minute</key><integer>$4</integer></dict>
  <key>StandardOutPath</key><string>${LOG_OUT}</string>
  <key>StandardErrorPath</key><string>${LOG_ERR}</string>
</dict></plist>
EOF
  chown root:wheel "$1"; chmod 644 "$1"
}
write_plist "$NIGHT_PLIST"   "$NIGHT_LABEL"   "$NIGHT_RUN_HOUR"   "$NIGHT_RUN_MIN"   night
write_plist "$MORNING_PLIST" "$MORNING_LABEL" "$MORNING_RUN_HOUR" "$MORNING_RUN_MIN" morning
echo "Wrote daemons: $NIGHT_PLIST , $MORNING_PLIST"

# The daemons run as $RUN_AS_USER, but /var/log is root-owned -- launchd can't
# create the StandardOut/Err files there and the job dies with EX_CONFIG (78).
# Pre-create them owned by the run user so logging (and the job) works.
touch "$LOG_OUT" "$LOG_ERR"
chown "$RUN_AS_USER" "$LOG_OUT" "$LOG_ERR"
chmod 644 "$LOG_OUT" "$LOG_ERR"

for p in "$NIGHT_PLIST" "$MORNING_PLIST"; do
  launchctl bootout system "$p" 2>/dev/null || true
  launchctl bootstrap system "$p" 2>/dev/null || launchctl load "$p"
done
echo "Loaded daemons."

# --- 4. recurring wake (23:00) + forced sleep (03:00) -------------
pmset repeat wakeorpoweron "$WAKE_DAYS" "$NIGHT_WAKE" sleep "$WAKE_DAYS" "$FORCE_SLEEP"
echo "Recurring schedule set: wake $NIGHT_WAKE, sleep $FORCE_SLEEP ($WAKE_DAYS)."

# --- 5. arm the FIRST morning wake now (today or tomorrow) --------
NOW_HM=$(date +%H%M); TARGET_HM=${MORNING_WAKE_HHMM/:/}
if (( 10#$NOW_HM < 10#$TARGET_HM )); then WHEN=$(date +"%m/%d/%Y"); else WHEN=$(date -v+1d +"%m/%d/%Y"); fi
pmset schedule wake "$WHEN $MORNING_WAKE_HHMM:00" || true
echo "Armed first morning wake: $WHEN $MORNING_WAKE_HHMM:00"

echo
echo "==================== DONE ===================="
echo "Verify:            pmset -g sched"
echo "Test night run:    sudo launchctl start ${NIGHT_LABEL}"
echo "Test morning run:  sudo launchctl start ${MORNING_LABEL}"
echo "Logs:              tail -f ${LOG_OUT} ${LOG_ERR}"
echo
echo "Reminders: keep it PLUGGED IN; use SLEEP (not Shut Down);"
echo "add ODDS_API_KEY to ${ENV_FILE}."
