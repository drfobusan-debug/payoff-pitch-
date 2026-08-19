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
#   * Afternoon: wake 12:45 -> close 12:50 (snapshot the DAY games' close)
#   * Evening:   wake 18:35 -> close 18:40 (snapshot the NIGHT games' close)
#
# The MORNING run (before noon) is the real job and does both, in order:
#   1) FULL PACKAGE -> today's slate priced (mlb-engine run), then the slate
#                           preview article + audio, the pitcher/batter
#                           regression articles + audio and the morning power
#                           screen are generated and emailed as ONE message
#                           (Excel bet sheet + all PDFs/MP3s).
#                           By mid-morning the VSIN public handle/bets splits have
#                           posted, so the picks use them -- that's why the slate
#                           runs in the morning. This mirrors the one-click
#                           PAYOFF PITCH.command shortcut, minus opening Excel.
#   2) mlb-engine opta   -> capture the outside benchmark, yesterday's graded
#                           calls then today's. VSIN's day offset clamps at
#                           yesterday, so a slate not captured within a day is
#                           gone for good -- it has to be a daemon step.
#   3) mlb-engine audit  -> grade YESTERDAY's finished games, write the report,
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
REPO_DIR="/Users/jong/payoff-pitch-"          # NOT under ~/Desktop: macOS TCC blocks
                                              # background daemons from reading Desktop
                                              # (venv activate -> "Operation not permitted")
VENV_DIR="$REPO_DIR/.venv"

# >>> Real engine commands (already baked in -- nothing to edit) <<<
# The morning job builds today's full package (slate + regression articles,
# emailed as one message via scripts.email_daily_package), then grades yesterday
# and emails the ledger/report. The package steps are inlined in the runner
# below so it stays a single source of truth with the one-click shortcut.
AUDIT_CMD="mlb-engine audit --report --email"  # grade yesterday; defaults to yesterday

# Schedule (24h). Change if you like.
NIGHT_WAKE="23:00:00"; NIGHT_RUN_HOUR=23; NIGHT_RUN_MIN=5
FORCE_SLEEP="03:00:00"
MORNING_WAKE_HHMM="10:00"; MORNING_RUN_HOUR=10; MORNING_RUN_MIN=5
# Closing-line snapshots (for CLV scoring). Two of them, because a game that has
# started is gone from the pre-match board: the evening capture alone would only
# ever price the night slate. Captures merge, latest price per selection winning,
# so the afternoon games keep the close taken while they were still quoted.
CLOSE_WAKE_HHMM="18:35"; CLOSE_RUN_HOUR=18; CLOSE_RUN_MIN=40
DAY_CLOSE_WAKE_HHMM="12:45"; DAY_CLOSE_RUN_HOUR=12; DAY_CLOSE_RUN_MIN=50

WAKE_DAYS="MTWRFSU"   # M T W R F S U = Mon..Sun
# ==================================================================

RUNNER="/usr/local/bin/run_engine.sh"
SUDOERS="/etc/sudoers.d/payoffpitch-pmset"
ENV_FILE="/etc/engine.env"
LOG_OUT="/var/log/engine.out.log"
LOG_ERR="/var/log/engine.err.log"
NIGHT_LABEL="com.franz.engine.night"
MORNING_LABEL="com.franz.engine.morning"
CLOSE_LABEL="com.franz.engine.close"
DAY_CLOSE_LABEL="com.franz.engine.dayclose"
NIGHT_PLIST="/Library/LaunchDaemons/${NIGHT_LABEL}.plist"
MORNING_PLIST="/Library/LaunchDaemons/${MORNING_LABEL}.plist"
CLOSE_PLIST="/Library/LaunchDaemons/${CLOSE_LABEL}.plist"
DAY_CLOSE_PLIST="/Library/LaunchDaemons/${DAY_CLOSE_LABEL}.plist"

if [[ "${1:-}" == "--uninstall" ]]; then
  echo "Uninstalling..."
  rm -f "$SUDOERS"
  for p in "$NIGHT_PLIST" "$MORNING_PLIST" "$CLOSE_PLIST" "$DAY_CLOSE_PLIST"; do
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

# --- 1. secrets file ----------------------------------------------
# Owned by $RUN_AS_USER (mode 600): the daemons run as that user and source
# this file, so root ownership would make it unreadable and the job would exit 1.
if [[ ! -f "$ENV_FILE" ]]; then
  printf '# private (chmod 600). e.g.:\n# ODDS_API_KEY=your_key_here\n# TEAMRANKINGS_EMAIL=you@example.com\n# TEAMRANKINGS_PASSWORD=your_password\n' > "$ENV_FILE"
  echo "Created $ENV_FILE -- add your ODDS_API_KEY line."
fi
chmod 600 "$ENV_FILE"; chown "$RUN_AS_USER" "$ENV_FILE"

# --- 1b. let the run user schedule wakes ---------------------------
# Every job re-arms the next machine wake through pmset, which is root-only:
# as $RUN_AS_USER it fails with "must be run as root", nothing re-arms, and the
# Mac stops waking for the morning card. Grant NOPASSWD on that one binary.
cat > "$SUDOERS" <<SUDO
# Installed by setup_engine_autorun.sh: the engine daemons run as $RUN_AS_USER
# and need pmset to schedule the next wake.
$RUN_AS_USER ALL=(root) NOPASSWD: /usr/bin/pmset
SUDO
chmod 440 "$SUDOERS"
visudo -cf "$SUDOERS" >/dev/null || { echo "bad sudoers drop-in; removing" >&2; rm -f "$SUDOERS"; }

# --- 2. runner: runs engine; if invoked as 'night', arm next 10:00 wake
cat > "$RUNNER" <<EOF
#!/bin/bash
set -euo pipefail
MODE="\${1:-run}"
# The daemons run as $RUN_AS_USER so the engine can read its own caches, but
# pmset is root-only -- unprivileged calls fail with "must be run as root" and
# the machine then never wakes for the next job. Escalate just this binary (see
# the sudoers drop-in this installer writes).
pmset_() {
  if [[ \$EUID -eq 0 ]]; then /usr/bin/pmset "\$@"; else sudo -n /usr/bin/pmset "\$@"; fi
}
[[ -f "$ENV_FILE" ]] && set -a && source "$ENV_FILE" && set +a
# Also read the user-level env file (same one the manual shortcut uses), so
# Gmail creds placed there work for the autorun too.
[[ -f "\$HOME/.mlb_engine/engine.env" ]] && set -a && source "\$HOME/.mlb_engine/engine.env" && set +a
cd "$REPO_DIR"
[[ -d "$VENV_DIR" ]] && source "$VENV_DIR/bin/activate"

# CA bundle: a fresh macOS venv has no root certificates, so the stdlib ssl
# used by the SMTP email step (and requests) fails with
# CERTIFICATE_VERIFY_FAILED. Point OpenSSL + requests at certifi's bundle.
_CERTS="\$(python -m certifi 2>/dev/null || true)"
if [[ -n "\$_CERTS" ]]; then
  export SSL_CERT_FILE="\$_CERTS"
  export REQUESTS_CA_BUNDLE="\$_CERTS"
fi

# WeasyPrint (PDF export) loads pango/cairo/gdk-pixbuf via ctypes; on macOS
# those live in the Homebrew lib dir, which is NOT on the default dyld search
# path. Point at it so the slate-preview + audit PDFs render.
for _brew_lib in /opt/homebrew/lib /usr/local/lib; do
  [[ -d "\$_brew_lib" ]] && export DYLD_FALLBACK_LIBRARY_PATH="\$_brew_lib:\${DYLD_FALLBACK_LIBRARY_PATH:-}"
done

if [[ "\$MODE" == "night" ]]; then
  # Keep-awake only: re-arm a one-shot wake for the NEXT morning at
  # $MORNING_WAKE_HHMM so the machine wakes after the 03:00 sleep. No engine work.
  NEXT=\$(date -v+1d +"%m/%d/%Y")
  pmset_ schedule wake "\$NEXT $MORNING_WAKE_HHMM:00" || echo "[\$(date)] could not arm morning wake" >&2
  echo "[\$(date)] armed morning wake for \$NEXT $MORNING_WAKE_HHMM:00"
elif [[ "\$MODE" == "close" ]]; then
  # Snapshot today's CLOSING market so tomorrow morning's audit can score closing
  # line value (CLV) -- the fast way to tell whether a pick had real edge. Runs
  # twice a day (afternoon + evening) and merges, because a started game leaves
  # the pre-match board. Cheap (~3 credits + one per event for props).
  mlb-engine close || echo "[\$(date)] 'mlb-engine close' exited non-zero" >&2
  # Re-arm tonight's evening capture in case the Mac would sleep before it.
  pmset_ schedule wake "\$(date +%m/%d/%Y) $CLOSE_WAKE_HHMM:00" || echo "[\$(date)] could not arm evening wake" >&2
else
  # THE daily job (before noon): build today's FULL package (slate + pitcher/
  # batter regression articles + audio) and email it as one message, then grade
  # yesterday and email the ledger/report.
  #
  # Keep the Mac awake WITHOUT wrapping the engine in caffeinate: caffeinate is
  # a SIP-protected /usr/bin binary, and launching it strips DYLD_* from the
  # environment -- so WeasyPrint, run as caffeinate's child, could not find the
  # Homebrew libs. Instead hold a background caffeinate tied to THIS script's
  # PID and run the engine as a direct child, so DYLD_*/SSL_CERT_FILE survive.
  /usr/bin/caffeinate -i -w \$\$ &

  # Arm a one-shot wake for TONIGHT's closing snapshot (same calendar day), so
  # the close daemon can fire even if the Mac would otherwise sleep by evening.
  pmset_ schedule wake "\$(date +%m/%d/%Y) $DAY_CLOSE_WAKE_HHMM:00" || echo "[\$(date)] could not arm afternoon wake" >&2

  # 1) price today's slate (writes Excel + previews/predictions JSON + Statcast
  #    cache pkl). No --email here: the package email below owns delivery.
  VSIN="\$HOME/.mlb_engine/vsin_today.csv"
  if [[ -f "\$VSIN" ]]; then
    mlb-engine run --vsin-csv "\$VSIN" || echo "[\$(date)] 'mlb-engine run' exited non-zero" >&2
  else
    mlb-engine run || echo "[\$(date)] 'mlb-engine run' exited non-zero" >&2
  fi

  # 2) slate + regression articles, then email the whole package as one message.
  # Today's card only. "newest on disk" would silently email yesterday's slate
  # as today's on any morning the run failed, which is exactly when it matters.
  OUT="\$HOME/.mlb_engine/output"
  today=\$(date +%Y-%m-%d)
  xlsx="\$OUT/mlb_recommendations_\$today.xlsx"
  if [[ -f "\$xlsx" ]]; then
    day="\$today"
    python -m scripts.regen_slate "\$day" || echo "[\$(date)] slate article failed" >&2
    pkl=\$(ls -t "\$HOME/.mlb_engine/cache/"statcast_*.pkl 2>/dev/null | head -1) || true
    if [[ -n "\$pkl" ]]; then
      python -m scripts.regen_regression "\$day" "\$(basename "\$pkl")" || echo "[\$(date)] regression articles failed" >&2
    else
      echo "[\$(date)] no Statcast cache pkl; skipping regression articles" >&2
    fi
    # The power screen reads the slate that was just priced, so it runs after it
    # and before delivery: its PDF rides in the package email instead of sending
    # a second one. Not fatal -- a slate with no soft arm on it screens nobody.
    python scripts/power_screen.py --date "\$day" \\
      || echo "[\$(date)] power screen failed" >&2
    python -m scripts.email_daily_package "\$day" || echo "[\$(date)] package email failed" >&2
  else
    echo "[\$(date)] no workbook for \$today; skipping package email" >&2
  fi

  # 3) capture the Opta benchmark, both halves of it. VSIN's day offset clamps
  #    at yesterday, so a slate missed by a day is gone permanently -- which is
  #    why this is a daemon step and not something to remember to run. day=-1
  #    brings back last night's graded outcomes; day=0 takes today's calls
  #    before the games start. Free, no credits, and a failure is not fatal.
  mlb-engine opta --day -1 || echo "[\$(date)] 'mlb-engine opta --day -1' exited non-zero" >&2
  mlb-engine opta || echo "[\$(date)] 'mlb-engine opta' exited non-zero" >&2

  # 4) grade yesterday + email the ledger/report.
  $AUDIT_CMD || echo "[\$(date)] '$AUDIT_CMD' exited non-zero" >&2
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
write_plist "$CLOSE_PLIST"   "$CLOSE_LABEL"   "$CLOSE_RUN_HOUR"   "$CLOSE_RUN_MIN"   close
write_plist "$DAY_CLOSE_PLIST" "$DAY_CLOSE_LABEL" "$DAY_CLOSE_RUN_HOUR" "$DAY_CLOSE_RUN_MIN" close
echo "Wrote daemons: $NIGHT_PLIST , $MORNING_PLIST , $CLOSE_PLIST , $DAY_CLOSE_PLIST"

# The daemons run as $RUN_AS_USER, but /var/log is root-owned -- launchd can't
# create the StandardOut/Err files there and the job dies with EX_CONFIG (78).
# Pre-create them owned by the run user so logging (and the job) works.
touch "$LOG_OUT" "$LOG_ERR"
chown "$RUN_AS_USER" "$LOG_OUT" "$LOG_ERR"
chmod 644 "$LOG_OUT" "$LOG_ERR"

for p in "$NIGHT_PLIST" "$MORNING_PLIST" "$CLOSE_PLIST" "$DAY_CLOSE_PLIST"; do
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
echo "Test close run:    sudo launchctl start ${CLOSE_LABEL}"
echo "Test day close:    sudo launchctl start ${DAY_CLOSE_LABEL}"
echo "Logs:              tail -f ${LOG_OUT} ${LOG_ERR}"
echo
echo "Reminders: keep it PLUGGED IN; use SLEEP (not Shut Down);"
echo "add ODDS_API_KEY to ${ENV_FILE}, and TEAMRANKINGS_EMAIL/PASSWORD for"
echo "tonight's outside picks (signed out that grid only publishes played slates)."
