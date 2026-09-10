#!/bin/bash
#
# setup_engine_autorun.sh   (Apple Silicon / M1 Pro edition)
# ------------------------------------------------------------------
# Runs the payoff-pitch- engine automatically, even at the LOGIN SCREEN
# (LaunchDaemons run as root at boot/wake, no login):
#
#   * Night:   wake 23:00           (keep-awake only -- no engine work)
#   * Forced sleep at 03:00
#   * Morning: wake 10:00  ->  run 10:05  (grade yesterday, capture the Opta
#                           benchmark, email the ledger/report; prices nothing)
#   * Day:     wake 11:30  ->  run 11:35  (price the DAY games, inside three
#                           hours of their first pitch, and email the day
#                           slate PDF + bet card)
#   * Night:   wake 18:30  ->  run 18:35  (the same for the NIGHT games)
#                           These two passes are where the bets are placed.
#   * Afternoon: wake 12:45 -> close 12:50 (snapshot the DAY games' close)
#   * Evening:   wake 18:35 -> close 18:40 (snapshot the NIGHT games' close)
#   * Saturday:  wake 09:00  ->  cfb 09:05  (price the college football board
#                           and email its article PDF + MP3 + workbook). The
#                           Friday night wake arms this one.
#   * Thu + Sun: wake 08:00  ->  nfl 08:05  (archive the NFL board, price the
#                           week, re-stamp the close, grade what has finished,
#                           email the card PDF + workbook). Thursday's pass is
#                           the week's card and grades last Sunday/Monday;
#                           Sunday's re-stamps the close before the early games
#                           and grades Thursday night. The Wed/Sat night wakes
#                           arm these.
#
# Why the bets are priced in two SLATE passes and not in the morning: over the
# ledger's 915 graded buys carrying a first-pitch stamp, the ones priced inside
# three hours of the game returned +5.7% (n=369) against -14.9% (n=546) for the
# ones priced earlier, and the sign repeats in every market with a sample. A
# 10:05 card prices most of a slate six or more hours out, on projected lineups,
# and the engine's clock gate (MLBE_LINEUP_CLOCK_GATE) refuses every buy priced
# that early. So the morning run prices nothing, and the SLATE passes -- one
# every SLATE_WINDOW_HOURS from 11:55 -- each price ONLY the games starting
# inside the next SLATE_WINDOW_HOURS (mlb-engine run --within-hours), off the
# posted lineups, on the board as it stands, then email that block's slate
# article PDF + MP3 and the bet card. The windows tile every MLB first pitch
# (12:05 through 22:10 local), so each game is bought by exactly one pass and
# none is refused on the clock for want of a pass. Each pass folds its games
# into the card the earlier passes wrote, so the audit still grades one card
# per slate. A pass whose window holds no game writes and emails nothing. The
# once-a-day pieces (regression articles, power screen) are written by the
# first pass that has games and ride only with that pass's email.
#
# The block names and times live in mlb_engine/slate_blocks.py (the article
# and the email read them from there); SLATE_RUNS below must match it.
#
# The MORNING run (before noon) is the bookkeeping, in order:
#   0) git pull --ff-only  -> run what has been MERGED, not whatever was on
#                           disk when the Mac was last touched.
#                           Skipped on a dirty, diverged or non-main checkout,
#                           and never fatal: see the runner for the guards.
#                           The runner body itself lives in the repo
#                           (scripts/macos/run_engine.sh), so that pull also
#                           updates what each pass writes and emails.
#   1) (no pricing)      -> the slate passes above own the card and the email.
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
# The slate passes price a block of games and email it (slate article + bet card
# via scripts.email_daily_package --block); the morning job grades yesterday and
# emails the ledger/report. The package steps are in scripts/macos/run_engine.sh so
# it stays a single source of truth with the one-click shortcut.
AUDIT_CMD="mlb-engine audit --report --email"  # grade yesterday; defaults to yesterday

# Schedule (24h). Change if you like.
NIGHT_WAKE="23:00:00"; NIGHT_RUN_HOUR=23; NIGHT_RUN_MIN=5
FORCE_SLEEP="03:00:00"
MORNING_WAKE_HHMM="10:00"; MORNING_RUN_HOUR=10; MORNING_RUN_MIN=5
# Closing-line snapshots (for CLV scoring). Two of them, because a game that has
# started is gone from the pre-match board: the evening capture alone would only
# ever price the night slate. Captures merge, latest price per selection winning,
# so the afternoon games keep the close taken while they were still quoted.
CLOSE_WAKE_HHMM="18:35"; DAY_CLOSE_WAKE_HHMM="12:45"
CLOSE_RUN_HOUR=18; CLOSE_RUN_MIN=40
DAY_CLOSE_RUN_HOUR=12; DAY_CLOSE_RUN_MIN=50
# The slate passes: each prices the games starting inside the next
# SLATE_WINDOW_HOURS and emails that block's slate PDF + bet card. The window
# is the clock gate's number (MLBE_LINEUP_STALE_HOURS) seen from the other
# side, so a game is bought inside three hours of first pitch or not at all,
# and the passes are one window apart so every first pitch falls in exactly
# one. Five minutes before the hour so a game starting on the hour is still
# ahead of the pass that owns it. Entries are "block=HH:MM", matching
# mlb_engine/slate_blocks.py.
SLATE_WINDOW_HOURS=3
SLATE_RUNS="matinee=11:55 afternoon=14:55 evening=17:55 late=20:55"

# College football: one pass on Saturday morning, before the MLB job so the two
# never share a machine hour. `cfb-engine run` prices the calendar day it runs
# on, so this is the Saturday slate only; Thursday/Friday cards stay one-click
# (scripts/macos/run_cfb_week.command --date ...).
CFB_WAKE_HHMM="09:00"; CFB_RUN_HOUR=9; CFB_RUN_MIN=5
CFB_WEEKDAY=6   # launchd: 0=Sunday .. 6=Saturday

# NFL: Thursday (the week's card) and Sunday (close re-stamp + Thursday grade),
# an hour before anything else so no two engines share a machine hour. Both are
# the same `nfl-engine job`, which appends only new positions, keeps the price of
# record and re-stamps the close until kickoff -- so re-running is safe.
NFL_WAKE_HHMM="08:00"; NFL_RUN_HOUR=8; NFL_RUN_MIN=5
NFL_WEEKDAYS="4 0"   # launchd: Thursday, Sunday

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
CFB_LABEL="com.franz.engine.cfb"
NFL_LABELS=(); NFL_PLISTS=()
for wd in $NFL_WEEKDAYS; do
  label="com.franz.engine.nfl${wd}"
  NFL_LABELS+=("$label")
  NFL_PLISTS+=("/Library/LaunchDaemons/${label}.plist")
done
NIGHT_PLIST="/Library/LaunchDaemons/${NIGHT_LABEL}.plist"
MORNING_PLIST="/Library/LaunchDaemons/${MORNING_LABEL}.plist"
CLOSE_PLIST="/Library/LaunchDaemons/${CLOSE_LABEL}.plist"
DAY_CLOSE_PLIST="/Library/LaunchDaemons/${DAY_CLOSE_LABEL}.plist"
CFB_PLIST="/Library/LaunchDaemons/${CFB_LABEL}.plist"

# One daemon per slate pass, labelled by its block so a single pass can be
# started, read in the log or removed on its own.
SLATE_LABELS=(); SLATE_PLISTS=(); SLATE_WAKES=""
for spec in $SLATE_RUNS; do
  label="com.franz.engine.slate${spec%%=*}"
  SLATE_LABELS+=("$label")
  SLATE_PLISTS+=("/Library/LaunchDaemons/${label}.plist")
  hm="${spec##*=}"
  # Wake five minutes ahead of the run so launchd finds the Mac up.
  SLATE_WAKES="$SLATE_WAKES $(printf '%02d:%02d' "$((10#${hm%%:*}))" "$((10#${hm##*:} - 5))")"
done
ALL_PLISTS=("$NIGHT_PLIST" "$MORNING_PLIST" "$CLOSE_PLIST" "$DAY_CLOSE_PLIST" "$CFB_PLIST" "${NFL_PLISTS[@]}" "${SLATE_PLISTS[@]}")

if [[ "${1:-}" == "--uninstall" ]]; then
  echo "Uninstalling..."
  rm -f "$SUDOERS"
  for p in "${ALL_PLISTS[@]}"; do
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

# --- 2. runner: a shim over the versioned scripts/macos/run_engine.sh
# The daemon steps live in the repo, so a merged change to what a pass writes
# or emails reaches the Mac on the runner's own `git pull` -- nobody has to
# re-run this installer. Only the schedule and paths are baked in here. The
# repo copy is duplicated to a private temp file before exec, because bash
# reads a script incrementally and the pull inside the run may rewrite it.
cat > "$RUNNER" <<EOF
#!/bin/bash
set -euo pipefail
export RUN_AS_USER="$RUN_AS_USER" REPO_DIR="$REPO_DIR" VENV_DIR="$VENV_DIR"
export ENV_FILE="$ENV_FILE" AUDIT_CMD="$AUDIT_CMD"
export MORNING_WAKE_HHMM="$MORNING_WAKE_HHMM" CLOSE_WAKE_HHMM="$CLOSE_WAKE_HHMM"
export DAY_CLOSE_WAKE_HHMM="$DAY_CLOSE_WAKE_HHMM" CFB_WAKE_HHMM="$CFB_WAKE_HHMM"
export NFL_WAKE_HHMM="$NFL_WAKE_HHMM" SLATE_WINDOW_HOURS="$SLATE_WINDOW_HOURS"
export SLATE_WAKES="$SLATE_WAKES"
SRC="$REPO_DIR/scripts/macos/run_engine.sh"
[[ -f "\$SRC" ]] || { echo "[\$(date)] runner missing: \$SRC (pull main)" >&2; exit 1; }
TMP="\$(mktemp -t run_engine)"
cp "\$SRC" "\$TMP"
trap 'rm -f "\$TMP"' EXIT
bash "\$TMP" "\$@"
EOF
chmod 755 "$RUNNER"
echo "Wrote runner: $RUNNER"

# --- 3. LaunchDaemons ---------------------------------------------
write_plist() {  # <path> <label> <hour> <min> <mode> [weekday 0-6]
  local weekday=""
  [[ -n "${6:-}" ]] && weekday="<key>Weekday</key><integer>$6</integer>"
  cat > "$1" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$2</string>
  <key>UserName</key><string>${RUN_AS_USER}</string>
  <key>ProgramArguments</key>
  <array><string>${RUNNER}</string><string>$5</string></array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>$3</integer><key>Minute</key><integer>$4</integer>${weekday}</dict>
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
write_plist "$CFB_PLIST" "$CFB_LABEL" "$CFB_RUN_HOUR" "$CFB_RUN_MIN" cfb "$CFB_WEEKDAY"
i=0
for wd in $NFL_WEEKDAYS; do
  write_plist "${NFL_PLISTS[$i]}" "${NFL_LABELS[$i]}" "$NFL_RUN_HOUR" "$NFL_RUN_MIN" nfl "$wd"
  i=$((i + 1))
done
i=0
for spec in $SLATE_RUNS; do
  hm="${spec##*=}"
  write_plist "${SLATE_PLISTS[$i]}" "${SLATE_LABELS[$i]}" \
    "$((10#${hm%%:*}))" "$((10#${hm##*:}))" "slate-${spec%%=*}"
  i=$((i + 1))
done
echo "Wrote daemons: ${ALL_PLISTS[*]}"

# The daemons run as $RUN_AS_USER, but /var/log is root-owned -- launchd can't
# create the StandardOut/Err files there and the job dies with EX_CONFIG (78).
# Pre-create them owned by the run user so logging (and the job) works.
touch "$LOG_OUT" "$LOG_ERR"
chown "$RUN_AS_USER" "$LOG_OUT" "$LOG_ERR"
chmod 644 "$LOG_OUT" "$LOG_ERR"

for p in "${ALL_PLISTS[@]}"; do
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
echo "Test CFB Saturday: sudo launchctl start ${CFB_LABEL}   (prices today's board,"
echo "                   spends Odds API credits and emails the slate)"
echo "Test NFL pass:     sudo launchctl start ${NFL_LABELS[0]}   (prices the week,"
echo "                   spends Odds API credits and emails the card; Sunday's is ${NFL_LABELS[1]})"
echo "Test a slate pass: sudo launchctl start ${SLATE_LABELS[0]}   (prices the games"
echo "                   inside ${SLATE_WINDOW_HOURS}h, spends Odds API credits and emails that"
echo "                   block's slate PDF + card; the others: ${SLATE_LABELS[*]:1})"
echo "Logs:              tail -f ${LOG_OUT} ${LOG_ERR}"
echo
echo "Reminders: keep it PLUGGED IN; use SLEEP (not Shut Down);"
echo "add ODDS_API_KEY to ${ENV_FILE}, and TEAMRANKINGS_EMAIL/PASSWORD for"
echo "tonight's outside picks (signed out that grid only publishes played slates)."
