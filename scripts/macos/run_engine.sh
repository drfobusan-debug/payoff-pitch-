#!/bin/bash
#
# scripts/macos/run_engine.sh -- the engine daemons' runner, versioned.
#
# setup_engine_autorun.sh installs /usr/local/bin/run_engine.sh as a thin shim
# that exports the schedule below and execs a copy of THIS file, so a merged
# change to a daemon step (a new once-a-day artifact, a reordered pass) reaches
# the Mac on the next `git pull` the runner itself performs -- without anyone
# re-running the installer with sudo. The shim only has to be reinstalled when
# the schedule or the paths change.
#
# Every knob is an environment variable with the installer's default, so the
# script also runs by hand:  bash scripts/macos/run_engine.sh slate-evening
#
set -euo pipefail
MODE="${1:-run}"

RUN_AS_USER="${RUN_AS_USER:-jong}"
REPO_DIR="${REPO_DIR:-/Users/jong/payoff-pitch-}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
ENV_FILE="${ENV_FILE:-/etc/engine.env}"
AUDIT_CMD="${AUDIT_CMD:-mlb-engine audit --report --email}"
MORNING_WAKE_HHMM="${MORNING_WAKE_HHMM:-10:00}"
CLOSE_WAKE_HHMM="${CLOSE_WAKE_HHMM:-18:35}"
DAY_CLOSE_WAKE_HHMM="${DAY_CLOSE_WAKE_HHMM:-12:45}"
CFB_WAKE_HHMM="${CFB_WAKE_HHMM:-09:00}"
NFL_WAKE_HHMM="${NFL_WAKE_HHMM:-08:00}"
SLATE_WINDOW_HOURS="${SLATE_WINDOW_HOURS:-3}"
# Wake times five minutes ahead of each slate pass (matinee 11:55 .. late 20:55).
SLATE_WAKES="${SLATE_WAKES:- 11:50 14:50 17:50 20:50}"
# The daemons run as the login user so the engine can read its own caches, but
# pmset is root-only -- unprivileged calls fail with "must be run as root" and
# the machine then never wakes for the next job. Escalate just this binary (see
# the sudoers drop-in setup_engine_autorun.sh writes).
pmset_() {
  if [[ $EUID -eq 0 ]]; then /usr/bin/pmset "$@"; else sudo -n /usr/bin/pmset "$@"; fi
}
[[ -f "${ENV_FILE}" ]] && set -a && source "${ENV_FILE}" && set +a
# Also read the user-level env file (same one the manual shortcut uses), so
# Gmail creds placed there work for the autorun too.
[[ -f "$HOME/.mlb_engine/engine.env" ]] && set -a && source "$HOME/.mlb_engine/engine.env" && set +a
[[ -f "$HOME/.cfb_engine/engine.env" ]] && set -a && source "$HOME/.cfb_engine/engine.env" && set +a
[[ -f "$HOME/.nfl_engine/engine.env" ]] && set -a && source "$HOME/.nfl_engine/engine.env" && set +a
cd "${REPO_DIR}"
[[ -d "${VENV_DIR}" ]] && source "${VENV_DIR}/bin/activate"

# CA bundle: a fresh macOS venv has no root certificates, so the stdlib ssl
# used by the SMTP email step (and requests) fails with
# CERTIFICATE_VERIFY_FAILED. Point OpenSSL + requests at certifi's bundle.
_CERTS="$(python -m certifi 2>/dev/null || true)"
if [[ -n "$_CERTS" ]]; then
  export SSL_CERT_FILE="$_CERTS"
  export REQUESTS_CA_BUNDLE="$_CERTS"
fi

# The clock gate and the slate-pass window are the same number seen from two
# sides: the gate refuses a row priced more than MLBE_LINEUP_STALE_HOURS before
# first pitch, and a slate pass prices exactly the games inside
# SLATE_WINDOW_HOURS. Tied here so raising one cannot leave the other behind,
# refusing rows no pass ever comes back for.
export MLBE_LINEUP_STALE_HOURS="${MLBE_LINEUP_STALE_HOURS:-$SLATE_WINDOW_HOURS}"

# WeasyPrint (PDF export) loads pango/cairo/gdk-pixbuf via ctypes; on macOS
# those live in the Homebrew lib dir, which is NOT on the default dyld search
# path. Point at it so the slate-preview + audit PDFs render.
for _brew_lib in /opt/homebrew/lib /usr/local/lib; do
  [[ -d "$_brew_lib" ]] && export DYLD_FALLBACK_LIBRARY_PATH="$_brew_lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
done

# Take whatever has been merged since the last run. Without this a job prices
# the slate with the code that happened to be on disk when the Mac was last
# touched by hand, so a merged screen or bug fix silently never reaches a card.
# Deliberately conservative, because a run that fails is worse than one running
# yesterday's code:
#   * --ff-only, so a dirty or diverged checkout is left exactly as it is
#   * only on `main`, so a branch left checked out is never fast-forwarded
#   * GIT_TERMINAL_PROMPT=0 plus a low-speed abort, so neither a credential
#     prompt nor a stalled fetch can hang a daemon with no terminal to answer it
#     (there is no `timeout` on a stock macOS to lean on)
#   * non-fatal throughout: on any failure the run continues on disk code
# An editable reinstall follows a pull that actually moved HEAD, since a merge
# can add a dependency or a console entry point.
pull_latest() {
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
  if [[ "$branch" != "main" ]]; then
    echo "[$(date)] on branch '$branch', not pulling; running the code on disk" >&2
  elif ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[$(date)] uncommitted changes in ${REPO_DIR}; not pulling" >&2
  else
    was=$(git rev-parse HEAD)
    if GIT_TERMINAL_PROMPT=0 git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 \
         pull --ff-only --quiet; then
      now=$(git rev-parse HEAD)
      if [[ "$was" != "$now" ]]; then
        echo "[$(date)] pulled ${was:0:8} -> ${now:0:8}"
        pip install -q -e . || echo "[$(date)] editable reinstall failed" >&2
      fi
    else
      echo "[$(date)] git pull failed; running the code on disk" >&2
    fi
  fi
}

# Arm the next of today's slate passes, so the Mac is awake for it.
arm_next_slate() {
  local nowhm; nowhm=$(date +%H:%M)
  for hm in ${SLATE_WAKES}; do
    if [[ "$nowhm" < "$hm" ]]; then
      pmset_ schedule wake "$(date +%m/%d/%Y) $hm:00" \
        || echo "[$(date)] could not arm the $hm slate wake" >&2
      return 0
    fi
  done
}

if [[ "$MODE" == "night" ]]; then
  # Keep-awake only: re-arm a one-shot wake for the NEXT morning at
  # ${MORNING_WAKE_HHMM} so the machine wakes after the 03:00 sleep. No engine work.
  NEXT=$(date -v+1d +"%m/%d/%Y")
  pmset_ schedule wake "$NEXT ${MORNING_WAKE_HHMM}:00" || echo "[$(date)] could not arm morning wake" >&2
  echo "[$(date)] armed morning wake for $NEXT ${MORNING_WAKE_HHMM}:00"
  # Friday night also arms the Saturday college football pass, which runs
  # before the MLB morning job.
  if [[ "$(date -v+1d +%u)" == "6" ]]; then
    pmset_ schedule wake "$NEXT ${CFB_WAKE_HHMM}:00" || echo "[$(date)] could not arm CFB wake" >&2
    echo "[$(date)] armed CFB wake for $NEXT ${CFB_WAKE_HHMM}:00"
  fi
  # Wednesday and Saturday nights arm the NFL pass (Thursday card, Sunday
  # close re-stamp). %u: 1=Mon .. 7=Sun.
  case "$(date -v+1d +%u)" in
    4|7)
      pmset_ schedule wake "$NEXT ${NFL_WAKE_HHMM}:00" || echo "[$(date)] could not arm NFL wake" >&2
      echo "[$(date)] armed NFL wake for $NEXT ${NFL_WAKE_HHMM}:00"
      ;;
  esac
elif [[ "$MODE" == "close" ]]; then
  # Snapshot today's CLOSING market so tomorrow morning's audit can score closing
  # line value (CLV) -- the fast way to tell whether a pick had real edge. Runs
  # twice a day (afternoon + evening) and merges, because a started game leaves
  # the pre-match board. Cheap (~3 credits + one per event for props).
  mlb-engine close || echo "[$(date)] 'mlb-engine close' exited non-zero" >&2
  # Re-arm tonight's evening capture in case the Mac would sleep before it.
  pmset_ schedule wake "$(date +%m/%d/%Y) ${CLOSE_WAKE_HHMM}:00" || echo "[$(date)] could not arm evening wake" >&2
elif [[ "$MODE" == "cfb" ]]; then
  # Saturday: price today's college football board and email the article PDF +
  # MP3 + workbook as one message. Same caffeinate arrangement as the morning
  # job so WeasyPrint keeps its DYLD path.
  /usr/bin/caffeinate -i -w $$ &
  pull_latest
  cfb-engine run || echo "[$(date)] 'cfb-engine run' exited non-zero" >&2
elif [[ "$MODE" == "nfl" ]]; then
  # Thursday/Sunday: capture -> price -> close -> grade -> card, emailed as one
  # message (card PDF + workbook). Same command as the one-click
  # scripts/macos/run_nfl_week.command, minus opening Excel. Off-season the
  # board is empty (--days 8), so nothing is priced and the job exits 0.
  /usr/bin/caffeinate -i -w $$ &
  pull_latest
  nfl-engine job --card --email || echo "[$(date)] 'nfl-engine job' exited non-zero" >&2
elif [[ "$MODE" == slate-* ]]; then
  # A slate pass: price the games starting inside the next ${SLATE_WINDOW_HOURS}
  # hours -- off posted lineups, on the board as it stands -- then write that
  # block's slate article and email it with the bet card. The run folds these
  # games into today's predictions/Excel and leaves every other game as it was,
  # so the audit still grades one card per slate and the refused early rows
  # keep their reasons. Same caffeinate arrangement as the morning job so
  # WeasyPrint keeps its DYLD path.
  BLOCK="${MODE#slate-}"
  /usr/bin/caffeinate -i -w $$ &
  arm_next_slate
  pull_latest
  VSIN="$HOME/.mlb_engine/vsin_today.csv"
  if [[ -f "$VSIN" ]]; then
    mlb-engine run --within-hours ${SLATE_WINDOW_HOURS} --vsin-csv "$VSIN" \
      || echo "[$(date)] $BLOCK 'mlb-engine run' exited non-zero" >&2
  else
    mlb-engine run --within-hours ${SLATE_WINDOW_HOURS} \
      || echo "[$(date)] $BLOCK 'mlb-engine run' exited non-zero" >&2
  fi
  # Today's card only. "newest on disk" would silently email yesterday's slate
  # as today's on any day the run failed, which is exactly when it matters.
  OUT="$HOME/.mlb_engine/output"
  day=$(date +%Y-%m-%d)
  if [[ -f "$OUT/mlb_recommendations_$day.xlsx" ]]; then
    mlb-engine card || echo "[$(date)] $BLOCK card exited non-zero" >&2
    python -m scripts.regen_slate "$day" --block "$BLOCK" \
      || echo "[$(date)] $BLOCK slate article failed" >&2
    # A pass with no games in its window writes no article, and sends nothing.
    if [[ ! -f "$OUT/PayoffPitch_Slate_${day}_$BLOCK.pdf" ]]; then
      echo "[$(date)] no $BLOCK games today; nothing to email" >&2
    else
      # The regression articles, the power screen and the totals sheet read the day's Statcast
      # and the card as priced so far; written once, by the first pass of the
      # day that has games, and emailed once, with that pass. A pass finding
      # the once-a-day stamp already on disk sends only its slate and the card.
      DAILY_STAMP="$OUT/.daily_sent_$day"
      WITH_DAILY=""
      if [[ ! -f "$DAILY_STAMP" ]]; then
        if [[ ! -f "$OUT/PayoffPitch_Regression_$day.pdf" ]]; then
          pkl=$(ls -t "$HOME/.mlb_engine/cache/"statcast_*.pkl 2>/dev/null | head -1) || true
          if [[ -n "$pkl" ]]; then
            python -m scripts.regen_regression --date "$day" --statcast "$(basename "$pkl")" \
              || echo "[$(date)] regression articles failed" >&2
          else
            echo "[$(date)] no Statcast cache pkl; skipping regression articles" >&2
          fi
        fi
        if [[ ! -f "$OUT/power_screen_$day.pdf" ]]; then
          python scripts/power_screen.py --date "$day" \
            || echo "[$(date)] power screen failed" >&2
        fi
        # --if-stale: a sheet written by an older band version is rescored, one
        # already on the current bands is kept.
        python -m scripts.totals_sheet "$day" --if-stale \
          || echo "[$(date)] totals sheet failed" >&2
        if [[ ! -f "$OUT/totals_audit_$day.xlsx" ]]; then
          python -m scripts.totals_audit "$day" \
            || echo "[$(date)] totals audit failed" >&2
        fi
        WITH_DAILY="--with-daily"
      fi
      # shellcheck disable=SC2086  # WITH_DAILY is one flag or nothing
      if python -m scripts.email_daily_package "$day" --block "$BLOCK" $WITH_DAILY; then
        [[ -n "$WITH_DAILY" ]] && touch "$DAILY_STAMP"
      else
        echo "[$(date)] $BLOCK package email failed" >&2
      fi
    fi
  else
    echo "[$(date)] no workbook for $day; skipping the $BLOCK email" >&2
  fi
else
  # The morning job (before noon): capture the Opta benchmark, then grade
  # yesterday and email the ledger/report.
  #
  # Keep the Mac awake WITHOUT wrapping the engine in caffeinate: caffeinate is
  # a SIP-protected /usr/bin binary, and launching it strips DYLD_* from the
  # environment -- so WeasyPrint, run as caffeinate's child, could not find the
  # Homebrew libs. Instead hold a background caffeinate tied to THIS script's
  # PID and run the engine as a direct child, so DYLD_*/SSL_CERT_FILE survive.
  /usr/bin/caffeinate -i -w $$ &

  # Arm a one-shot wake for TONIGHT's closing snapshot (same calendar day), so
  # the close daemon can fire even if the Mac would otherwise sleep by evening.
  pmset_ schedule wake "$(date +%m/%d/%Y) ${DAY_CLOSE_WAKE_HHMM}:00" || echo "[$(date)] could not arm afternoon wake" >&2

  # 0) take whatever has been merged since yesterday.
  pull_latest

  # Arm the first of today's slate passes.
  arm_next_slate

  # 1)-2) nothing is priced or emailed here: a card written now would sit six or
  #       more hours out from most of the slate and be refused on the clock. The
  #       slate passes own the card, the articles and the email.

  # 3) capture the Opta benchmark, both halves of it. VSIN's day offset clamps
  #    at yesterday, so a slate missed by a day is gone permanently -- which is
  #    why this is a daemon step and not something to remember to run. day=-1
  #    brings back last night's graded outcomes; day=0 takes today's calls
  #    before the games start. Free, no credits, and a failure is not fatal.
  mlb-engine opta --day -1 || echo "[$(date)] 'mlb-engine opta --day -1' exited non-zero" >&2
  mlb-engine opta || echo "[$(date)] 'mlb-engine opta' exited non-zero" >&2

  # 4) grade yesterday + email the ledger/report.
  ${AUDIT_CMD} || echo "[$(date)] '${AUDIT_CMD}' exited non-zero" >&2
fi
