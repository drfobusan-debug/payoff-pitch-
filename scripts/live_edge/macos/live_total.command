#!/bin/bash
# One sport's live-total paper test: poll from before the first start until the slate is final,
# then email this run's tickets and the running report as its own email. Paper only; at most
# --max-credits Odds API calls (totals only, 1 credit each).
#   live_total.command cbb|nba|nhl|cfb|nfl
# LIVE_<SPORT>_HOURS / LIVE_<SPORT>_MAX_CREDITS override the defaults below.
set -uo pipefail
sport="${1:?usage: live_total.command cbb|nba|nhl|cfb|nfl}"
case "$sport" in
    cbb) hours=15 ;;
    nba | nhl) hours=14 ;;
    cfb) hours=16 ;;
    nfl) hours=13 ;;
    *) echo "unknown sport: $sport" >&2; exit 2 ;;
esac
up=$(echo "$sport" | tr '[:lower:]' '[:upper:]')
hours_var="LIVE_${up}_HOURS"
credits_var="LIVE_${up}_MAX_CREDITS"
# The CBB preamble: repo + .venv, and THE_ODDS_API_KEY / Gmail from ~/.cbb_engine/engine.env.
# shellcheck disable=SC1091
. "$(dirname "$0")/../../cbb/macos/_env.sh"
# The sport's own engine.env on top, where it has one (CFB/NBA/NHL/NFL keep keys there).
if [ "$sport" != cbb ] && [ -f "$HOME/.${sport}_engine/engine.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$HOME/.${sport}_engine/engine.env"
    set +a
fi
python -m live_edge.stoppage run --sport "$sport" --hours "${!hours_var:-$hours}" \
    --max-credits "${!credits_var:-600}" --email
