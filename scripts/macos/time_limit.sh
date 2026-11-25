# shellcheck shell=bash
# Run a step for at most N seconds, then stop it and carry on.
#
# Sourced by the card shortcuts. Their podcast read runs before the card and has
# no overall limit, only per-request ones; on 2026-11-24 the CBB 11:30 read ran
# until 03:32 the next morning, so neither card that day was priced, and launchd
# skipped the 16:30 run because the 11:30 one was still going.
#
# Usage: run_for SECONDS command [args...]   (status 124 when it was stopped)

run_for() {
    local secs="$1" pid dog rc start=$SECONDS
    shift
    "$@" &
    pid=$!
    (
        s=""
        trap 'kill "$s" 2>/dev/null; exit 0' TERM
        sleep "$secs" &
        s=$!
        wait "$s"
        echo "$1: still running after ${secs}s; stopped so the card goes out" >&2
        kill -TERM "$pid" 2>/dev/null
        sleep 15 &
        s=$!
        wait "$s"
        kill -KILL "$pid" 2>/dev/null
    ) &
    dog=$!
    wait "$pid"
    rc=$?
    kill -TERM "$dog" 2>/dev/null
    wait "$dog" 2>/dev/null
    if [ $((SECONDS - start)) -ge "$secs" ] && [ "$rc" -gt 128 ]; then
        rc=124
    fi
    return "$rc"
}
