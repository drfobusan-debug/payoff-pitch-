#!/bin/bash
# Weekly NBA ledger audit email: the last 7 slates, then the season to date
# (same tables as `nba-engine audit`). Graded rows only -- free, no Odds API
# credits, and it changes no price, gate or buy. Extra arguments go straight to
# `nba-engine audit` (e.g. --to you@example.com, --recent 14).
set -uo pipefail
_src=$0
while [ -L "$_src" ]; do
    _link=$(readlink "$_src")
    case $_link in
        /*) _src=$_link ;;
        *) _src=$(dirname "$_src")/$_link ;;
    esac
done
. "$(dirname "$_src")/_env.sh"

# Grade yesterday first: a Sunday slate is otherwise graded only by that
# evening's card, after this Monday morning send.
nba-engine grade || true
nba-engine audit --recent 7 --email "$@"
