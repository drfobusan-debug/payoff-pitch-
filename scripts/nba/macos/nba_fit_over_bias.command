#!/bin/bash
# Fit the prop Over bias on the Mac (stats.nba/ESPN reach, the history archive):
# check archive coverage, archive the ESPN finals, then fit and push
# params/over_bias to engine-state. Free -- no Odds API credits. Seasons default
# to 2023-24..2025-26; pass --season ... to replace them.
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

if [ $# -eq 0 ]; then
    set -- --season 2023-24 --season 2024-25 --season 2025-26
fi
nba-engine coverage "$@" && nba-engine boxes "$@" && nba-engine fit-over-bias "$@" --push
