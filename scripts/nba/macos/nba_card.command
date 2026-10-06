#!/bin/bash
# One-click NBA daily package: record a board-only pricing pass (featured board,
# 3 Odds API credits), grade yesterday if not yet graded (free ESPN finals), then
# render the slate PDF + betting workbook and email both.
#   --no-price   skip the pricing pass: re-render from the passes already
#                recorded today (0 credits). Implied by --date.
# Every other argument goes to `nba-engine card` (e.g. --to ADDR, --date D).
# Runs as a Desktop symlink; the link is resolved so the checkout is found.
set -uo pipefail
_src=$0
while [ -L "$_src" ]; do
    _link=$(readlink "$_src")
    case $_link in
        /*) _src=$_link ;;
        *) _src=$(dirname "$_src")/$_link ;;
    esac
done
# shellcheck disable=SC1091
. "$(dirname "$_src")/_env.sh"

PRICE=1
ARGS=()
for a in "$@"; do
    case $a in
        --no-price) PRICE=0 ;;
        --date|--date=*) PRICE=0; ARGS+=("$a") ;;
        *) ARGS+=("$a") ;;
    esac
done

if [ "$PRICE" = 1 ]; then
    nba-engine capture --board-only --record "card-$(date +%H%M)" \
        || echo "pricing pass failed; carding from the passes already recorded"
fi
nba-engine grade || true
nba-engine card --email ${ARGS[@]+"${ARGS[@]}"}
echo
echo "Package: ${NBAE_OUTPUT_DIR:-${NBAE_DATA_DIR:-$HOME/.nba_engine}/output}"
