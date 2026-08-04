#!/bin/bash
# One-click daily college-football card: prices today's slate, then emails the
# Excel bet sheet + article (PDF) + MP3 narration, and opens the workbook.
cd "$(dirname "$0")/../../.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

cfb-engine run

latest=$(ls -t "$HOME/.cfb_engine/output/"*.xlsx 2>/dev/null | head -1)
[ -n "$latest" ] && xdg-open "$latest"
