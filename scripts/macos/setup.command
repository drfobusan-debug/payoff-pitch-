#!/bin/bash
# One-time setup: create venv and install the engine.
cd "$(dirname "$0")/../.." || exit 1
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
echo
echo "Setup complete. Put the daily run on your Desktop as a LINK, never a copy --"
echo "a copy stops tracking git pull and prices with whatever build it finds:"
echo "  ln -s \"$(pwd)/scripts/macos/run_predictions.command\" \"\$HOME/Desktop/PAYOFF PITCH.command\""
echo "For a ledger shortcut with the ledger icon, run: scripts/macos/install_ledger_shortcut.command"
