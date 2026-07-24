@echo off
REM One-time setup: create venv and install the engine.
setlocal
cd /d "%~dp0..\.."
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .
echo.
echo Setup complete. Create a desktop shortcut to scripts\windows\run_predictions.bat
echo For a ledger shortcut with the ledger icon, run: scripts\windows\install_ledger_shortcut.bat
pause
endlocal
