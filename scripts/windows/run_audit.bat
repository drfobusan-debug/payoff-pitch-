@echo off
REM Nightly self-audit: grade yesterday's recommendations, update the scorecard.
setlocal
cd /d "%~dp0..\.."
call .venv\Scripts\activate.bat
mlb-engine audit
echo Scorecard: %USERPROFILE%\.mlb_engine\audit\scorecard.csv
pause
endlocal
