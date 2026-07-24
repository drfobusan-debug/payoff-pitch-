@echo off
REM Open the audit ledger workbook (builds it via an audit if it doesn't exist yet).
setlocal
cd /d "%~dp0..\.."
set "LEDGER=%USERPROFILE%\.mlb_engine\output\ledger.xlsx"
if not exist "%LEDGER%" (
    echo No ledger yet - running an audit to build it...
    call .venv\Scripts\activate.bat
    mlb-engine audit
)
if exist "%LEDGER%" (
    start "" "%LEDGER%"
) else (
    echo Still no ledger - audit a slate first ^(run_audit.bat^).
    pause
)
endlocal
