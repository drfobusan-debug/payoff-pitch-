@echo off
REM One-click daily MLB predictions -> opens the Excel workbook.
setlocal
cd /d "%~dp0..\.."
call .venv\Scripts\activate.bat

set "VSIN=%USERPROFILE%\.mlb_engine\vsin_today.csv"
if exist "%VSIN%" (
    mlb-engine run --vsin-csv "%VSIN%"
) else (
    mlb-engine run
)

REM open the most recent workbook
for /f "delims=" %%f in ('dir /b /o-d "%USERPROFILE%\.mlb_engine\output\*.xlsx"') do (
    start "" "%USERPROFILE%\.mlb_engine\output\%%f"
    goto :done
)
:done
endlocal
