@echo off
REM Create a "Ledger" desktop shortcut (.lnk) with the ledger icon.
setlocal
cd /d "%~dp0..\.."
set "REPO=%CD%"
set "ICON=%REPO%\assets\ledger.ico"
set "TARGET=%REPO%\scripts\windows\open_ledger.bat"
set "LNK=%USERPROFILE%\Desktop\Ledger.lnk"

if not exist "%ICON%" (
    echo Generating ledger icon...
    call .venv\Scripts\activate.bat 2>nul
    python scripts\make_ledger_icon.py assets
)

powershell -NoProfile -Command ^
  "$w = New-Object -ComObject WScript.Shell; $s = $w.CreateShortcut('%LNK%'); $s.TargetPath = '%TARGET%'; $s.WorkingDirectory = '%REPO%'; $s.IconLocation = '%ICON%'; $s.Description = 'Open the MLB engine audit ledger'; $s.Save()"

echo Created %LNK%
echo A green Ledger icon is now on your Desktop - double-click it to open the ledger.
endlocal
