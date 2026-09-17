@echo off
setlocal
title Revit Family Manifest - Scan

REM ====================================================================
REM  Double-click this file to scan your family library.
REM  Nothing to remember, no PowerShell to type.
REM
REM  Keep this file in the SAME FOLDER as Scan-RevitFamilies.ps1.
REM ====================================================================


REM --- Your library. Edit these if it ever moves. ----------------------
set "LIB1=D:\Dropbox\.0REVIT FAMILIES\Manufacturers"
set "LIB2=D:\Dropbox\.0REVIT FAMILIES\DOWNLOAD"


REM --- Where the CSV lands. ---------------------------------------------
set "OUTDIR=%USERPROFILE%\Desktop"

REM  Prefer it to go straight to Google Drive so the sheet can import it?
REM  Delete the REM from the next line and fix the drive letter, then put a
REM  REM in front of the line above.
REM set "OUTDIR=G:\My Drive\RevitManifest"


REM --- Revit release detection. -----------------------------------------
REM  OFF by default on purpose. If the library is Dropbox online-only, this
REM  forces Dropbox to download every single family. Turn it on only once
REM  the folder is set to "Available offline": delete the REM below and put
REM  a REM in front of the blank line above it.
set "READVERSION="
REM set "READVERSION=-ReadVersion"

REM ====================================================================
REM  Nothing below here needs editing.
REM ====================================================================

set "PS1=%~dp0Scan-RevitFamilies.ps1"

if not exist "%PS1%" (
  echo.
  echo   Could not find Scan-RevitFamilies.ps1
  echo.
  echo   It has to sit in this same folder:
  echo     %~dp0
  echo.
  echo   Download it from GitHub and drop it next to this file.
  echo.
  pause
  exit /b 1
)

if not exist "%LIB1%" (
  echo.
  echo   Library folder not found:
  echo     %LIB1%
  echo.
  echo   If your library moved, open this .cmd file in Notepad
  echo   ^(right-click it, Edit^) and fix the LIB1 line near the top.
  echo.
  pause
  exit /b 1
)

set "ROOTS='%LIB1%'"
if defined LIB2 if exist "%LIB2%" set "ROOTS='%LIB1%','%LIB2%'"

if not exist "%OUTDIR%" mkdir "%OUTDIR%" 2>nul

echo.
echo   Scanning your Revit family library...
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { Unblock-File -LiteralPath '%PS1%' -ErrorAction SilentlyContinue; & '%PS1%' -Roots %ROOTS% -OutDir '%OUTDIR%' %READVERSION% }"

if errorlevel 1 (
  echo.
  echo   The scan reported a problem. The message above says what.
)

echo.
echo   Press any key to close.
pause >nul
endlocal
