@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0tools\launch_publish.ps1" -Target GitHub
set "publish_exit=%ERRORLEVEL%"
echo.
pause
exit /b %publish_exit%
