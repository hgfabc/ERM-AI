@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
echo Publishing all stream/fold models to one Hugging Face PR. Interrupted uploads can be resumed.
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0tools\launch_publish.ps1" -Target HuggingFace -ModelPublishMode PullRequest
set "publish_exit=%ERRORLEVEL%"
echo.
pause
exit /b %publish_exit%
