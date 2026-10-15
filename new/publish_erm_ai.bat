@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
echo Publishing source and all models. Hugging Face uses one resumable PR for maintainer review.
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0tools\launch_publish.ps1" -Target All -ModelPublishMode PullRequest
set "publish_exit=%ERRORLEVEL%"
echo.
if not "%publish_exit%"=="0" echo Publishing stopped. Read the message above; completed uploads are kept.
pause
exit /b %publish_exit%
