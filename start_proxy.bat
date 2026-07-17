@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

echo ========================================
echo LLM API Key Proxy - Запуск
echo ========================================
echo.

cd /d "%~dp0"

REM Auto-update before launch
echo.
if exist ".git" (
    git diff --quiet
    if errorlevel 1 (
        echo Local changes detected. Skipping auto-update to avoid conflicts.
    ) else (
        echo Checking for updates...
        git pull --quiet
        if errorlevel 1 (
            echo Warning: git pull failed, continuing with current version...
        ) else (
            echo Repository is up to date.
        )
    )
) else (
    echo Not a git repository, skipping update.
)
echo.

REM Disable aiodns to fix "Domain name not found" errors when ping works
REM This must be set BEFORE Python imports aiohttp
set AIOHTTP_NO_EXTENSIONS=1

REM Activate virtual environment if available
if exist ".venv\Scripts\activate.bat" (
    echo Активация виртуального окружения...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Активация виртуального окружения...
    call venv\Scripts\activate.bat
) else (
    echo Внимание: виртуальное окружение не найдено, используется системный Python
)

echo.
if "%PROXY_HOST%"=="" set PROXY_HOST=127.0.0.1
if "%PROXY_PORT%"=="" set PROXY_PORT=8000

set "LISTEN_PID="
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /R /C:":%PROXY_PORT% .*LISTENING"') do (
    set "LISTEN_PID=%%P"
)

if defined LISTEN_PID (
    echo.
    echo Port %PROXY_PORT% is already in use by PID !LISTEN_PID!.
    tasklist /FI "PID eq !LISTEN_PID!"
    echo.
    echo This usually means the previous proxy process did not exit cleanly.
    set /p STOP_OLD_PROXY=Stop this process and start a fresh proxy? [y/N]:
    if /I "!STOP_OLD_PROXY!"=="Y" (
        echo Stopping PID !LISTEN_PID!...
        taskkill /PID !LISTEN_PID! /T /F
        if errorlevel 1 (
            echo Failed to stop PID !LISTEN_PID!. Please close it manually.
            pause
            exit /b 1
        )
        timeout /t 2 /nobreak >nul
    ) else (
        echo Existing process left running. Startup cancelled.
        pause
        exit /b 1
    )
)

echo Запуск прокси-сервера на http://%PROXY_HOST%:%PROXY_PORT%
echo.
echo.
echo Для остановки нажмите Ctrl+C
echo ========================================
echo.

python src/proxy_app/main.py --host %PROXY_HOST% --port %PROXY_PORT% || py src/proxy_app/main.py --host %PROXY_HOST% --port %PROXY_PORT%

pause
