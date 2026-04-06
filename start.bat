@echo off
REM Ghost Platform Startup Script for Windows

echo Starting Ghost Platform...
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Set environment
set PYTHONPATH=%~dp0src
set GHOST_LOG_FILE=ghost.log

REM Parse command line arguments
if "%1"=="agent" goto run_agent
if "%1"=="server" goto run_server
if "%1"=="daemon" goto run_daemon

:run_server
echo Starting MCP Server...
python -m ghost.mcp_server
goto end

:run_agent
echo Starting Training Agent...
python -m ghost.agents.training_agent
goto end

:run_daemon
echo Starting Agent in daemon mode...
start /min python -m ghost.agents.training_agent
echo Agent running in background
goto end

:end
pause
