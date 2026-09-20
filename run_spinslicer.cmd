@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo SpinSlicer environment was not found.
    echo Create .venv and install requirements before starting the source version.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "SpinSlicer.py"
if errorlevel 1 pause
