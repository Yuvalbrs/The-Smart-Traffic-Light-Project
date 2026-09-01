@echo off
REM Double-click entry point for the demo. Delegates everything to run_app.py; this file only
REM has to (1) resolve its own directory so it works regardless of the double-clicker's cwd,
REM and (2) keep the window open on failure so the error is readable instead of the console
REM flashing shut the instant a double-clicked .bat's script exits non-zero.
setlocal

REM %~dp0 = the drive+path of this .bat file, trailing backslash included - resolves the repo
REM root even when launched from Explorer (cwd is unpredictable there) or a shortcut.
set REPO_ROOT=%~dp0
set PYTHON=%REPO_ROOT%.venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo ERROR: %PYTHON% not found.
    echo Create the virtual environment first:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

"%PYTHON%" "%REPO_ROOT%run_app.py" %*

if errorlevel 1 (
    echo.
    echo run_app.py exited with an error - see above.
    pause
    exit /b 1
)

endlocal
