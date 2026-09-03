@echo off
REM Double-click entry point for a fresh clone. Everything lives in scripts\setup.py; this file
REM only has to (1) find a Python that exists BEFORE the virtual environment does, (2) resolve the
REM repo root regardless of the double-clicker's cwd, and (3) keep the window open on failure so
REM the error is readable instead of the console flashing shut.
setlocal

set REPO_ROOT=%~dp0

REM The venv does not exist yet - that is what this script creates - so we need a system Python.
REM `py` is the Windows launcher and is the more reliable of the two; fall back to whatever `python`
REM resolves to. Note the Microsoft Store stub also answers to `python` and does nothing useful,
REM which is why the launcher is tried first.
set PYTHON=
where py >nul 2>&1 && set PYTHON=py -3
if not defined PYTHON (
    where python >nul 2>&1 && set PYTHON=python
)

if not defined PYTHON (
    echo ERROR: no Python found on PATH.
    echo Install Python 3.11 or newer from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer.
    pause
    exit /b 1
)

pushd "%REPO_ROOT%"
%PYTHON% -m scripts.setup %*
set RC=%ERRORLEVEL%
popd

if not "%RC%"=="0" (
    echo.
    echo setup failed - see above.
    pause
    exit /b %RC%
)

endlocal
