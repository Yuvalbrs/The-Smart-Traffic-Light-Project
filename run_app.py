"""One-command launcher: build the SPA if needed, start the FastAPI hub, open the browser.

The demo used to need three manual steps in three terminals (uvicorn, `npm run dev`, Unity
Play). This collapses the "run it" part to one: double-click `run_app.bat` (which just calls
`.venv\\Scripts\\python.exe run_app.py`) or run `python run_app.py` directly. Unity stays a
separate, optional step (see `unity/README.md`) - it is a 3-D *view* of the same live episode,
not a dependency of the web app.

What this does NOT do: start `npm run dev`. The FastAPI hub serves the already-built SPA
(`frontend/dist/`) at `/` itself, so a Vite dev server on :5173 would just be a second,
redundant static-file server that the operator would then have to remember to close.

Run::

    .venv\\Scripts\\python.exe run_app.py
    .venv\\Scripts\\python.exe run_app.py --no-browser --port 8001
    .venv\\Scripts\\python.exe run_app.py --rebuild        # force a fresh frontend build
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_VENV_DIR = _REPO_ROOT / ".venv"
_DB_PATH = _REPO_ROOT / "data" / "traffic.db"
_FRONTEND_DIR = _REPO_ROOT / "frontend"
_DIST_DIR = _FRONTEND_DIR / "dist"
_NODE_MODULES_DIR = _FRONTEND_DIR / "node_modules"

# How long to wait for `/health` to return 200 before giving up. Cold start includes SUMO/
# libsumo import and network build (scripts/build_network.py runs lazily on first session, but
# the app import itself is not instant), so this is generous on purpose - a demo laptop failing
# here with a vague timeout is worse than waiting an extra 10s.
_HEALTH_TIMEOUT_S = 60.0
_HEALTH_POLL_INTERVAL_S = 0.5


class LauncherError(Exception):
    """A prerequisite failed or a step could not complete. Message is shown as-is, no traceback."""


def _fail(message: str) -> "NoReturn":  # noqa: F821 - typing.NoReturn not imported to stay stdlib-minimal
    raise LauncherError(message)


def check_sumo_home() -> None:
    """SUMO_HOME must be set - traci/libsumo import against it, and that import happens at
    `src.api.server` import time (see src/api/server.py -> src/env/*), well before /health
    would ever answer."""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        _fail(
            "SUMO_HOME is not set.\n"
            "  Install SUMO (https://sumo.dlr.de) if it is not already, then set it, e.g. in "
            "PowerShell:\n"
            '    setx SUMO_HOME "C:\\Program Files (x86)\\Eclipse\\Sumo"\n'
            "  Close and reopen the terminal (or double-click run_app.bat again) after setx - "
            "it does not affect the current session."
        )
    if not Path(sumo_home).exists():
        _fail(
            f"SUMO_HOME is set to {sumo_home!r} but that path does not exist.\n"
            "  Fix the SUMO_HOME environment variable to point at your SUMO install."
        )


def check_venv() -> None:
    """The repo's own `.venv` must exist. This does not guarantee the *running* interpreter is
    it (someone can `python run_app.py` with a different Python on PATH) - that mismatch would
    surface immediately as an ImportError for fastapi/uvicorn, which is diagnosed separately."""
    if not _VENV_DIR.exists():
        _fail(
            f"No virtual environment found at {_VENV_DIR}.\n"
            "  Create it and install dependencies:\n"
            "    python -m venv .venv\n"
            "    .venv\\Scripts\\pip install -r requirements.txt"
        )


def check_database() -> None:
    if not _DB_PATH.exists():
        _fail(
            f"{_DB_PATH} not found.\n"
            "  Initialize the results database:\n"
            "    .venv\\Scripts\\python.exe -m scripts.init_db"
        )


def ensure_frontend_built(*, skip_build: bool, rebuild: bool) -> None:
    """Make sure `frontend/dist/` exists (the hub serves it as the SPA at `/`).

    Building is the one prerequisite this launcher can fix on its own instead of just failing,
    because on a fresh checkout `frontend/dist/` is gitignored (see `frontend/.gitignore` /
    build output) and a demo laptop should not need a manual `npm run build` the night before.
    """
    if rebuild and _DIST_DIR.exists():
        shutil.rmtree(_DIST_DIR)

    if _DIST_DIR.exists() and not rebuild:
        return

    if skip_build:
        _fail(
            f"{_DIST_DIR} not found and --skip-build was passed, so it cannot be built.\n"
            "  Either drop --skip-build, or build it yourself first:\n"
            "    cd frontend && npm install && npm run build"
        )

    npm = shutil.which("npm")
    if npm is None:
        _fail(
            "frontend/dist/ is missing and npm is not on PATH, so it cannot be built "
            "automatically.\n"
            "  Install Node.js (https://nodejs.org) so `npm` is on PATH, then re-run, or build "
            "manually:\n"
            "    cd frontend && npm install && npm run build"
        )

    if not _NODE_MODULES_DIR.exists():
        print("[run_app] frontend/node_modules/ missing - running `npm install` first...")
        _stream(["npm", "install"], cwd=_FRONTEND_DIR)

    print(
        "[run_app] Building the frontend (frontend/dist/ missing or --rebuild was passed). "
        "The first build takes a few minutes (tsc + vite); later ones are much faster."
    )
    _stream(["npm", "run", "build"], cwd=_FRONTEND_DIR)

    if not _DIST_DIR.exists():
        # npm run build exiting 0 without producing dist/ would be a silent failure worth
        # catching explicitly rather than letting uvicorn 404 on `/` and leaving the operator
        # to guess why.
        _fail(
            f"`npm run build` finished but {_DIST_DIR} still does not exist.\n"
            "  Check the build output above for errors and re-run with --rebuild."
        )


def _stream(cmd: list[str], *, cwd: Path) -> None:
    """Run a command with output streamed live (not captured), raising on a non-zero exit."""
    # shell=True on Windows: `npm`/`npm.cmd` resolved via shutil.which() above is passed as
    # cmd[0], but npm ships as a .cmd shim, and CreateProcess cannot exec a .cmd directly
    # without going through the shell - subprocess would raise FileNotFoundError/WinError 193
    # otherwise despite shutil.which() having found it.
    result = subprocess.run(cmd, cwd=str(cwd), shell=True)
    if result.returncode != 0:
        _fail(f"`{' '.join(cmd)}` failed (exit {result.returncode}) - see output above.")


def start_hub(port: int) -> subprocess.Popen:
    """Start the FastAPI hub as a subprocess so a Ctrl+C here can shut it down cleanly and so
    its stdout/stderr (uvicorn's own startup log, including a port-in-use error) streams
    straight to this console instead of being swallowed."""
    env = dict(os.environ)
    # LIBSUMO_AS_TRACI=1: use the in-process libsumo backend (see requirements.txt), matching
    # every other entry point in this repo (train_matrix.py, server.py's own docstring).
    env["LIBSUMO_AS_TRACI"] = "1"
    python = sys.executable  # the interpreter running this launcher - already the venv's.
    cmd = [
        python, "-m", "uvicorn", "src.api.server:app",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    # CREATE_NEW_PROCESS_GROUP: lets us send CTRL_BREAK_EVENT to just the child on Ctrl+C
    # instead of relying on Windows' default behaviour of delivering Ctrl+C to the whole
    # console process group (which would race this parent's own KeyboardInterrupt handling).
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env, creationflags=creationflags)


def wait_for_health(port: int, proc: subprocess.Popen) -> None:
    """Poll GET /health until it returns 200, the process dies, or we time out."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + _HEALTH_TIMEOUT_S
    print(f"[run_app] waiting for the hub at {url} ...")
    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            _fail(
                f"The hub process exited early (code {exit_code}) before answering /health.\n"
                "  Scroll up for uvicorn's own error - the most common cause is another "
                f"process already listening on port {port}. Try:\n"
                f"    .venv\\Scripts\\python.exe run_app.py --port {port + 1}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    json.loads(resp.read())  # confirm it is actually our JSON, not a stray 200
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass  # normal while the server is still starting up
        time.sleep(_HEALTH_POLL_INTERVAL_S)
    _terminate(proc)
    _fail(
        f"The hub did not answer {url} within {_HEALTH_TIMEOUT_S:.0f}s.\n"
        "  It may still be importing SUMO/libsumo, or something upstream is hanging. Re-run "
        "with a longer wait is not an option here - check the uvicorn output above for the "
        "actual error."
    )


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort clean shutdown of the hub subprocess. Never raises - this runs from both
    the happy-path exit and the Ctrl+C handler, and a shutdown routine that itself throws just
    replaces one traceback with another."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-browser", action="store_true", help="do not auto-open the browser")
    parser.add_argument("--port", type=int, default=8000, help="hub port (default: 8000)")
    parser.add_argument(
        "--skip-build", action="store_true",
        help="do not build the frontend even if frontend/dist/ is missing (fail instead)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="force a fresh frontend build even if frontend/dist/ already exists",
    )
    args = parser.parse_args()

    try:
        check_sumo_home()
        check_venv()
        check_database()
        ensure_frontend_built(skip_build=args.skip_build, rebuild=args.rebuild)

        proc = start_hub(args.port)
        try:
            wait_for_health(args.port, proc)
        except LauncherError:
            _terminate(proc)
            raise

        url = f"http://127.0.0.1:{args.port}/"
        print()
        print("=" * 60)
        print(f"  Smart Traffic Intersection - running at {url}")
        print("  Press Ctrl+C here to stop it.")
        print("=" * 60)
        print()
        if not args.no_browser:
            webbrowser.open(url)

        # Block until Ctrl+C (or the hub dies on its own) - this call is what makes Ctrl+C
        # land here in Python rather than only in the child's console.
        proc.wait()
        if proc.returncode not in (0, None):
            _fail(f"The hub exited unexpectedly (code {proc.returncode}). See output above.")

    except KeyboardInterrupt:
        print("\n[run_app] stopping...")
        try:
            _terminate(proc)  # type: ignore[possibly-undefined]
        except NameError:
            pass  # Ctrl+C landed before `proc` was ever created
        sys.exit(0)
    except LauncherError as exc:
        print(f"\n[run_app] ERROR: {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
