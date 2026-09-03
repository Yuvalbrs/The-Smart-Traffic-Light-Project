"""One-command launcher: build the SPA if needed, start the FastAPI hub, open the browser.

The demo used to need three manual steps in three terminals (uvicorn, `npm run dev`, Unity
Play). This collapses the "run it" part to one: double-click `run_app.bat` (which just calls
`.venv\\Scripts\\python.exe run_app.py`) or run `python run_app.py` directly.

The 3-D window is now one click further in, not a fourth terminal: the dashboard's **open 3-D
view** button posts to `/viewer`, which launches the built Unity player onto the same episode
feed (`src/api/viewer.py`). It is deliberately NOT started here. It is a *view* of a live
episode, not a dependency of the web app; a machine with no player build would fail at startup
for something optional, and a demo that always opens a second window cannot choose not to.
Where no build exists the button explains how to make one, and `unity/README.md` covers the
Unity-Editor route.

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


def check_no_hub_already_running(port: int) -> None:
    """Refuse to start a second hub over the first one's database.

    This is the project's sharpest footgun made unarmable. `data/traffic.db` is SQLite in WAL mode
    and the container reaches it through a Docker Desktop bind mount, which on Windows cannot share
    the file with the host. If the container is up and the native hub is started too, the
    container's handle goes stale and **every subsequent write fails silently**: episodes still run
    and still stream to the dashboard and the 3-D viewer, they simply never reach the database and
    vanish from the archive. Nothing in the UI reports it.

    The README has said "never run both" since the day it was diagnosed, but a rule you have to
    remember is not a guard. This one is checked, every start, by the process that would cause it.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            if resp.status != 200:
                return
            json.loads(resp.read())  # a stray 200 from something else is not our hub
    except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError):
        return  # nothing there, or not us - the normal case

    _fail(
        f"A Smart-Traffic hub is ALREADY answering on port {port}.\n"
        "  Almost always this is `docker compose up` still running.\n"
        "\n"
        "  Starting a second hub over the same data/traffic.db is the one failure mode that does\n"
        "  not announce itself: the container's SQLite handle goes stale and every later write is\n"
        "  silently discarded. Episodes appear to run perfectly and are never recorded.\n"
        "\n"
        "  Use one or the other:\n"
        "    docker compose down            # then re-run this\n"
        f"    ...or open the running one:   http://127.0.0.1:{port}/\n"
        f"    ...or use another port:       .\\run_app.bat --port {port + 1}"
    )


def _sumo_home_from_wheel() -> Path | None:
    """SUMO_HOME as provided by the `eclipse-sumo` wheel, if it is installed.

    The wheel ships the binaries and the tools tree inside the package itself and exports the
    directory as `sumo.SUMO_HOME`, but it sets no environment variable - so a machine whose only
    SUMO is the wheel looks, to a bare `os.environ` check, exactly like a machine with no SUMO.
    """
    try:
        import sumo  # noqa: PLC0415 - optional, and only meaningful at this point in startup
    except ImportError:
        return None
    home = Path(getattr(sumo, "SUMO_HOME", "") or Path(sumo.__file__).resolve().parent)
    # Trust the layout only if it is really there: sumolib.checkBinary() resolves through
    # SUMO_HOME/bin, and src/env/ imports the tools tree.
    return home if (home / "bin").is_dir() and (home / "tools").is_dir() else None


def check_sumo_home() -> None:
    """Make SUMO usable, or say exactly what is missing.

    Two ways to have SUMO, and both are accepted, because requiring the first is what made this
    launcher fail on every machine that was not the development one:

      * a **native install** with SUMO_HOME set (how the dev machine is configured); or
      * the **`eclipse-sumo` wheel** in the virtual environment, which ships `sumo` and
        `netconvert` itself. `scripts/setup.py` installs it when no native SUMO is found, so this
        is the ordinary case on a fresh laptop.

    traci/libsumo import against SUMO_HOME at `src.api.server` import time (src/api/server.py ->
    src/env/*), well before /health would ever answer, so it has to be right before the hub starts.
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home and Path(sumo_home).exists():
        return

    wheel_home = _sumo_home_from_wheel()
    if wheel_home is not None:
        # Exported, not merely used locally: start_hub() launches uvicorn as a child process, which
        # inherits this environment. Setting it here is what makes the hub itself find SUMO.
        os.environ["SUMO_HOME"] = str(wheel_home)
        os.environ["PATH"] = str(wheel_home / "bin") + os.pathsep + os.environ.get("PATH", "")
        print(f"[run_app] SUMO_HOME unset; using the eclipse-sumo wheel at {wheel_home}")
        return

    if sumo_home:
        _fail(
            f"SUMO_HOME is set to {sumo_home!r} but that path does not exist, and the "
            "eclipse-sumo wheel is not installed either.\n"
            "  Either fix SUMO_HOME to point at your SUMO install, or let setup do it:\n"
            "    setup.bat"
        )
    _fail(
        "No SUMO found: SUMO_HOME is not set and the eclipse-sumo wheel is not installed.\n"
        "  The easiest fix installs the wheel into .venv for you - no native install needed:\n"
        "    setup.bat\n"
        "  Or, if you have SUMO natively (https://sumo.dlr.de), point at it in PowerShell:\n"
        '    setx SUMO_HOME "C:\\Program Files (x86)\\Eclipse\\Sumo"\n'
        "  setx does not affect the current terminal - reopen it, or just double-click "
        ".\\run_app.bat again."
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
        check_no_hub_already_running(args.port)
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
