"""One-command setup for a fresh clone: check the machine, fetch the data, install the stack.

    Run::  python -m scripts.setup            # full native setup
           python -m scripts.setup --check    # report only, change nothing
           python -m scripts.setup --docker   # fetch data + viewer only; Docker builds the rest

Standard library ONLY, on purpose: this is the script that CREATES the virtual environment, so it
has to run under whatever Python the user happens to have, before a single dependency is installed.

The problem it solves: `checkpoints/`, `runs/`, `data/` and `config/routes/` are gitignored, so a
fresh clone has no database, no trained models and no routes. Getting from `git clone` to a working
system meant reading two READMEs, finding a releases page, downloading four zips and extracting
each into the right directory. Every one of those was a step someone could get silently wrong - and
a half-extracted bundle does not look like a missing download, it looks like a broken program: an
empty Compare tab and a controller list with no AI in it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
API = "https://api.github.com/repos/Yuvalbrs/The-Smart-Traffic-Light-Project/releases/latest"
RELEASES = "https://github.com/Yuvalbrs/The-Smart-Traffic-Light-Project/releases/latest"
IS_WIN = platform.system() == "Windows"

#: asset name -> (a path that exists once it is extracted, what breaks without it)
ASSETS = {
    "traffic-db.zip": ("data/traffic.db", "the Compare tab has no campaign to show"),
    "checkpoints.zip": ("checkpoints/lstm", "no AI controllers - only the three baselines"),
    "external-data.zip": ("data/external", "the real-measured-demand scenario cannot be rebuilt"),
    "SmartTrafficViz-win64.zip": (
        "unity/SmartTrafficViz/Build/SmartTrafficViz.exe",
        "no 3-D viewer (the dashboard still works)",
    ),
}

OK, WARN, BAD = " ok ", "warn", "MISS"


def _run(cmd: list[str]) -> tuple[bool, str]:
    """Return (usable, first line of output). Any failure at all means 'not usable'."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        out = (p.stdout or p.stderr).strip()
        return p.returncode == 0, out.splitlines()[0] if out else ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# ---------------------------------------------------------------------------------------------
# 1. the doctor
# ---------------------------------------------------------------------------------------------
def doctor() -> dict[str, bool]:
    """Report what this machine can and cannot do. Changes nothing."""
    print("\n== this machine =============================================================")
    found: dict[str, bool] = {}

    py_ok = sys.version_info >= (3, 11)
    found["python"] = py_ok
    tail = "" if py_ok else "   <- need 3.11 or newer"
    print(f"[{OK if py_ok else BAD}] Python {platform.python_version()}{tail}")

    sumo_ok, sumo_v = _run(["sumo", "--version"])
    found["sumo"] = sumo_ok
    print(f"[{OK if sumo_ok else WARN}] SUMO   {sumo_v if sumo_ok else 'not on PATH'}")
    if not sumo_ok:
        print("       not a problem: the native path needs the SUMO *binaries*, and this script")
        print("       installs the eclipse-sumo wheel (which ships them) into the venv for you.")
        print("       Docker needs nothing installed at all.")

    node_ok, node_v = _run(["node", "--version"])
    found["node"] = node_ok
    tail = "" if node_ok else "   <- needed to build the dashboard natively"
    print(f"[{OK if node_ok else WARN}] Node   {node_v if node_ok else 'not on PATH'}{tail}")

    docker_ok, _ = _run(["docker", "info"])
    found["docker"] = docker_ok
    print(f"[{OK if docker_ok else WARN}] Docker {'running' if docker_ok else 'not running'}")

    print("\n== project data ============================================================")
    for name, (marker, consequence) in ASSETS.items():
        present = (REPO / marker).exists()
        found[name] = present
        tail = "" if present else f"-> {consequence}"
        print(f"[{OK if present else BAD}] {name:26s} {tail}")

    print()
    found["venv"] = _venv_python().exists()
    print(f"[{OK if found['venv'] else BAD}] .venv{'' if found['venv'] else '  -> not created yet'}")
    dist_ok = (REPO / "frontend" / "dist" / "index.html").exists()
    found["frontend"] = dist_ok
    tail = "" if dist_ok else "  -> not built (Docker builds its own copy)"
    print(f"[{OK if dist_ok else BAD}] frontend/dist{tail}")
    return found


# ---------------------------------------------------------------------------------------------
# 2. the assets
# ---------------------------------------------------------------------------------------------
def _release() -> dict:
    req = urllib.request.Request(
        API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "smart-traffic-rl-setup"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _download(url: str, dest: Path, expect: str | None) -> None:
    """Stream to a .part file, verify, then move. A half-download must never look complete."""
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "smart-traffic-rl-setup", "Accept": "application/octet-stream"},
    )
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=300) as r, part.open("wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            h.update(chunk)
            done += len(chunk)
            pct = f"{100 * done / total:5.1f}%" if total else ""
            print(f"\r      {dest.name:28s} {done / 1e6:6.1f} MB {pct}", end="", flush=True)
    print()
    if expect and h.hexdigest() != expect:
        part.unlink(missing_ok=True)
        raise SystemExit(
            f"FATAL: {dest.name} failed its checksum.\n"
            f"       expected {expect}\n"
            f"       got      {h.hexdigest()}\n"
            "       The download was corrupted, or the release changed mid-flight. Re-run."
        )
    part.replace(dest)


def fetch_assets(force: bool = False) -> None:
    print("\n== data, models and the viewer =============================================")
    try:
        rel = _release()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"FATAL: cannot reach the GitHub releases API ({exc}).\n"
            f"       Download the four zips by hand from\n         {RELEASES}\n"
            "       and extract each one in this directory, then re-run with --check."
        ) from exc

    print(f"  release {rel.get('tag_name')}")
    by_name = {a["name"]: a for a in rel.get("assets", [])}
    cache = REPO / ".cache" / "release"
    cache.mkdir(parents=True, exist_ok=True)

    for name, (marker, _) in ASSETS.items():
        if (REPO / marker).exists() and not force:
            print(f"  [{OK}] {name:26s} already extracted")
            continue
        asset = by_name.get(name)
        if not asset:
            print(f"  [{WARN}] {name:26s} not in this release - skipping")
            continue
        digest = (asset.get("digest") or "").removeprefix("sha256:") or None
        zpath = cache / name
        if not zpath.exists() or force:
            _download(asset["browser_download_url"], zpath, digest)
        else:
            print(f"      {name:26s} using cached download")
        with zipfile.ZipFile(zpath) as z:
            # Refuse absolute paths and ..-traversal before writing a single byte.
            for m in z.namelist():
                if m.startswith(("/", "\\")) or ".." in Path(m).parts:
                    raise SystemExit(f"FATAL: {name} contains an unsafe path: {m}")
            z.extractall(REPO)
        print(f"  [{OK}] {name:26s} extracted")


# ---------------------------------------------------------------------------------------------
# 3. the native stack
# ---------------------------------------------------------------------------------------------
def _venv_python() -> Path:
    return REPO / (".venv/Scripts/python.exe" if IS_WIN else ".venv/bin/python")


def install_native(have_sumo: bool) -> None:
    print("\n== python environment ======================================================")
    py = _venv_python()
    if not py.exists():
        print("  creating .venv ...")
        subprocess.run([sys.executable, "-m", "venv", str(REPO / ".venv")], check=True)
    else:
        print("  .venv already present")

    def pip(*args: str) -> None:
        subprocess.run(
            [str(py), "-m", "pip", "install", "--disable-pip-version-check", *args],
            check=True,
            cwd=REPO,
        )

    print("  installing requirements.txt ...")
    pip("-q", "-r", "requirements.txt")
    print("  installing torch (CPU build) ...")
    pip("-q", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    if not have_sumo:
        # Ships the `sumo` and `netconvert` BINARIES, which is what sumolib.checkBinary() resolves.
        # This is the whole reason a machine with no native SUMO install can still run episodes.
        print("  no native SUMO - installing the eclipse-sumo wheel (it ships the binaries) ...")
        pip("-q", "eclipse-sumo==1.27.0")

    print("\n== dashboard ===============================================================")
    if (REPO / "frontend" / "dist" / "index.html").exists():
        print("  frontend/dist already built")
    elif shutil.which("npm"):
        print("  npm ci && npm run build ...")
        npm = "npm.cmd" if IS_WIN else "npm"
        subprocess.run([npm, "ci"], cwd=REPO / "frontend", check=True)
        subprocess.run([npm, "run", "build"], cwd=REPO / "frontend", check=True)
    else:
        print("  [warn] no npm, so the dashboard cannot be built here.")
        print("         The REST API and the 3-D viewer still work. Use Docker for the dashboard.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Set up a fresh clone of Smart-Traffic-RL.")
    ap.add_argument("--check", action="store_true", help="report only; change nothing")
    ap.add_argument(
        "--docker", action="store_true", help="fetch data + viewer only; Docker provides the rest"
    )
    ap.add_argument("--force", action="store_true", help="re-download assets even if present")
    args = ap.parse_args()

    print("Smart-Traffic-RL setup")
    found = doctor()
    if args.check:
        print("\n--check: nothing was changed.")
        return

    if not found["python"]:
        raise SystemExit("\nFATAL: Python 3.11+ is required. Install it and re-run.")

    fetch_assets(force=args.force)

    if args.docker:
        print("\n== next ====================================================================")
        print("  docker compose up --build     then open http://localhost:8000")
        print("  3-D viewer (runs natively):   unity/SmartTrafficViz/Build/SmartTrafficViz.exe")
        return

    install_native(have_sumo=found["sumo"])
    print("\n== next ====================================================================")
    print("  .\\run_app.bat" if IS_WIN else "  python run_app.py")
    print("  then open http://localhost:8000")
    print("  3-D viewer (start the hub first): unity/SmartTrafficViz/Build/SmartTrafficViz.exe")


if __name__ == "__main__":
    main()
