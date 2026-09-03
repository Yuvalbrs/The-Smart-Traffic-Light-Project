"""Build the four release assets, reproducibly, from the current working tree.

    Run::  .venv/Scripts/python -m scripts.make_release_assets

Why this is a script and not four `zip` commands typed by hand:

  * The v1.0.0 assets were built ad hoc and went stale silently. The Unity viewer in that release
    predated 22 commits to `unity/`, and nothing about the release page said so. A script that
    stamps the commit into the bundle makes "which build is this?" answerable from the zip itself.

  * `data/traffic.db` is opened in WAL mode. A committed transaction lives in `traffic.db-wal`
    until a checkpoint folds it into the main file, so zipping `traffic.db` on its own can ship a
    database that is missing the most recent work with no error anywhere. That is exactly what
    happened to v1.0.0's `traffic-db.zip`. This script REFUSES to package an un-checkpointed
    database rather than trusting whoever runs it to remember.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIST = REPO / "dist"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _guard_database() -> Path:
    """Fail loudly if the database has un-checkpointed WAL content, or is not readable."""
    db = REPO / "data" / "traffic.db"
    if not db.exists():
        sys.exit("FATAL: data/traffic.db does not exist - nothing to package.")

    wal = db.with_name(db.name + "-wal")
    wal_size = wal.stat().st_size if wal.exists() else 0
    if wal_size > 0:
        sys.exit(
            f"FATAL: {wal.name} is {wal_size} bytes - the database has committed data that is NOT\n"
            f"       yet in {db.name}, and packaging it now would ship an incomplete campaign.\n"
            f"       Fold it in first (nothing may hold the database open):\n"
            f'         python -c "import sqlite3;c=sqlite3.connect(r\'{db}\');'
            f"c.execute('PRAGMA wal_checkpoint(TRUNCATE)')\""
        )

    import sqlite3

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            sys.exit("FATAL: data/traffic.db fails its integrity check.")
        episodes = con.execute("SELECT COUNT(*) FROM episode").fetchone()[0]
        campaign = con.execute(
            "SELECT COUNT(*) FROM episode e JOIN experiment_run r ON e.run_id_fk = r.id"
            " WHERE r.git_sha = 'e1b8d42'"
        ).fetchone()[0]
    finally:
        con.close()

    if campaign != 900:
        sys.exit(
            f"FATAL: the confirmatory campaign e1b8d42 has {campaign} episodes, expected 900.\n"
            "       Shipping a partial campaign would make the Compare screen disagree with the"
            " report."
        )
    print(f"  database OK: {episodes} episodes total, {campaign} in campaign e1b8d42, WAL empty")
    return db


def _write_zip(name: str, members: list[tuple[Path, str]], stamp: str) -> Path:
    DIST.mkdir(exist_ok=True)
    out = DIST / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in members:
            z.write(src, arc)
        z.writestr("BUILD_INFO.txt", stamp)
    return out


def _walk(root: Path, prefix: str, skip_hidden: bool = True) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if skip_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            p = Path(dirpath) / fn
            out.append((p, f"{prefix}/{p.relative_to(root).as_posix()}"))
    return out


def main() -> None:
    sha = _git_sha()
    stamp = f"Smart-Traffic-RL release asset\nbuilt from commit: {sha}\n"
    print(f"packaging from commit {sha[:7]}\n")

    # --- 1. the database -------------------------------------------------------------------
    print("traffic-db.zip")
    db = _guard_database()
    db_zip = _write_zip("traffic-db.zip", [(db, "data/traffic.db")], stamp)

    # --- 2. models -------------------------------------------------------------------------
    # Everything the hub lists at GET /models, uncurated. A curated bundle would make a fresh
    # machine behave differently from the one the results came off, which defeats the point.
    # Leading-underscore directories are the repo's quarantine convention and stay out.
    print("checkpoints.zip")
    members: list[tuple[Path, str]] = []
    runs = REPO / "runs"
    n_runs = 0
    for run in sorted(runs.iterdir()) if runs.is_dir() else []:
        if not run.is_dir() or run.name.startswith("_"):
            continue
        if not (run / "checkpoints").is_dir():
            continue
        n_runs += 1
        # The LAST ep*.pt and best.pt only. catalog.py resolves a model as
        # `sorted(glob("ep*.pt"))[-1]`, so the intermediate epochs are never loaded by the
        # application - they exist for training-curve forensics, and metrics.csv already carries
        # that curve. Shipping all 312 of them made this asset 89.6 MB instead of 20.5 MB for a
        # bundle the hub cannot tell apart.
        eps = sorted((run / "checkpoints").glob("ep*.pt"), key=lambda p: int(p.stem[2:]))
        wanted = [eps[-1]] if eps else []
        best = run / "checkpoints" / "best.pt"
        if best.exists():
            wanted.append(best)
        for pt in wanted:
            members.append((pt, f"runs/{run.name}/checkpoints/{pt.name}"))
        for extra in ("config.yaml", "metrics.csv"):
            if (run / extra).exists():
                members.append((run / extra, f"runs/{run.name}/{extra}"))
    lstm = REPO / "checkpoints" / "lstm"
    if lstm.is_dir():
        members += _walk(lstm, "checkpoints/lstm")
    print(f"  {n_runs} runs, {len(members)} files")
    ck_zip = _write_zip("checkpoints.zip", members, stamp)

    # --- 3. the Unity viewer ---------------------------------------------------------------
    print("SmartTrafficViz-win64.zip")
    build = REPO / "unity" / "SmartTrafficViz" / "Build"
    if not (build / "SmartTrafficViz.exe").exists():
        sys.exit("FATAL: no Unity build. Build it first (see README).")
    viz_zip = _write_zip(
        "SmartTrafficViz-win64.zip",
        _walk(build, "unity/SmartTrafficViz/Build"),
        stamp,
    )

    # --- 4. external data ------------------------------------------------------------------
    print("external-data.zip")
    ext = REPO / "data" / "external"
    ext_zip = _write_zip("external-data.zip", _walk(ext, "data/external"), stamp)

    print("\n  asset                        size      sha256")
    for z in (viz_zip, ck_zip, db_zip, ext_zip):
        print(f"  {z.name:28s} {z.stat().st_size/1e6:7.1f}MB  {_sha256(z)}")
    print(f"\nwrote {DIST}")


if __name__ == "__main__":
    main()
