#!/usr/bin/env bash
# Bring a cold container up to a runnable state, then hand over to the hub.
#
# Everything here is IDEMPOTENT and only fills in what is missing. The route files and the database
# are gitignored build products, so a fresh clone (and therefore a fresh image) has neither — and
# without them the hub starts fine and then fails on the first episode, which is the worst possible
# time to find out. Generating them at boot is what makes `docker compose up` actually mean "it
# runs", rather than "it started".
#
# It never overwrites what is already on the volume: a mounted data/ with a real campaign in it
# must survive a rebuild untouched.

set -euo pipefail

say() { printf '\n[entrypoint] %s\n' "$*"; }

# --- 1. SUMO must be genuinely present, not merely pip-installed --------------------------------
# checkBinary() resolves through PATH, so proving the binary answers here turns a confusing
# mid-episode failure into a clear one-line failure at startup.
if ! sumo --version >/dev/null 2>&1; then
    say "FATAL: the 'sumo' binary is not runnable. src/env/sumo_env.py needs it via checkBinary()."
    exit 1
fi
say "SUMO: $(sumo --version 2>/dev/null | head -1)"

# --- 1b. The bind mounts must be WRITABLE by this user -------------------------------------------
# The container runs as uid 10001, not root. Docker Desktop on Windows normally makes bind mounts
# world-writable, but on a Linux host they arrive owned by the host user and the hub would fail on
# its first write - creating the database, or opening a JSONL trace mid-episode. Checking here
# turns that into one actionable line instead of a stack trace forty minutes into a demo.
for d in data runs checkpoints config/routes; do
    mkdir -p "$d" 2>/dev/null || true
    if [ ! -w "$d" ]; then
        say "FATAL: $d is not writable by uid $(id -u)."
        say "       Fix on the host with:  sudo chown -R 10001:10001 $d"
        exit 1
    fi
done
say "bind mounts writable as uid $(id -u)"

# --- 2. Route files -----------------------------------------------------------------------------
# config/routes/ is gitignored (built deterministically by scripts/build_routes.py), so it is
# normal for it to be empty on a first run and normal for it to be full on every run after.
shopt -s nullglob
routes=(config/routes/*.rou.xml)
shopt -u nullglob

if [ ${#routes[@]} -eq 0 ]; then
    say "no route files — generating them (deterministic, ~1 min)"
    python -m scripts.build_routes
    say "routes built"
else
    say "routes present (${#routes[@]} files) — leaving them alone"
fi

# --- 3. The database ----------------------------------------------------------------------------
# init_db is itself idempotent: it creates missing tables AND adds columns the models grew since
# the file was made, so running it on a populated database is a no-op rather than a risk.
if [ ! -f data/traffic.db ]; then
    say "no data/traffic.db — creating an empty schema"
    mkdir -p data
    python -m scripts.init_db
    say "database created (EMPTY: the Compare tab needs a real campaign, or mount the release's traffic.db)"
else
    say "database present — applying any missing columns (idempotent)"
    python -m scripts.init_db
fi

# --- 4. Say plainly what this container is not ---------------------------------------------------
if [ ! -d frontend/dist ]; then
    say "WARNING: frontend/dist missing — the API works, but there is no dashboard to open."
fi
say "the Unity 3-D client is NOT in this container (Windows GUI binary); run it natively"
say "     against http://localhost:8000 — see README-docker.md"

say "starting: $*"
exec "$@"
