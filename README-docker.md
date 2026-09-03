# Running Smart-Traffic-RL with Docker

One command builds the React dashboard, installs Python and SUMO, generates the route files, and
starts the hub:

```bash
docker compose up --build
```

Then open **http://localhost:8000**.

That is the dashboard, the REST API and both 1 Hz WebSocket channels — the whole backend. First
build takes 5–15 minutes (PyTorch and the SUMO wheel dominate); afterwards it is seconds.

To stop: `Ctrl-C`, or `docker compose down` from another terminal.

---

## What is in the container, and what is not

| Piece | Where it runs | Why |
|---|---|---|
| FastAPI hub (REST + 2 WebSocket channels) | **container** | |
| SUMO + TraCI (via `libsumo`, in-process) | **container** | `libsumo` ships Linux wheels |
| React dashboard | **container** | built in stage 1, served by the hub itself |
| SQLite database, traces, checkpoints | **bind-mounted volumes** | see below |
| Unity 3-D client | **natively on Windows** | see below |

### Why the database is not its own container

This project embeds **SQLite**. It is a *file* opened in-process by the hub, not a server that
listens on a port — so unlike MongoDB or MySQL there is no process to containerise. `data/` is
mounted through as a volume instead, which is why a campaign you ran on the host is still there
after a rebuild.

Turning it into a real database server would mean porting the WAL and per-connection PRAGMA layer
in `src/db/engine.py` to Postgres, and rewriting the parts of the analysis that assume a single
writer. That is a project, not a Dockerfile.

### Why the Unity client is not a container

It is a **Windows GUI binary**; these containers are Linux and headless. Run it natively:

```
unity/SmartTrafficViz/Build/SmartTrafficViz.exe
```

It connects to `ws://127.0.0.1:8000/ws/unity`, which compose publishes — so it talks to the
container with no reconfiguration at all. Start the stack first, then the viewer.

This is the same shape as the containerised exercise this was modelled on: the servers and the
datastore were containers, and the client application was not.

---

## The database starts EMPTY

A fresh container creates the schema but has no data, so the **Compare tab will be blank**. The
recorded campaign is a release asset, not source — it is far too large for git:

```bash
# from the repository root, before `docker compose up`
setup.bat --docker
```

That fetches the four release assets from the *latest* release, verifies each against the
checksum GitHub publishes for it, and extracts them exactly where the bind mounts expect. It
installs no Python packages and no SUMO — the container provides those — so it is only the data
step. Re-running it is a no-op.

Pinning the version by hand instead is still fine; the assets are ordinary zips extracted at the
repository root. Without `checkpoints.zip` the DQN controllers cannot load and only Webster,
max-pressure and actuated will run.

---

## What happens on first start

`docker/entrypoint.sh` brings a cold container up to a runnable state. Every step is idempotent and
none of them overwrites data that already exists:

1. **Verifies `sumo` actually runs.** `src/env/sumo_env.py` resolves it through
   `sumolib.checkBinary()`, so a missing binary would otherwise surface as a confusing failure in
   the middle of the first episode instead of one clear line at startup.
2. **Generates `config/routes/*.rou.xml` if absent** (~1 min, deterministic). These are gitignored
   build products, so a fresh clone has none — and the hub starts happily without them and then
   fails on the first episode.
3. **Creates or migrates the database.** `scripts/init_db.py` adds missing tables *and* any columns
   the models have grown, so running it against a populated database is a no-op.

---

## Common problems

**Port 8000 already in use.** Something else is on it — often a native `uvicorn` from `run_app.bat`,
or another project's container (`docker ps` will show it). No file edit needed:

```bash
HUB_PORT=8080 docker compose up      # -> http://localhost:8080
```

Only the host side moves; the container still listens on 8000. Note the Unity client hard-codes
`127.0.0.1:8000`, so if you remap it will sit on `connecting`.

**Compare tab is empty.** Expected on a fresh database — fetch `traffic-db.zip` above.

**No AI controllers in the dropdown.** `checkpoints/` is empty; fetch `checkpoints.zip`.

**Build is slow the first time.** PyTorch CPU is a few hundred MB. It is cached afterwards; only
`docker compose build --no-cache` pays it again.

---

## Running the tests inside the container

```bash
docker compose run --rm --entrypoint pytest hub -q
```

```
427 passed, 5 skipped
```

The image ships `tests/` and `golden_hashes.json`, so this includes the two reproducibility tests
that hash a reference run against the committed golden file. They pass **inside the container** —
which is the claim worth making: the environment reproduces the recorded results, not merely the
developer machine that produced them.

`--entrypoint pytest` bypasses the boot script, which would otherwise start the hub instead.

---

## Why the image is built the way it is

**Base images are pinned by digest, not by tag.** `python:3.11-slim` is a moving target that will
be a different image next month. This repository pre-registers its hypotheses and stamps `git_sha`
and `sumo_version` onto every run — an environment that silently changes underneath it would break
that contract in the one place nobody thinks to look.

**Python dependencies come from `requirements-docker.lock`, not `requirements.txt`.** The latter is
loosely pinned on purpose; it is the developer-facing list. The lock is a `pip freeze` taken inside
a verified-working container, so the image resolves the same 96 packages every time. Regenerate it
with:

```bash
docker compose run --rm --entrypoint bash hub -c 'pip freeze --all'
```

**`torch` is pinned to `2.12.1+cpu` to match the machine that produced the campaign.** Left to
resolve freely the container picked up `2.14.0`. The checkpoints load either way, but re-running an
evaluation under a different torch than the published results were generated with is precisely the
silent provenance drift this project exists to prevent.

**The build imports the native extensions and fails if they don't load.** `libsumo`, `traci`,
`sumolib` and `torch` are imported, and `sumo --version` is executed, as a build step. The
libXrender bug produced a perfectly green build that died on startup; that class of fault can no
longer leave the builder.

**The container runs as uid 10001, not root.** The entrypoint checks the bind mounts are writable
by that uid first, so a permissions problem is one actionable line rather than a stack trace
part-way through an episode.

**`npm ci`, with no `|| npm install` fallback.** The value of `ci` is that it fails loudly when the
lockfile and `package.json` disagree; a fallback would quietly build the dashboard from a different
dependency tree than the one committed.

**`.gitattributes` forces LF on shell scripts.** A CRLF `entrypoint.sh` is not untidy, it is
unrunnable: the kernel reads the shebang literally and reports `env: 'bash\r': No such file or
directory`, exit 127, with no other clue. This repo is developed on Windows, so without that file a
fresh clone reintroduces the bug every time.

### Size

3.03 GB, and roughly 1.5 GB of that is irreducible: torch is 773 MB and the SUMO wheels are ~715 MB
across four directories. `--no-compile` removes ~12,000 `.pyc` files that `PYTHONDONTWRITEBYTECODE`
means the runtime never reads anyway. The remainder is scipy, pandas, matplotlib and the notebook
stack, kept so the analysis scripts and the full test suite run in the same image as the service.

---

## Do NOT run the native hub and the container at the same time

Both write `data/traffic.db`, and on Docker Desktop for Windows the bind mount cannot share a
SQLite file safely between the host and a running container. Opening the database from the host —
`run_app.bat`, a script, even a one-line `sqlite3` query — can invalidate the container's open
handle. The container then fails **every** subsequent write with:

```
unable to open database file
```

It fails *silently* from the user's point of view: episodes still run to completion and still
stream to the clients, they simply never reach the database, so the run vanishes from the archive
and the end-of-episode summary reports that no such run exists.

**Rules:**

- Use **one** of them at a time — `run_app.bat` **or** `docker compose up`, never both.
- To query the database while the container is up, go through the container:
  ```bash
  docker compose exec hub python -c "import sqlite3; ..."
  ```
- If writes have already started failing, `docker compose restart hub` restores them. Nothing is
  corrupted; the handle is simply stale.

For the demo itself, prefer `run_app.bat`. Docker is the reproducible-environment story, not the
lowest-risk path for a live presentation.
