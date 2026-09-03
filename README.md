# Smart Traffic Intersection Management System

Final-year CS capstone (solo). A **Deep RL (plain DQN)** controller selects the next 
signal phase at a **single 4-way intersection** in the **SUMO** simulator (via TraCI), with the
RL state augmented by a **LSTM forecast** of near-future traffic. Evaluated against three
non-RL baselines (Webster fixed-time, max-pressure, SUMO actuated) under a multi-seed protocol.



## How to run it

Windows is the primary target; **[Linux](#on-linux) works too, minus the 3-D viewer.**
Install the three prerequisites below, then three commands.

### 0. Prerequisites

`setup.bat` installs SUMO and torch for you. It cannot install the toolchain it runs *on*:

```powershell
winget install --id Python.Python.3.11 --exact --source winget
winget install --id Git.Git --exact --source winget
winget install --id OpenJS.NodeJS.LTS --exact --source winget
```

Then **close the terminal and open a new one** - PATH changes do not reach a window that was
already open - and check that all three answer:

```powershell
py --version      # 3.11 or newer
git --version
node --version    # 18 or newer
```

No `winget`? Install the same three by hand: [Python](https://www.python.org/downloads/) - tick
**"Add python.exe to PATH"** in the installer, and avoid the Microsoft Store build, which answers
to `python` and does nothing useful - then [Git](https://git-scm.com/download/win) and
[Node.js LTS](https://nodejs.org).

**`venv` needs no separate install.** It is part of the Python standard library and the Windows
installer ships it; `setup.bat` creates `.venv` itself. (The habit of installing it comes from
Linux, where `python3-venv` really is a separate package.)

**Node.js is not optional if you want the dashboard.** `frontend/dist/` is not committed, so with
no `npm` on the machine there is nothing to serve at `/`: the REST API and the 3-D viewer still
work, but the Live and Compare tabs do not exist. Docker builds its own copy and needs none of
this.

### 1. Get it

```bash
git clone https://github.com/Yuvalbrs/The-Smart-Traffic-Light-Project.git
cd The-Smart-Traffic-Light-Project
```

### 2. Set it up

```bash
setup.bat
```

One command, and it is safe to re-run — every step is skipped if it is already done.

It checks the machine and tells you what it found, then downloads the four release assets
(~67 MB, checksum-verified) and extracts them where they belong, creates `.venv`, installs the
requirements and the CPU build of torch, **installs SUMO into the venv if you do not already have
it**, and builds the dashboard.

Why anything needs downloading at all: `data/`, `runs/`, `checkpoints/` and `config/routes/` are
gitignored — a 7 MB database and the trained weights do not belong in a repository — so they ship
as release assets instead. A clone without them is not broken, it is empty, and the two look
alike:

| Asset | Without it |
|---|---|
| `traffic-db.zip` | the Compare tab has no campaign to show |
| `checkpoints.zip` | no AI controllers — only Webster, max-pressure and actuated |
| `external-data.zip` | the real-measured-demand scenario cannot be rebuilt |
| `SmartTrafficViz-win64.zip` | no 3-D viewer (the dashboard still works) |

To see what your machine has without changing anything: `setup.bat --check`. That is also the
first thing to run when something does not work.

### 3. Start it

```bash
run_app.bat
```

That builds the dashboard if needed, starts the hub, and opens **http://localhost:8000**.

Open **Live**, pick a scene and a controller, press **RUN**. Use `5x` speed — one episode is a
full simulated hour, so `1x` really does take an hour. **Reset** stops a run and clears the
junction.

When the episode ends you get a **Simulation complete** panel with that run's seven KPIs, and on
scenes SCN-01 to SCN-05 a button to compare it against the recorded campaign.

### 4. The 3-D viewer

With the hub already running:

```
unity\SmartTrafficViz\Build\SmartTrafficViz.exe
```

It connects to the hub by itself. Windows will warn that the publisher is unknown — the executable
is unsigned; choose *More info* → *Run anyway*.

The viewer is a **client**. Started on its own it draws the junction and reports `connecting`,
because the simulation runs in the hub, not in the viewer.

---

### On Linux

Everything runs except the 3-D viewer, which is a Windows binary
(`src/api/viewer.py` resolves `SmartTrafficViz.exe`, and the only Unity build target is
`BuildWindows`). The dashboard, the hub, training and the full test suite are unaffected, and the
Compare tab - the one that shows the campaign - is part of the dashboard, not the viewer.

**0. Prerequisites.** The project needs **Python 3.11 or newer**, and this is the step that bites:
Ubuntu 22.04 ships 3.10, which is too old. Check first:

```bash
python3 --version
```

Debian / Ubuntu 24.04+ (already 3.12):

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git nodejs npm
```

Ubuntu 22.04 or older, where `python3` is 3.10:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv git nodejs npm
```

Fedora: `sudo dnf install -y python3.11 git nodejs npm`. Arch: `sudo pacman -S python git nodejs npm`.

`python3-venv` **is** a real package here - unlike on Windows, Debian and Ubuntu split it out of the
standard library, and `setup.bat`'s job of creating `.venv` fails without it.

**1-2. Get it and set it up.** There is no `setup.sh`; `setup.bat` is only a launcher, and the
script it launches is portable:

```bash
git clone https://github.com/Yuvalbrs/The-Smart-Traffic-Light-Project.git
cd The-Smart-Traffic-Light-Project
python3.11 -m scripts.setup --check     # or python3, if that is already 3.11+
python3.11 -m scripts.setup
```

It creates `.venv`, installs the requirements and CPU torch, installs the `eclipse-sumo` wheel if
no native SUMO is on `PATH`, and builds the dashboard - the same as on Windows.

**3. Start it.**

```bash
.venv/bin/python run_app.py
```

Then open <http://localhost:8000>. Some of the launcher's *error* messages still quote Windows
paths (`.venv\Scripts\...`); the equivalent is always `.venv/bin/`.

**Or use Docker, which is the better path here.** Docker runs natively on Linux rather than in a
VM, and the SQLite bind-mount hazard described below is a Docker-Desktop-on-Windows problem that
does not apply:

```bash
python3.11 -m scripts.setup --docker
GIT_SHA=$(git rev-parse HEAD) docker compose up --build
```

Still do not run `run_app.py` and the container at the same time - two processes writing one
SQLite file is a bad idea on any OS.

---

### Instead of steps 2–3: Docker

If you would rather install nothing at all — no Python packages, no SUMO, no Node — and you have
Docker Desktop running, the backend comes up in one command:

```bash
setup.bat --docker          # fetches the data and the viewer only
docker compose up --build   # then open http://localhost:8000
```

The first build takes 5–15 minutes; after that it is seconds. The 3-D viewer is a Windows binary,
so it still runs natively (step 4) and still connects to `localhost:8000`. Details in
[`README-docker.md`](README-docker.md).

**Do not run both.** `run_app.bat` and the container both write `data/traffic.db`, and on Windows
a Docker bind mount cannot share a SQLite file with the host. Use one or the other. `run_app.bat`
now refuses to start if a hub is already answering on the port, so the accident is hard to have —
but if it has already happened, `docker compose restart hub` fixes it and nothing is corrupted.

The failure it prevents is worth knowing about, because it is silent: the container's handle goes
stale and every later write is discarded, so episodes run, stream to both clients, and are simply
never recorded.

### Other commands

```bash
LIBSUMO_AS_TRACI=1 .venv/Scripts/python -m pytest -q      # 432 tests, ~90 s
.venv/Scripts/python -m scripts.train_dqn --seed 42 --variant plain --run-dir runs/mine
```

`LIBSUMO_AS_TRACI=1` matters: without it SUMO falls back to a socket client roughly ten times
slower.

## Status

**Complete.** Simulator, DQN training stack, frozen-LSTM forecaster, three baselines, evaluation
campaign, statistical analysis, FastAPI hub, React dashboard and Unity 3-D client are all built
and tested (**432 tests**).

### Results

The confirmatory campaign is **900 episodes** — 5 scenarios x 15 held-out seeds x 12 controllers —
analysed exactly as pre-registered (paired Wilcoxon signed-rank, Holm-Bonferroni within families).
On the test scenario SCN-05, average waiting time:

| comparison | effect | Holm p | |
|---|---|---|---|
| vs Webster fixed-time | **-15.6 s (-44%)** | 0.0013 | RL wins |
| vs SUMO actuated | **-4.3 s (-17%)** | 0.0013 | RL wins |
| vs max-pressure | +2.0 s | 0.0013 | RL loses |

Gridlock is **0%** across all five scenarios.

**The headline hypothesis was refuted, and that is the finding.** H2 asked whether adding the
frozen LSTM forecast to the agent's state improves it. Pre-registered before any of this data
existed, the answer is that it makes the agent significantly *worse* (avg wait +0.62 s, p=0.004;
p95 wait +2.0 s, p=0.041). H3's control settles why: a **random** forecast beats the real one,
so the loss is attributable to the forecast information itself rather than the extra input
capacity — which matches the forecaster's own standalone skill of **-0.31** at the 90 s horizon
(worse than assuming queues do not change).

Everything above is reproducible from `data/eval/analysis/`. The pre-registration, its six dated
amendments, and the full defect history are in the vault's `preregistration.md` and `decisions.md`.

## Layout

```
src/env/         SUMO gym environment, the 12-movement / 8-phase model, safety masking
src/ml/          DQN agent, training loop, replay buffer, frozen LSTM forecaster + wrapper
src/baselines/   Webster fixed-time, max-pressure, SUMO actuated
src/metrics/     KPI extraction from trip-info + traces
src/api/         FastAPI hub: live sessions, WebSocket fan-out, REST replay
src/db/          SQLite (WAL) results schema
src/trace/       JSONL frame writer + the sim_frame contract
src/provenance/  version hashing; the single source of truth for the deployed forecaster
src/repro/       the golden-hash reproducibility gate
src/scenarios/   scenario definitions and loading
scripts/         network/route builders, training, evaluation, analysis (each has a `Run::` docstring)
frontend/        React + Vite dashboard (`npm run dev` -> :5173)
unity/           Unity 6 3-D client (see `unity/README.md`)
tests/           432 tests
```

## Requirements

**You need Python 3.11+, git and Node.js** (see [Prerequisites](#0-prerequisites) for the
one-line installs). **`setup.bat` handles everything else** — including SUMO, which is the
one that used to catch people out. Run `setup.bat --check` to see where you stand.

What it installs for you, and why each is needed:

- **SUMO 1.20+** — the *binaries*, not only the Python bindings. `netconvert` really does run on
  every live session start (`src/api/live.py` calls `build_net`) and `sumo` is resolved through
  `sumolib.checkBinary`, so `pip install libsumo` alone is **not** enough. Either works:
  - a native install, with `SUMO_HOME` set and the binaries on `PATH` (that is how the dev
    machine is set up: SUMO 1.27.0, `SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo`); or
  - `pip install eclipse-sumo==1.27.0`, which ships `sumo` and `netconvert` inside the wheel.
    `setup.bat` installs this into `.venv` when it finds no native SUMO, and `run_app.py` will
    then locate `SUMO_HOME` from the wheel by itself. This is also what the Docker image uses,
    and it runs full episodes with no native SUMO installed at all.
- **Node.js 18+** — only to build the dashboard from source; `frontend/dist/` is not committed.
  Without it the REST API and the 3-D viewer still work, and Docker builds its own copy anyway.
- **torch** — deliberately absent from `requirements.txt` so the CPU/CUDA choice stays yours;
  `setup.bat` installs the CPU build. See [GPU / torch](#gpu--torch). Required by every DQN
  controller.

### What a fresh clone does *not* include

`checkpoints/`, `runs/`, `data/traffic.db`, `data/lstm/` and `config/routes/` are gitignored —
they are large, and reproducible. A clone therefore runs the **three baselines** immediately and
nothing else. `setup.bat` fetches the rest; failing that, you can train a DQN from the Train tab,
and the Compare tab says it has no data rather than showing an empty table.

## GPU / torch

`torch` is intentionally **not** in `requirements.txt`. T-00-01 does not need it, and the CUDA
build must be installed with the correct index URL for the GPU (an NVIDIA GPU is present on the dev
machine). Install it when the ML phase starts, e.g.:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

