# Smart Traffic Intersection Management System

Final-year CS capstone (solo). A **Deep RL (plain DQN)** controller selects the next NEMA
signal phase at a **single 4-way intersection** in the **SUMO** simulator (via TraCI), with the
RL state augmented by a **frozen LSTM forecast** of near-future traffic. Evaluated against three
non-RL baselines (Webster fixed-time, max-pressure, SUMO actuated) under a multi-seed protocol.

Positioned as **replication-plus-adaptation** of MPLight (Chen et al., 2020) — not novel research.

> **Design docs live in an Obsidian vault** (the authoring source of truth); see
> [`docs/README.md`](docs/README.md). Everything the code *reads at runtime* ships here —
> `config/movements.yaml` is the movement/phase spec, and a clone is self-contained.

## How to run it

There are three ways in. Pick the first one that matches you.

---

### A. "I just want to see it work" — download the release

No Python, no build. **Windows only.**

1. Go to the [latest release](https://github.com/Yuvalbrs/The-Smart-Traffic-Light-Project/releases/latest).
2. Download **`SmartTrafficViz-win64.zip`** and extract it anywhere.
3. Run `unity/SmartTrafficViz/Build/SmartTrafficViz.exe`.

Windows will warn that the publisher is unknown — the executable is unsigned. Choose
*More info* -> *Run anyway*.

**Important:** the 3-D client is a *viewer*. On its own it draws the junction and reports
`connecting`, because the simulation itself runs in the Python hub. For traffic to appear you need
option B or C running as well.

---

### B. Run the whole thing from source (the normal way)

**You need:** Windows, Python 3.11+, Node.js, and SUMO 1.20+ with `SUMO_HOME` set.

```bash
git clone https://github.com/Yuvalbrs/The-Smart-Traffic-Light-Project.git
cd The-Smart-Traffic-Light-Project

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then add the data and the trained models. **These are not in git** — a 7 MB database and 20 MB of
model weights do not belong in a repository — so they ship as release assets. Download these from
the [latest release](https://github.com/Yuvalbrs/The-Smart-Traffic-Light-Project/releases/latest)
and extract each one **in the repository root**:

| Asset | Without it |
|---|---|
| `traffic-db.zip` | the Compare tab is empty |
| `checkpoints.zip` | no AI controllers — only Webster, max-pressure and actuated |
| `external-data.zip` | the real-measured-demand scenario cannot be rebuilt |
| `SmartTrafficViz-win64.zip` | no 3-D viewer |

Now start it:

```bash
run_app.bat
```

That builds the dashboard if needed, starts the hub, and opens the browser at
**http://localhost:8000**.

**To watch an episode:** open **Live**, choose a scene and a controller, press **RUN**. A speed of
`5x` is a good default — a full episode is one simulated hour, so `1x` really does take an hour.
Press **Reset** to stop a run and clear the junction.

When the episode ends you get a **Simulation complete** panel with that run's seven KPIs, and — on
scenes SCN-01 to SCN-05 — a button to compare it against the recorded campaign.

**For the 3-D view:** with the hub running, launch
`unity/SmartTrafficViz/Build/SmartTrafficViz.exe`. It connects to the hub automatically.

---

### C. Run the backend in Docker

Everything except the 3-D client, in one command. Needs Docker Desktop running.

```bash
docker compose up --build
```

Then open **http://localhost:8000**. First build takes 5-15 minutes; after that it is seconds.
Full details in [`README-docker.md`](README-docker.md).

---

### One rule: never run B and C at the same time

Both write the same `data/traffic.db`, and on Windows the Docker bind mount cannot share a SQLite
file with the host. If you open the database from the host while the container is running, the
container's handle goes stale and **every** later write fails — episodes still run and still
stream, they simply never get recorded, and they vanish from the archive.

Use `run_app.bat` **or** `docker compose up`. If writes have already started failing,
`docker compose restart hub` fixes it; nothing is corrupted.

---

### Quick reference

```bash
run_app.bat                                        # everything, one double-click
docker compose up --build                          # the backend, containerised

LIBSUMO_AS_TRACI=1 .venv/Scripts/python -m pytest -q          # 432 tests, ~90 s
.venv/Scripts/python -m scripts.train_dqn --seed 42 --variant plain --run-dir runs/mine
```

`LIBSUMO_AS_TRACI=1` matters: without it SUMO falls back to a socket client that is roughly ten
times slower.

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
tests/           295 tests
```

## Requirements

- **Python 3.11+**
- **SUMO 1.20+** — the *binaries*, not only the Python bindings. `netconvert` really does run
  on every live session start (`src/api/live.py` calls `build_net`) and `sumo` is resolved
  through `sumolib.checkBinary`, so `pip install libsumo` alone is **not** enough.
  Either of these works:
  - a native install, with `SUMO_HOME` set and the binaries on `PATH` (verified locally:
    SUMO 1.27.0, `SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo`); or
  - `pip install eclipse-sumo==1.27.0`, which ships `sumo` and `netconvert` itself. That is
    what the Docker image uses, and it runs full episodes with no native SUMO installed.
- **Node.js 18+** — the dashboard is built from source; `frontend/dist/` is not committed.
- **torch** — deliberately absent from `requirements.txt` so the CPU/CUDA choice stays yours.
  See [GPU / torch](#gpu--torch-do-this-deliberately-later). Required by every DQN controller.

### What a fresh clone does *not* include

`checkpoints/`, `runs/`, `data/traffic.db` and `data/lstm/` are gitignored — they are large and
reproducible. So a clone runs the **three baselines** immediately, and for the DQN controllers you
either train one from the Train tab or fetch the checkpoint bundle from the GitHub release. The
Compare tab needs `data/traffic.db`; without it, it says so rather than showing an empty table.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### GPU / torch (do this deliberately, later)

`torch` is intentionally **not** in `requirements.txt`. T-00-01 does not need it, and the CUDA
build must be installed with the correct index URL for the GPU (an NVIDIA GPU is present on the dev
machine). Install it when the ML phase starts, e.g.:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

