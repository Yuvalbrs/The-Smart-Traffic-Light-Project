# Smart Traffic Intersection Management System

Final-year CS capstone (solo). A **Deep RL (plain DQN)** controller selects the next NEMA
signal phase at a **single 4-way intersection** in the **SUMO** simulator (via TraCI), with the
RL state augmented by a **frozen LSTM forecast** of near-future traffic. Evaluated against three
non-RL baselines (Webster fixed-time, max-pressure, SUMO actuated) under a multi-seed protocol.

Positioned as **replication-plus-adaptation** of MPLight (Chen et al., 2020) — not novel research.

> **Design docs live in an Obsidian vault** (the authoring source of truth); see
> [`docs/README.md`](docs/README.md). Everything the code *reads at runtime* ships here —
> `config/movements.yaml` is the movement/phase spec, and a clone is self-contained.

## Run it

```bash
run_app.bat            # builds the UI if needed, starts the hub, opens the dashboard
```

Then open the **Live** tab and press *start session*. Three tabs: **Live** (watch an episode
stream at 1 Hz), **Train** (train a controller and watch its reward curve), **Compare**
(every controller's KPIs on shared seeds). A Unity 3-D client renders the same live episode —
see [`unity/README.md`](unity/README.md).

Prerequisites: Python 3.11+, SUMO 1.20+ with `SUMO_HOME` set, Node.js, and `torch`
(installed separately — see [Setup](#setup)). Without the trained checkpoints (gitignored, see
below) the three **baseline** controllers still run; the DQN controllers need either the
checkpoint download or a training run from the Train tab.

## Status

**Complete.** Simulator, DQN training stack, frozen-LSTM forecaster, three baselines, evaluation
campaign, statistical analysis, FastAPI hub, React dashboard and Unity 3-D client are all built
and tested (**390 tests**).

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
- **SUMO 1.20+** with `SUMO_HOME` set and `sumo`/`sumo-gui`/`netconvert` on `PATH`
  (verified locally: SUMO 1.27.0, `SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo`).
  A native install, not a pip package: `netconvert` runs on every session start and has no
  wheel. `pip install libsumo` alone is not enough.
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

