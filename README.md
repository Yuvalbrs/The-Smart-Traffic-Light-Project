# Smart Traffic Intersection Management System

Final-year CS capstone (solo). A **Deep RL (plain DQN)** controller selects the next NEMA
signal phase at a **single 4-way intersection** in the **SUMO** simulator (via TraCI), with the
RL state augmented by a **frozen LSTM forecast** of near-future traffic. Evaluated against three
non-RL baselines (Webster fixed-time, max-pressure, SUMO actuated) under a multi-seed protocol.

Positioned as **replication-plus-adaptation** of MPLight (Chen et al., 2020) — not novel research.

> **The blueprint/spec is NOT in this repo.** It lives in the Obsidian vault (the single source of
> truth). See [`docs/README.md`](docs/README.md) for where to find it. Code conforms to the spec.

## Status

**Implementation complete; results are being re-run.** The simulator, the DQN training stack,
the evaluation campaign, the FastAPI hub, the React dashboard and the Unity 3-D client are all
built and tested (295 tests). What is *not* current is the numbers.

On 2026-08-28 a 3-D rendering of the simulation exposed a right-of-way defect. Pulling that
thread uncovered four independent modelling defects plus a safety bound that was specified but
never enforced — each invisible to the test suite, and together enough to invalidate every
result the project had produced. All are fixed and regression-tested; the training matrix,
evaluation campaign and statistical analysis are being re-run on the corrected environment.

Do not cite any number from `data/eval/` or the analysis notebook until that re-run lands. The
full account, with evidence and the remaining work, is in the vault's `decisions.md` (entries
dated 2026-08-28) and `hot.md`.

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
- **SUMO 1.20+** with `SUMO_HOME` set and `sumo`/`sumo-gui` on `PATH`
  (verified locally: SUMO 1.27.0, `SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo`).

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

