# Smart Traffic Intersection Management System

Final-year CS capstone (solo). A **Deep RL (plain DQN)** controller selects the next NEMA
signal phase at a **single 4-way intersection** in the **SUMO** simulator (via TraCI), with the
RL state augmented by a **frozen LSTM forecast** of near-future traffic. Evaluated against three
non-RL baselines (Webster fixed-time, max-pressure, SUMO actuated) under a multi-seed protocol.

Positioned as **replication-plus-adaptation** of MPLight (Chen et al., 2020) — not novel research.

> **The blueprint/spec is NOT in this repo.** It lives in the Obsidian vault (the single source of
> truth). See [`docs/README.md`](docs/README.md) for where to find it. Code conforms to the spec.

## Status

Skeleton scaffolded. First implementation task is **T-00-01** (the trivial end-to-end SUMO↔Python
loop) — not yet started. Nothing else begins until it passes.

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

## Layout

```
smart-traffic-rl/
├── CLAUDE.md          # permanent project rules (read every session)
├── README.md
├── requirements.txt
├── pyproject.toml     # black (line 100), ruff, pytest
├── docs/              # signpost only -> spec lives in the Obsidian vault
├── config/            # SUMO net/route/cfg + scenario YAMLs
├── scripts/           # entry points (e.g. python -m scripts.vertical_slice)
├── src/
│   └── env/           # SUMO env + TraCI bridge (built per task, not yet)
└── tests/
```

More directories (`src/ml`, `src/baselines`, `src/data`, `src/api`, `src/metrics`, `frontend/`,
`unity/`) are added as later tasks reach them — see the master task list in the vault.
