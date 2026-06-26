"""Forecaster-rescue #1a - Collect DQN-induced traces for LSTM bootstrap fine-tuning.

The LSTM was trained on WEBSTER traces but at RL inference it forecasts on DQN-induced states
(lstm-forecasting.md "Distribution shift" - the known limitation). This runs the trained HYBRID
DQN (greedy, seed-123 model - the non-gridlock-prone one) through the env on the bootstrap
scenarios, recording the SAME per-step queue/count CSVs as T-01-05 but from the DQN's own state
distribution. The bootstrap fine-tune (scripts/bootstrap_forecaster.py) then trains on these so
the forecaster is adapted to the states it actually sees.

Scenarios: SCN-01/03 (train) + SCN-04 (val) + SCN-06 (held-out test, the powered scenario).
SCN-02 is skipped (full gridlock -> degenerate maxed-out states). Output -> data/lstm_dqn/
(gitignored, regenerable), file naming ``scn_<NN>_seed_<SS>.csv`` so the LSTM windower picks
them up by scenario.

Run::

    LIBSUMO_AS_TRACI=1 python -m scripts.collect_dqn_traces
"""

from __future__ import annotations

from pathlib import Path

from scripts.build_network import build_net
from scripts.env_factory import build_env
from src.env.intersection import N_MOVEMENTS
from src.ml.dqn import DQNAgent
from src.ml.hybrid_wrapper import HYBRID_OBS_DIM, load_forecaster

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "data" / "lstm_dqn"
_RUNS = _REPO_ROOT / "runs"
_OFFICIAL_LSTM = (
    _REPO_ROOT / "checkpoints" / "lstm" / "lstm__data-8eb28eecdefb__lstm-df67afd839d4.pt"
)
SCENARIOS = ["SCN-01", "SCN-03", "SCN-04", "SCN-06"]
SEEDS = [0, 1, 2, 3, 4]
_HEADER = (
    ["step", "sim_time"]
    + [f"q_M{i}" for i in range(N_MOVEMENTS)]
    + [f"c_M{i}" for i in range(N_MOVEMENTS)]
)


def _load_hybrid_agent() -> DQNAgent:
    import torch
    agent = DQNAgent(HYBRID_OBS_DIM)
    state = torch.load(_RUNS / "hybrid_seed123" / "checkpoints" / "ep299.pt", map_location="cpu")
    agent.online.load_state_dict(state["online"])
    agent.online.eval()
    return agent


def collect_one(scenario_id: str, seed: int, agent: DQNAgent, forecaster) -> tuple[Path, int]:
    """Run the greedy hybrid DQN on (scenario, seed); write a per-step queue/count CSV."""
    env = build_env(scenario_id, seed, forecaster=forecaster)  # HybridStateWrapper
    rows = [",".join(_HEADER)]
    n = 0
    try:
        obs, info = env.reset()
        done = False
        step = 0
        while not done:
            action = agent.act(obs, info["mask"], epsilon=0.0)
            obs, _r, terminated, truncated, info = env.step(action)
            queue, count = env.movement_features()
            step += 1
            cells = [str(step), f"{info['sim_time']:.0f}"]
            cells += [str(int(v)) for v in queue] + [str(int(v)) for v in count]
            rows.append(",".join(cells))
            n += 1
            done = terminated or truncated
    finally:
        env.close()
    scn_num = scenario_id.split("-")[1]
    out = _OUT_DIR / f"scn_{scn_num}_seed_{seed:02d}.csv"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out, n


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_net()
    agent = _load_hybrid_agent()
    forecaster = load_forecaster(str(_OFFICIAL_LSTM))
    total = 0
    for scenario_id in SCENARIOS:
        for seed in SEEDS:
            out, n = collect_one(scenario_id, seed, agent, forecaster)
            print(f"[dqn-traces] {scenario_id} seed {seed:02d} -> {out.name} ({n} rows)")
            total += 1
    print(f"[dqn-traces] OK - {total} CSVs -> {_OUT_DIR.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
