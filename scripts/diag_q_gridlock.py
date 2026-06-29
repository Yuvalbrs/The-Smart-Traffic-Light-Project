"""Diagnostic gate for the risk-sensitive DQN (IQN+CVaR) decision.

Question: when the trained plain DQN drives a heavy scenario into gridlock, is that
a VALUE-ESTIMATION failure (the Q-function stays optimistic while the episode collapses,
i.e. it overestimates the gridlock tail) or a CAPACITY failure (Q collapses with the
state, the agent "knows" it is doomed and nothing can help)?

If overestimation: a distributional/risk-sensitive objective (CVaR over the return
distribution) has a real signal to exploit - at branch-point states the gridlock-bound
action carries a heavy lower tail that mean-Q hides. If capacity: no objective helps.

For each (scenario, eval_seed) we run the plain ep299 agent greedily and log, per step:
  - total standing queue (congestion proxy)
  - max_a Q(s,a) over LEGAL actions (the DQN's own greedy value)
  - Q(s, a_MP): the Q the DQN assigns to the max-pressure action (the robust choice)
  - whether the DQN's greedy action agrees with max-pressure
Episodes are flagged gridlock via insertion_backlog_fraction > 0.10 (same rule the KPI
extractor uses). We then contrast Q behaviour in low- vs high-congestion states and in
gridlock vs clean episodes.

Run::

    LIBSUMO_AS_TRACI=1 python -m scripts.diag_q_gridlock
    LIBSUMO_AS_TRACI=1 python -m scripts.diag_q_gridlock --scenarios SCN-05 --eval-seeds 7000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.baselines.max_pressure import MaxPressureController
from src.env.sumo_env import SUMOEnv
from src.ml.dqn import OBS_DIM, DQNAgent
from src.scenarios.config import SCENARIO_DIR, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS = _REPO_ROOT / "runs"
_MASK_FILL = -1e9
_GRIDLOCK_BACKLOG = 0.10  # same threshold the KPI extractor flags gridlock_censored at


def _load_plain(seed: int) -> DQNAgent:
    agent = DQNAgent(OBS_DIM)
    state = torch.load(_RUNS / f"plain_seed{seed}" / "checkpoints" / "ep299.pt", map_location="cpu")
    agent.online.load_state_dict(state["online"])
    agent.online.eval()
    return agent


def _q_values(agent: DQNAgent, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        return agent.online(t).squeeze(0).numpy()


def run_episode(scenario, eval_seed: int, agent: DQNAgent, mp: MaxPressureController):
    """Drive one greedy episode; return per-step arrays + the gridlock flag."""
    env = SUMOEnv(write_routes(scenario, eval_seed), episode_length_s=scenario.duration_s,
                  sumo_seed=eval_seed, signal_mode="rl")
    rows = []  # (total_queue, max_legal_q, q_at_mp, agree)
    gridlocked = False
    try:
        obs, info = env.reset()
        done = False
        while not done:
            mask = info["mask"]
            q = _q_values(agent, obs)
            q_masked = np.where(mask, q, _MASK_FILL)
            dqn_a = int(np.argmax(q_masked))
            mp_a = mp.select_action(obs, mask)
            tot_q = float(env.movement_features()[0].sum())
            rows.append((tot_q, float(q_masked.max()), float(q[mp_a]), int(dqn_a == mp_a)))
            obs, _r, terminated, truncated, info = env.step(dqn_a)
            done = terminated or truncated
        ep = info.get("episode", {})
        gridlocked = ep.get("insertion_backlog_fraction", 0.0) > _GRIDLOCK_BACKLOG
    finally:
        env.close()
    return np.array(rows, dtype=np.float64), gridlocked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="+", default=["SCN-05", "SCN-04"])
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[7000, 7001, 7002])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])  # train seed(s) of the agent
    args = p.parse_args()

    build_net()
    mp = MaxPressureController.from_spec()
    agents = {s: _load_plain(s) for s in args.seeds}

    print(f"{'scenario':9} {'eseed':6} {'aseed':5} {'grid':4} "
          f"{'maxQ_lo':>9} {'maxQ_hi':>9} {'Qmp_hi':>9} {'agree%':>7} {'peakQ':>6}  congestion(lo->hi)")
    print("-" * 100)
    for scn_id in args.scenarios:
        scn = load_scenario(SCENARIO_DIR / f"scn_{scn_id.split('-')[1]}.yaml")
        for es in args.eval_seeds:
            for aseed, agent in agents.items():
                rows, grid = run_episode(scn, es, agent, mp)
                tq, maxq, qmp, agree = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
                # split states by congestion: low = bottom third of queue, high = top third
                lo = tq <= np.percentile(tq, 33)
                hi = tq >= np.percentile(tq, 67)
                print(f"{scn_id:9} {es:<6} {aseed:<5} {'YES' if grid else ' no':4} "
                      f"{maxq[lo].mean():9.1f} {maxq[hi].mean():9.1f} {qmp[hi].mean():9.1f} "
                      f"{100*agree.mean():6.0f}% {maxq.max():6.1f}  "
                      f"{tq[lo].mean():.0f} -> {tq[hi].mean():.0f}")

    print("\nRead: if maxQ_hi (DQN value in the MOST congested states) stays HIGH / near maxQ_lo")
    print("while the episode gridlocks, the Q-function OVERESTIMATES the tail -> CVaR has signal.")
    print("If maxQ_hi collapses far below maxQ_lo, the agent already 'knows' -> weaker CVaR case.")
    print("Qmp_hi vs maxQ_hi shows whether the DQN values the robust max-pressure action below its own.")


if __name__ == "__main__":
    main()
