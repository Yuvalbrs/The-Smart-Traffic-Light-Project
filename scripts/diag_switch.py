"""Is the DQN's feasible-load gridlock caused by OVER-SWITCHING (phase thrashing)?

Under fast-shifting demand (SCN-08) the DQN beats Webster on wait but gridlocks 20-40%. Hypothesis:
it thrashes between axes; each phase change costs yellow(3s)+all-red(2s) lost time, so a high switch
rate collapses throughput -> jam. Webster's fixed cycle has a bounded switch rate. This logs, per
(controller, eval_seed): switch fraction (phase-changes / decisions) and the gridlock flag.

Run:: LIBSUMO_AS_TRACI=1 python -m scripts.diag_switch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.baselines.webster import WebsterController, webster_plan_for_scenario
from src.env.sumo_env import SUMOEnv
from src.ml.dqn import OBS_DIM, DQNAgent
from src.scenarios.config import SCENARIO_DIR, load_scenario

_RUNS = Path(__file__).resolve().parent.parent / "runs"
_GRIDLOCK_BACKLOG = 0.10


def _load_plain(seed: int) -> DQNAgent:
    import torch
    a = DQNAgent(OBS_DIM)
    st = torch.load(_RUNS / f"plain_seed{seed}" / "checkpoints" / "ep299.pt", map_location="cpu")
    a.online.load_state_dict(st["online"]); a.online.eval()
    return a


def _run(scn, es, pick):
    env = SUMOEnv(write_routes(scn, es), episode_length_s=scn.duration_s, sumo_seed=es, signal_mode="rl")
    changes = decisions = 0
    last = None
    try:
        obs, info = env.reset()
        if hasattr(pick, "reset"):
            pick.reset(env)
        done = False
        while not done:
            a = pick(obs, info["mask"]) if callable(pick) else pick.select_action(obs, info["mask"])
            decisions += 1
            if last is not None and a != last:
                changes += 1
            last = a
            obs, _r, term, trunc, info = env.step(a)
            done = term or trunc
        grid = info.get("episode", {}).get("insertion_backlog_fraction", 0.0) > _GRIDLOCK_BACKLOG
    finally:
        env.close()
    return changes / decisions if decisions else 0.0, grid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="SCN-08")
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[7000, 7001, 7002])
    args = p.parse_args()
    build_net()
    scn = load_scenario(SCENARIO_DIR / f"scn_{args.scenario.split('-')[1]}.yaml")
    plan = webster_plan_for_scenario(scn)
    plains = {s: _load_plain(s) for s in (42, 123, 2024)}

    print(f"{'controller':16} {'eseed':6} {'switch_frac':>11} {'grid':>5}")
    print("-" * 42)
    for es in args.eval_seeds:
        sf, g = _run(scn, es, WebsterController(plan))
        print(f"{'webster':16} {es:<6} {sf:>11.2f} {'YES' if g else 'no':>5}")
        for s, ag in plains.items():
            sf, g = _run(scn, es, lambda o, m, _a=ag: int(_a.act(o, m, epsilon=0.0)))
            print(f"{'plain-s'+str(s):16} {es:<6} {sf:>11.2f} {'YES' if g else 'no':>5}")


if __name__ == "__main__":
    main()
