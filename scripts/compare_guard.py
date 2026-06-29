"""Can a HARD queue-guard remove the DQN's gridlock without losing its wait edge?

On the fast-shifting feasible scenarios (SCN-08/09) the plain DQN beats Webster on wait (~1.8s vs
2.6s) but gridlocks 20-60% (Webster 0%) - it starves one axis into a recoverable jam. This is NOT
the SCN-05 capacity wall (load is feasible); it's a policy artifact. The guard: let the DQN choose
freely, but if any movement's queue exceeds a danger cap, RESTRICT the choice to legal phases that
serve the over-cap movement(s) - pre-empting the starvation jam before spillback. The DQN still
picks (max-Q within the restricted set), so its wait-optimization is preserved; no retrain, no
handover to Webster (unlike the failed reactive supervisor), a hard action constraint (unlike the
failed soft reward penalty).

Per-seed x per-scenario split. References: webster + bare plain.

Run:: LIBSUMO_AS_TRACI=1 python -m scripts.compare_guard --scenarios SCN-08 SCN-09
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_actuated import build_actuated_add
from scripts.build_network import build_net
from scripts.eval_runner import Algo, _load_agent, run_eval_episode
from src.baselines.webster import WebsterController, webster_plan_for_scenario
from src.env.intersection import _VAULT_MOVEMENTS, load_phase_movements
from src.scenarios.config import SCENARIO_DIR, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS = _REPO_ROOT / "runs"
_OUT = _REPO_ROOT / "data" / "eval" / "compare_guard"
SEEDS = (42, 123, 2024)
QUEUE_CAP = 15.0  # per-movement queue (veh) above which the guard forces service of that movement


class QueueGuardController:
    """Plain DQN whose action is constrained to serve any movement whose queue exceeds QUEUE_CAP."""

    def __init__(self, agent: Any, action_movements: dict[int, tuple[int, ...]], cap: float) -> None:
        self.agent = agent
        self.action_movements = {int(a): tuple(int(m) for m in ms) for a, ms in action_movements.items()}
        self.cap = float(cap)
        self._env: Any = None
        self.guard_steps = 0
        self.total_steps = 0

    def reset(self, env: Any) -> None:
        self._env = env
        self.guard_steps = self.total_steps = 0

    def select_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        self.total_steps += 1
        queues = np.asarray(self._env.movement_features()[0], dtype=np.float64)  # (12,) standing queue
        over = set(np.nonzero(queues > self.cap)[0].tolist())
        if over:
            # phases that are legal AND serve at least one over-cap movement
            guard = np.zeros_like(mask)
            for a, ms in self.action_movements.items():
                if mask[a] and over.intersection(ms):
                    guard[a] = True
            if guard.any():  # otherwise no legal phase can serve it now (min-green/yellow) -> free choice
                self.guard_steps += 1
                return int(self.agent.act(obs, guard, epsilon=0.0))
        return int(self.agent.act(obs, mask, epsilon=0.0))

    @property
    def guard_frac(self) -> float:
        return self.guard_steps / self.total_steps if self.total_steps else 0.0


def _agg(records):
    cens = [c for _w, _t, c in records]
    valid_wait = [w for w, _t, c in records if c == 0 and not np.isnan(w)]
    thru = [t for _w, t, _c in records if not np.isnan(t)]
    return (np.mean(valid_wait) if valid_wait else float("nan"),
            np.mean(thru) if thru else float("nan"), 100 * np.mean(cens))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="+", default=["SCN-08", "SCN-09"])
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[7000, 7001, 7002, 7003, 7004])
    p.add_argument("--cap", type=float, default=QUEUE_CAP)
    args = p.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    build_net()
    build_actuated_add()
    phase_mv = load_phase_movements(_VAULT_MOVEMENTS)
    plain = {s: _load_agent(_RUNS / f"plain_seed{s}" / "checkpoints" / "ep299.pt", 20) for s in SEEDS}

    for scn_id in args.scenarios:
        scn = load_scenario(SCENARIO_DIR / f"scn_{scn_id.split('-')[1]}.yaml")
        plan = webster_plan_for_scenario(scn)
        ep_len, warm = scn.duration_s, 300.0

        def _run(algo: Algo):
            k, _r = run_eval_episode(scn, es, algo, work_dir=_OUT, episode_length_s=ep_len, warmup_s=warm)
            return k.avg_waiting_time, k.throughput, int(k.gridlock_censored)

        rows: dict[str, list] = {}
        for es in args.eval_seeds:
            rows.setdefault("webster", []).append(_run(Algo("webster", "baseline", controller=WebsterController(plan))))
            for s in SEEDS:
                rows.setdefault(f"plain-s{s}", []).append(_run(Algo(f"plain{s}", "dqn", agent=plain[s])))
                rows.setdefault(f"guard/plain-s{s}", []).append(
                    _run(Algo(f"grd{s}", "baseline",
                              controller=QueueGuardController(plain[s], phase_mv, args.cap))))

        print(f"\n=== {scn_id} ===  (cap={args.cap})  (wait s | throughput | %gridlock)", flush=True)
        order = ["webster"] + [f"{v}-s{s}" for s in SEEDS for v in ("plain", "guard/plain")]
        for name in order:
            w, t, g = _agg(rows[name])
            print(f"  {name:16} wait={w:6.2f} | thru={t:7.1f} | grid={g:4.0f}%", flush=True)


if __name__ == "__main__":
    main()
