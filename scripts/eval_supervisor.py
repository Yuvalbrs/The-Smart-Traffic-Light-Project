"""Safety Supervisor evaluation - does the RL+Webster hybrid fix the gridlock weakness?

Compares, on the same paired eval episodes:
  * webster      - the robust classical fallback alone
  * plain        - the DQN alone (gridlocks ~93% under heavy load)
  * supervisor@tau - DQN + Webster fallback, saturation threshold tau (sweep)

across a light scenario (SCN-01, DQN should be preserved), the powered shifting scenario (SCN-06),
and the oversaturated regimes (SCN-05 = test, SCN-02 = heaviest). Reports per (scenario,
controller): gridlock-censor rate, mean throughput, mean wait (valid eps), and the supervisor's
mean active_frac (how often the fallback held control - the graceful-degradation measure). The
threshold sweep + active_frac calibrate tau and produce the robustness story.

Run::

    LIBSUMO_AS_TRACI=1 python -m scripts.eval_supervisor
    LIBSUMO_AS_TRACI=1 python -m scripts.eval_supervisor --scenarios SCN-05 --eval-seeds 7000  # smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.build_actuated import build_actuated_add
from scripts.build_network import build_net
from scripts.eval_runner import Algo, _load_agent, run_eval_episode
from src.baselines.webster import WebsterController, webster_plan_for_scenario
from src.ml.supervisor import SafetySupervisor
from src.scenarios.config import SCENARIO_DIR, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS = _REPO_ROOT / "runs"
_OUT = _REPO_ROOT / "data" / "eval" / "supervisor"
SEEDS = (42, 123, 2024)
DEFAULT_EVAL_SEEDS = (7000, 7001, 7002)
DEFAULT_SCENARIOS = ["SCN-01", "SCN-06", "SCN-05", "SCN-02"]
THRESHOLDS = (15.0, 30.0, 60.0, 120.0)  # incl. low = proactive (engage Webster BEFORE gridlock forms)


def _agg(records: list[tuple[float, float, int]]):
    """(mean wait over valid, mean throughput over all, gridlock %) from (wait, thru, censored)."""
    cens = [c for _w, _t, c in records]
    valid_wait = [w for w, _t, c in records if c == 0 and not np.isnan(w)]
    thru = [t for _w, t, _c in records if not np.isnan(t)]
    wait = np.mean(valid_wait) if valid_wait else float("nan")
    return wait, (np.mean(thru) if thru else float("nan")), 100 * np.mean(cens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--episode-length", type=int, default=None)
    args = parser.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    build_net()
    build_actuated_add()
    agents = {s: _load_agent(_RUNS / f"plain_seed{s}" / "checkpoints" / "ep299.pt", 20) for s in SEEDS}

    for scn_id in args.scenarios:
        scn = load_scenario(SCENARIO_DIR / f"scn_{scn_id.split('-')[1]}.yaml")
        plan = webster_plan_for_scenario(scn)
        ep_len = args.episode_length
        warm = 300.0 if (ep_len or 3600) > 300 else 0.0

        def _run(algo: Algo) -> tuple[float, float, int]:
            kpis, _r = run_eval_episode(scn, es, algo, work_dir=_OUT,
                                        episode_length_s=ep_len or scn.duration_s, warmup_s=warm)
            return kpis.avg_waiting_time, kpis.throughput, int(kpis.gridlock_censored)

        rows: dict[str, list] = {}
        active: dict[str, list] = {}

        for es in args.eval_seeds:
            rows.setdefault("webster", []).append(
                _run(Algo("webster", "baseline", controller=WebsterController(plan))))
            for s in SEEDS:
                rows.setdefault("plain", []).append(
                    _run(Algo(f"plain-s{s}", "dqn", agent=agents[s])))
                for tau in THRESHOLDS:
                    sup = SafetySupervisor(agents[s], WebsterController(plan), threshold=tau)
                    rec = _run(Algo(f"sup{tau:.0f}-s{s}", "baseline", controller=sup))
                    rows.setdefault(f"supervisor@{tau:.0f}", []).append(rec)
                    active.setdefault(f"supervisor@{tau:.0f}", []).append(sup.active_frac)

        print(f"\n=== {scn_id} ===   (wait s | throughput | %gridlock | fallback-active%)")
        order = ["webster", "plain"] + [f"supervisor@{t:.0f}" for t in THRESHOLDS]
        for name in order:
            wait, thru, grid = _agg(rows[name])
            af = f" | act={100 * np.mean(active[name]):3.0f}%" if name in active else ""
            print(f"  {name:16} wait={wait:6.2f} | thru={thru:7.1f} | grid={grid:4.0f}%{af}")

    print("\nGoal: on heavy scenarios the supervisor's gridlock% should fall toward Webster's "
          "while preserving the DQN's wait on light scenarios. active% shows when the fallback engaged.")


if __name__ == "__main__":
    main()
