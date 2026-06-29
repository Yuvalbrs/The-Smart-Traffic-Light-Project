"""Head-to-head: which fix pushes the risk-sensitive DQN's gridlock toward Webster?

On the SAME paired eval episodes (SCN-04 feasible / SCN-06 powered / SCN-05 heavy), compares:
  * webster                  - the robust analytical reference
  * plain-s42                - the locked scalar DQN
  * iqn-base@0.05            - Improvement 0: risk-sensitive DQN, eval-only CVaR (undertrained)
  * iqn-ra@0.05 (ep299/best) - Improvement 1: warm-started + risk-averse-TRAINED IQN
  * selector/iqn-base        - Improvement 2: episode-level Webster floor over the base IQN
  * selector/iqn-ra          - Improvement 1+2 combined (floor over the better DQN)

Run:: LIBSUMO_AS_TRACI=1 python -m scripts.compare_improvements
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.build_actuated import build_actuated_add
from scripts.build_network import build_net
from scripts.eval_runner import Algo, _load_agent, run_eval_episode
from src.baselines.webster import WebsterController, webster_plan_for_scenario
from src.ml.supervisor import EpisodeLevelSelector
from src.scenarios.config import SCENARIO_DIR, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS = _REPO_ROOT / "runs"
_OUT = _REPO_ROOT / "data" / "eval" / "compare"
ALPHA = 0.05
TAU = 150.0


def _agg(records):
    cens = [c for _w, _t, c in records]
    valid_wait = [w for w, _t, c in records if c == 0 and not np.isnan(w)]
    thru = [t for _w, t, _c in records if not np.isnan(t)]
    return (np.mean(valid_wait) if valid_wait else float("nan"),
            np.mean(thru) if thru else float("nan"), 100 * np.mean(cens))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="+", default=["SCN-04", "SCN-06", "SCN-05"])
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[7000, 7001, 7002, 7003, 7004])
    args = p.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    build_net()
    build_actuated_add()

    plain = _load_agent(_RUNS / "plain_seed42" / "checkpoints" / "ep299.pt", 20)
    iqn_base = _load_agent(_RUNS / "iqn_seed42" / "checkpoints" / "ep299.pt", 20, cvar_alpha=ALPHA)
    iqn_ra = _load_agent(_RUNS / "iqn-ra_seed42" / "checkpoints" / "ep299.pt", 20, cvar_alpha=ALPHA)
    iqn_ra_best = _load_agent(_RUNS / "iqn-ra_seed42" / "checkpoints" / "best.pt", 20, cvar_alpha=ALPHA)

    for scn_id in args.scenarios:
        scn = load_scenario(SCENARIO_DIR / f"scn_{scn_id.split('-')[1]}.yaml")
        plan = webster_plan_for_scenario(scn)
        ep_len = scn.duration_s
        warm = 300.0

        def _run(algo: Algo):
            k, _r = run_eval_episode(scn, es, algo, work_dir=_OUT, episode_length_s=ep_len, warmup_s=warm)
            return k.avg_waiting_time, k.throughput, int(k.gridlock_censored)

        rows: dict[str, list] = {}
        for es in args.eval_seeds:
            rows.setdefault("webster", []).append(_run(Algo("webster", "baseline", controller=WebsterController(plan))))
            rows.setdefault("plain", []).append(_run(Algo("plain", "dqn", agent=plain)))
            rows.setdefault("iqn-base", []).append(_run(Algo("iqnb", "dqn", agent=iqn_base)))
            rows.setdefault("iqn-ra(ep299)", []).append(_run(Algo("iqnra", "dqn", agent=iqn_ra)))
            rows.setdefault("iqn-ra(best)", []).append(_run(Algo("iqnrab", "dqn", agent=iqn_ra_best)))
            rows.setdefault("sel/iqn-base", []).append(
                _run(Algo("selb", "baseline", controller=EpisodeLevelSelector(iqn_base, WebsterController(plan), threshold=TAU))))
            rows.setdefault("sel/iqn-ra", []).append(
                _run(Algo("selra", "baseline", controller=EpisodeLevelSelector(iqn_ra, WebsterController(plan), threshold=TAU))))

        print(f"\n=== {scn_id} ===   (wait s | throughput | %gridlock)")
        for name in ["webster", "plain", "iqn-base", "iqn-ra(ep299)", "iqn-ra(best)", "sel/iqn-base", "sel/iqn-ra"]:
            w, t, g = _agg(rows[name])
            print(f"  {name:15} wait={w:6.2f} | thru={t:7.1f} | grid={g:4.0f}%")


if __name__ == "__main__":
    main()
