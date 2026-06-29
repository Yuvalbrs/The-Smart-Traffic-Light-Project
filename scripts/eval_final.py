"""Final head-to-head: the Webster-warm-started risk-sensitive DQN vs everything.

The trade-off question (sess16): which controller is BOTH low-gridlock AND low-wait? On the same
paired eval episodes (SCN-04 feasible / SCN-06 powered / SCN-05 heavy):
  * webster / plain                 - references
  * iqn-webster@{1.0..0.05}         - the pick: BC-warm-started from Webster, CVaR swept at eval
  * selector/iqn-webster            - Webster floor over the new model (gridlock guarantee)

Read against the previously-measured iqn-base / iqn-ra / selector numbers (same eval seeds). The
win condition: gridlock ~Webster on heavy AND wait clearly below Webster on feasible (a genuine
Pareto improvement, not safety bought by surrendering flow).

Run:: LIBSUMO_AS_TRACI=1 python -m scripts.eval_final
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
_OUT = _REPO_ROOT / "data" / "eval" / "final"
ALPHAS = (1.0, 0.25, 0.10, 0.05)
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
    p.add_argument("--run", default="iqn-webster_seed42")
    args = p.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    build_net()
    build_actuated_add()

    plain = _load_agent(_RUNS / "plain_seed42" / "checkpoints" / "ep299.pt", 20)
    iqn = _load_agent(_RUNS / args.run / "checkpoints" / "ep299.pt", 20)  # alpha set per-run below

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
            rows.setdefault("plain", []).append(_run(Algo("plain", "dqn", agent=plain)))
            for a in ALPHAS:
                iqn.cvar_alpha = a
                rows.setdefault(f"iqn-web@{a:.2f}", []).append(_run(Algo("iw", "dqn", agent=iqn)))
            iqn.cvar_alpha = 0.05
            rows.setdefault("sel/iqn-web", []).append(
                _run(Algo("sel", "baseline", controller=EpisodeLevelSelector(iqn, WebsterController(plan), threshold=TAU))))

        print(f"\n=== {scn_id} ===   (wait s | throughput | %gridlock)")
        for name in ["webster", "plain", *[f"iqn-web@{a:.2f}" for a in ALPHAS], "sel/iqn-web"]:
            w, t, g = _agg(rows[name])
            print(f"  {name:13} wait={w:6.2f} | thru={t:7.1f} | grid={g:4.0f}%")


if __name__ == "__main__":
    main()
