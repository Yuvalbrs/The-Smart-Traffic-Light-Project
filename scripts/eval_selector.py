"""Improvement 2 eval: episode-level Webster floor over the risk-sensitive DQN.

Tests whether deciding DQN-vs-Webster UP FRONT (from an early demand probe) reaches Webster's
gridlock-robustness while keeping the DQN's wait edge on light episodes - the "guaranteed floor"
alternative to making the RL itself robust (Improvement 1). On the SAME paired eval episodes:
  * webster              - the robust reference
  * plain                - the scalar DQN (gridlocks ~93% under load)
  * iqn@0.05             - the risk-sensitive DQN alone (no floor) - Improvement-0 reference
  * selector@{tau}       - EpisodeLevelSelector(iqn@0.05, Webster, threshold=tau): probe 180 s of
                           Webster, then commit to the DQN (light) or stay Webster (heavy)

Reports per (scenario, controller): gridlock %, throughput, wait, and the selector's mean
fallback-active fraction (≈1.0 ⇒ routed to Webster, low ⇒ routed to the DQN).

Run::

    LIBSUMO_AS_TRACI=1 python -m scripts.eval_selector
    LIBSUMO_AS_TRACI=1 python -m scripts.eval_selector --scenarios SCN-05 --eval-seeds 7000  # smoke
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
_OUT = _REPO_ROOT / "data" / "eval" / "selector"
SEEDS = (42, 123, 2024)
DEFAULT_EVAL_SEEDS = (7000, 7001, 7002, 7003, 7004)
DEFAULT_SCENARIOS = ["SCN-04", "SCN-06", "SCN-05"]
THRESHOLDS = (150.0,)  # cumulative insertions over the 300 s probe; light <=146 vs heavy >=153
BASE_ALPHA = 0.05  # the risk-sensitive DQN the floor wraps (best low-alpha from the sweep)


def _agg(records: list[tuple[float, float, int]]):
    cens = [c for _w, _t, c in records]
    valid_wait = [w for w, _t, c in records if c == 0 and not np.isnan(w)]
    thru = [t for _w, t, _c in records if not np.isnan(t)]
    wait = np.mean(valid_wait) if valid_wait else float("nan")
    return wait, (np.mean(thru) if thru else float("nan")), 100 * np.mean(cens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(THRESHOLDS))
    parser.add_argument("--iqn-run", default="iqn_seed42", help="runs/<name>/checkpoints/ep299.pt to wrap")
    parser.add_argument("--episode-length", type=int, default=None)
    args = parser.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    build_net()
    build_actuated_add()

    plain = {s: _load_agent(_RUNS / f"plain_seed{s}" / "checkpoints" / "ep299.pt", 20)
             for s in SEEDS if (_RUNS / f"plain_seed{s}" / "checkpoints" / "ep299.pt").exists()}
    iqn_ckpt = _RUNS / args.iqn_run / "checkpoints" / "ep299.pt"
    if not iqn_ckpt.exists():
        parser.error(f"missing IQN checkpoint {iqn_ckpt}")
    iqn = _load_agent(iqn_ckpt, 20, cvar_alpha=BASE_ALPHA)

    for scn_id in args.scenarios:
        scn = load_scenario(SCENARIO_DIR / f"scn_{scn_id.split('-')[1]}.yaml")
        plan = webster_plan_for_scenario(scn)
        ep_len = args.episode_length or scn.duration_s
        warm = 300.0 if ep_len > 300 else 0.0

        def _run(algo: Algo) -> tuple[float, float, int]:
            kpis, _r = run_eval_episode(scn, es, algo, work_dir=_OUT,
                                        episode_length_s=ep_len, warmup_s=warm)
            return kpis.avg_waiting_time, kpis.throughput, int(kpis.gridlock_censored)

        rows: dict[str, list] = {}
        active: dict[str, list] = {}
        for es in args.eval_seeds:
            rows.setdefault("webster", []).append(
                _run(Algo("webster", "baseline", controller=WebsterController(plan))))
            for s, ag in plain.items():
                rows.setdefault("plain", []).append(_run(Algo(f"plain-s{s}", "dqn", agent=ag)))
            rows.setdefault(f"iqn@{BASE_ALPHA:.2f}", []).append(
                _run(Algo("iqn", "dqn", agent=iqn)))
            for tau in args.thresholds:
                sel = EpisodeLevelSelector(iqn, WebsterController(plan), threshold=tau)
                rows.setdefault(f"selector@{tau:.0f}", []).append(
                    _run(Algo(f"sel{tau:.0f}", "baseline", controller=sel)))
                active.setdefault(f"selector@{tau:.0f}", []).append(sel.active_frac)

        print(f"\n=== {scn_id} ===   (wait s | throughput | %gridlock | fallback-active%)")
        order = ["webster", "plain", f"iqn@{BASE_ALPHA:.2f}"] + [f"selector@{t:.0f}" for t in args.thresholds]
        for name in order:
            if name not in rows:
                continue
            wait, thru, grid = _agg(rows[name])
            af = f" | act={100 * np.mean(active[name]):3.0f}%" if name in active else ""
            print(f"  {name:13} wait={wait:6.2f} | thru={thru:7.1f} | grid={grid:4.0f}%{af}")

    print("\nGoal: selector gridlock% should match Webster on heavy scenarios (routed to Webster) "
          "while keeping the DQN's low wait on feasible SCN-04 (routed to the DQN, act% low).")


if __name__ == "__main__":
    main()
