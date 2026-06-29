"""T-03-09 eval: does risk-sensitive (IQN + CVaR) DQN fix the gridlock weakness?

The headline experiment for the distributional contribution. On the SAME paired eval episodes,
compares:
  * webster                - the robust classical reference (0% gridlock under load)
  * plain-s{seed}          - the locked scalar DQN (gridlocks ~93% under heavy load)
  * iqn-s{seed}@{alpha}     - the trained IQN, CVaR action-selection swept over alpha

CVaR is an action-SELECTION knob, so ONE trained IQN per seed is evaluated at every alpha (no
retraining): alpha=1.0 is risk-neutral (mean-Q, the distributional control), lower alpha is more
risk-averse (optimizes the gridlock tail). Reports per (scenario, controller): gridlock-censor
rate, mean throughput, mean wait (valid eps). The alpha column is the publishable mean-vs-tail
tradeoff curve; the comparison vs plain at alpha=1.0 isolates "distributional architecture" from
"risk aversion", and vs Webster shows whether risk aversion reaches the analytical floor.

Scenarios: SCN-04 (feasible - the DQN's wait edge must survive), SCN-06 (powered, n=15 - the
statistical-power scenario), SCN-05 (the designated heavy/oversaturated test).

Run::

    LIBSUMO_AS_TRACI=1 python -m scripts.eval_iqn
    LIBSUMO_AS_TRACI=1 python -m scripts.eval_iqn --scenarios SCN-05 --eval-seeds 7000  # smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.build_actuated import build_actuated_add
from scripts.build_network import build_net
from scripts.eval_runner import Algo, _load_agent, run_eval_episode
from src.baselines.webster import WebsterController, webster_plan_for_scenario
from src.scenarios.config import SCENARIO_DIR, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS = _REPO_ROOT / "runs"
_OUT = _REPO_ROOT / "data" / "eval" / "iqn"
SEEDS = (42, 123, 2024)
DEFAULT_EVAL_SEEDS = (7000, 7001, 7002, 7003, 7004)
DEFAULT_SCENARIOS = ["SCN-04", "SCN-06", "SCN-05"]
ALPHAS = (1.0, 0.25, 0.1, 0.05)  # risk-neutral -> increasingly risk-averse


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
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    parser.add_argument("--episode-length", type=int, default=None)
    args = parser.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    build_net()
    build_actuated_add()

    plain = {s: _load_agent(_RUNS / f"plain_seed{s}" / "checkpoints" / "ep299.pt", 20)
             for s in SEEDS if (_RUNS / f"plain_seed{s}" / "checkpoints" / "ep299.pt").exists()}
    iqn = {s: _load_agent(_RUNS / f"iqn_seed{s}" / "checkpoints" / "ep299.pt", 20)
           for s in SEEDS if (_RUNS / f"iqn_seed{s}" / "checkpoints" / "ep299.pt").exists()}
    if not iqn:
        parser.error("no trained IQN checkpoints found (runs/iqn_seed*/checkpoints/ep299.pt) - train first")

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
        for es in args.eval_seeds:
            rows.setdefault("webster", []).append(
                _run(Algo("webster", "baseline", controller=WebsterController(plan))))
            for s, ag in plain.items():
                rows.setdefault("plain", []).append(_run(Algo(f"plain-s{s}", "dqn", agent=ag)))
            for s, ag in iqn.items():
                for a in args.alphas:
                    ag.cvar_alpha = a  # CVaR is an action-selection knob: reuse the one model
                    rows.setdefault(f"iqn@{a:.2f}", []).append(
                        _run(Algo(f"iqn-s{s}-a{a:.2f}", "dqn", agent=ag)))

        print(f"\n=== {scn_id} ===   (wait s | throughput | %gridlock)")
        order = ["webster", "plain"] + [f"iqn@{a:.2f}" for a in args.alphas]
        for name in order:
            if name not in rows:
                continue
            wait, thru, grid = _agg(rows[name])
            print(f"  {name:12} wait={wait:6.2f} | thru={thru:7.1f} | grid={grid:4.0f}%")

    print("\nGoal: iqn@low-alpha gridlock% should fall toward Webster's WITHOUT losing the DQN wait "
          "edge on feasible SCN-04; the alpha column is the mean-vs-tail tradeoff curve.")


if __name__ == "__main__":
    main()
