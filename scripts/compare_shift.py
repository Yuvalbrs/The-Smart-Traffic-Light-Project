"""Does TRAINING on fast shifts (SCN-10 in the rotation) fix the DQN's gridlock on held-out
shifting demand - without regressing the other regimes?

Attempt #10 / the generalization hypothesis: the 9 prior fixes all patched a steady-trained model
from the outside and failed. plain-shift instead LEARNS shift dynamics (trained on SCN-01/02/03 +
SCN-10). Held-out test = SCN-08/09 (distinct from the SCN-10 it trained on). No-regression check =
SCN-04 (feasible), SCN-06 (near-sat balanced), SCN-05 (oversaturation, must still degrade, not win).

Win condition: on SCN-08/09, plain-shift gridlock drops toward Webster's 0% while keeping the
~1.8s wait edge (Webster ~2.6s). Per-seed x per-scenario split.

Run:: LIBSUMO_AS_TRACI=1 python -m scripts.compare_shift
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
_OUT = _REPO_ROOT / "data" / "eval" / "compare_shift"
SEEDS = (42, 123, 2024)


def _agg(records):
    cens = [c for _w, _t, c in records]
    valid_wait = [w for w, _t, c in records if c == 0 and not np.isnan(w)]
    thru = [t for _w, t, _c in records if not np.isnan(t)]
    return (np.mean(valid_wait) if valid_wait else float("nan"),
            np.mean(thru) if thru else float("nan"), 100 * np.mean(cens))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # held-out shift tests first, then the no-regression set
    p.add_argument("--scenarios", nargs="+", default=["SCN-08", "SCN-09", "SCN-04", "SCN-06", "SCN-05"])
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[7000, 7001, 7002, 7003, 7004])
    args = p.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)
    build_net()
    build_actuated_add()

    plain = {s: _load_agent(_RUNS / f"plain_seed{s}" / "checkpoints" / "ep299.pt", 20) for s in SEEDS}
    shift = {s: _load_agent(_RUNS / f"plain-shift_seed{s}" / "checkpoints" / "ep299.pt", 20) for s in SEEDS}

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
                rows.setdefault(f"plain-shift-s{s}", []).append(_run(Algo(f"shift{s}", "dqn", agent=shift[s])))

        print(f"\n=== {scn_id} ===   (wait s | throughput | %gridlock)", flush=True)
        order = ["webster"] + [f"{v}-s{s}" for s in SEEDS for v in ("plain", "plain-shift")]
        for name in order:
            w, t, g = _agg(rows[name])
            print(f"  {name:18} wait={w:6.2f} | thru={t:7.1f} | grid={g:4.0f}%", flush=True)


if __name__ == "__main__":
    main()
