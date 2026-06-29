"""Does episode-level controller selection over the LOCKED plain DQN fix saturation-fragility
without losing the feasible-scenario wait edge - across all 3 seeds?

sel/plain = EpisodeLevelSelector(plain-DQN, Webster, threshold=150): run Webster for a 300 s
demand probe, then commit the rest of the episode to plain DQN if light, or stay on Webster if
heavy. No training - reuses the locked plain ep299 checkpoints. This is the sess15 "data-guaranteed"
fix, now over the stable plain DQN rather than the seed-unstable iqn-ra.

Per-seed x per-scenario split (the anti-aggregate discipline). Reference: webster + bare plain.

Run:: LIBSUMO_AS_TRACI=1 python -m scripts.compare_selector_plain
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
_OUT = _REPO_ROOT / "data" / "eval" / "compare_sel_plain"
TAU = 150.0
SEEDS = (42, 123, 2024)


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
                rows.setdefault(f"sel/plain-s{s}", []).append(
                    _run(Algo(f"sel{s}", "baseline",
                              controller=EpisodeLevelSelector(plain[s], WebsterController(plan), threshold=TAU))))

        print(f"\n=== {scn_id} ===   (wait s | throughput | %gridlock)", flush=True)
        order = ["webster"] + [f"{v}-s{s}" for s in SEEDS for v in ("plain", "sel/plain")]
        for name in order:
            w, t, g = _agg(rows[name])
            print(f"  {name:15} wait={w:6.2f} | thru={t:7.1f} | grid={g:4.0f}%", flush=True)


if __name__ == "__main__":
    main()
