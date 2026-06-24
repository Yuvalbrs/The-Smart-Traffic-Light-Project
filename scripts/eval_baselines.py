"""T-04-05 - Baseline eval dry-run: the first REAL KPI numbers.

Runs the three locked baselines - **Webster**, **max-pressure**, **SUMO-actuated**
- through ``SUMOEnv`` (emitting both the JSONL trace and the SUMO trip-info XML)
and scores each episode with the single :func:`extract_kpis` extractor. Purpose
(backlog T-04-05): exercise the data -> sim -> KPI pipeline end to end on real
episodes - weeks before the full eval runner (T-04-01) feeds real DQN checkpoints
through it - and produce the first "do the baselines rank sensibly?" table.

Scope vs T-04-01: this is the *quick scorecard*. It does NOT write SQLite rows or
the full provenance chain - that is the official runner's job. It DOES already use
the load-bearing pairing discipline: within one ``(scenario, seed)`` every
controller runs on the **same SUMO seed** (controller is the inner loop), so the
samples are paired exactly as the Wilcoxon test (pre-registration.md) needs.

The trip-info wiring this exercises (``SUMOEnv(tripinfo_path=...)``) is the
prerequisite hot.md flagged: the env emitted JSONL but not trip-info, so the
extractor could not run end to end until now.

Run::

    python -m scripts.eval_baselines                         # all scenarios, 1 seed
    python -m scripts.eval_baselines --scenario SCN-02 --seeds 3
    python -m scripts.eval_baselines --episode-length 600    # quick smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from scripts.build_actuated import build_actuated_add
from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.baselines.max_pressure import MaxPressureController
from src.baselines.webster import WebsterController, webster_plan_for_scenario
from src.metrics.kpi_extractor import EpisodeKPIs, extract_kpis
from src.env.sumo_env import SUMOEnv
from src.scenarios.config import SCENARIO_DIR, Scenario, load_all, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "data" / "eval"

# Controller is the INNER loop so every controller in a (scenario, seed) cell runs
# on the same SUMO vehicle seed -> paired samples (evaluation-methodology.md).
CONTROLLERS = ("webster", "max_pressure", "actuated")

# The scalars shown in the printed table (5b -> worst_movement_max_wait scalar).
_TABLE_KPIS = (
    ("avg_waiting_time", "avg_wait", "s", "down"),
    ("avg_queue_length", "avg_queue", "veh", "down"),
    ("throughput", "throughput", "veh/h", "up"),
    ("num_stops", "stops", "/veh", "down"),
    ("wait_p95", "p95_wait", "s", "down"),
    ("fairness_std", "fairness", "s", "down"),
    ("worst_movement_max_wait", "worst_max", "s", "down"),
)


def _make_controller(name: str, scenario: Scenario):
    """Build a baseline controller, or ``None`` for SUMO-driven actuated mode."""
    if name == "webster":
        return WebsterController(webster_plan_for_scenario(scenario))
    if name == "max_pressure":
        return MaxPressureController.from_spec()
    if name == "actuated":
        return None  # SUMO's own program drives the lights (signal_mode="actuated")
    raise ValueError(f"unknown controller {name!r}")


def run_episode(
    scenario: Scenario,
    seed: int,
    controller: str,
    *,
    work_dir: Path,
    episode_length_s: int,
    warmup_s: float,
) -> EpisodeKPIs:
    """Run one (scenario, seed, controller) episode and return its 7 KPIs."""
    route = write_routes(scenario, seed)
    stem = f"{scenario.id}_seed{seed:02d}_{controller}"
    jsonl = work_dir / f"{stem}.jsonl"
    tripinfo = work_dir / f"{stem}.tripinfo.xml"
    ctrl = _make_controller(controller, scenario)

    env = SUMOEnv(
        route,
        episode_length_s=episode_length_s,
        sumo_seed=seed,
        signal_mode="actuated" if controller == "actuated" else "rl",
        trace_path=jsonl,
        tripinfo_path=tripinfo,
    )
    counters: dict | None = None
    try:
        obs, info = env.reset()
        if ctrl is not None:
            ctrl.reset(env)
        done = False
        while not done:
            action = 0 if ctrl is None else ctrl.select_action(obs, info["mask"])
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        counters = info.get("episode")  # loaded/departed/arrived (gridlock guard)
    finally:
        env.close()  # SUMO finalizes the trip-info XML here

    return extract_kpis(
        jsonl,
        tripinfo,
        episode_counters=counters,
        warmup_s=warmup_s,
        episode_length_s=episode_length_s,
    )


def _fmt(values: list[float]) -> str:
    """``mean`` (one seed) or ``mean +/- std`` (several), NaN-safe."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return "  n/a"
    if arr.size == 1:
        return f"{arr[0]:7.1f}"
    return f"{arr.mean():7.1f}+/-{arr.std(ddof=1):.1f}"


def _print_scenario_table(scenario_id: str, by_ctrl: dict[str, list[EpisodeKPIs]]) -> None:
    """Print one comparison table: rows = controllers, cols = the 7 KPIs."""
    print(f"\n=== {scenario_id} ===")
    head = f"{'controller':<14}" + "".join(f"{short:>14}" for _, short, _, _ in _TABLE_KPIS)
    print(head)
    print("-" * len(head))
    for ctrl in CONTROLLERS:
        eps = by_ctrl.get(ctrl, [])
        cells = [_fmt([getattr(k, field) for k in eps]) for field, _, _, _ in _TABLE_KPIS]
        flags = ""
        if any(k.gridlock_censored for k in eps):
            flags = "  [gridlock-censored]"
        print(f"{ctrl:<14}" + "".join(f"{c:>14}" for c in cells) + flags)
    print("(avg_wait/queue/stops/p95/fairness/worst_max: lower better; throughput: higher better)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="single scenario id (e.g. SCN-02); default all")
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds (from the front); default 1")
    parser.add_argument("--episode-length", type=int, help="override episode length (s), for smoke runs")
    parser.add_argument("--warmup", type=float, default=300.0, help="KPI warm-up cutoff (s); default 300")
    args = parser.parse_args()

    build_net()  # net + actuated additional-file must match current sources
    build_actuated_add()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = (
        [load_scenario(SCENARIO_DIR / f"scn_{args.scenario.split('-')[1]}.yaml")]
        if args.scenario
        else load_all()
    )

    records: list[dict] = []
    for scenario in scenarios:
        seeds = list(scenario.seeds[: args.seeds])
        episode_length_s = args.episode_length or scenario.duration_s
        by_ctrl: dict[str, list[EpisodeKPIs]] = {c: [] for c in CONTROLLERS}
        for seed in seeds:
            for controller in CONTROLLERS:  # inner loop -> shared seed -> paired
                kpis = run_episode(
                    scenario, seed, controller,
                    work_dir=_OUT_DIR, episode_length_s=episode_length_s, warmup_s=args.warmup,
                )
                by_ctrl[controller].append(kpis)
                records.append({
                    "scenario": scenario.id, "seed": seed, "controller": controller,
                    **dataclasses.asdict(kpis),
                })
                print(f"[eval] {scenario.id} seed {seed:02d} {controller:<13} "
                      f"avg_wait={kpis.avg_waiting_time:6.1f}s throughput={kpis.throughput:7.1f} "
                      f"{'CENSORED' if kpis.gridlock_censored else ''}")
        _print_scenario_table(scenario.id, by_ctrl)

    out_path = _OUT_DIR / "baseline_dryrun.json"
    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"\n[eval] OK - {len(records)} episodes -> {out_path.relative_to(_REPO_ROOT)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
