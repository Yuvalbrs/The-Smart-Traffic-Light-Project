"""T-01-05 - Generate the 50 LSTM training CSVs (5 scenarios x 10 seeds).

Runs the **Webster** baseline (the controller the backlog pins for T-01-05;
``baselines-implementation.md``) through ``SUMOEnv`` for each ``(scenario, seed)``
and records a per-decision-step time series of per-movement queue + count. The
windowing into ``(12, 24)`` inputs / ``(3, 12)`` targets is the LSTM data loader's
job (T-03-01) - this script only produces the raw series.

Each CSV row = one 10 s decision step::

    step, sim_time, q_M0..q_M11, c_M0..c_M11

* ``q_*`` queue (halting count) per movement - the LSTM forecast target/feature;
* ``c_*`` incoming vehicle count per movement - an extra LSTM input feature.

Determinism: the route file is byte-identical per ``(scenario, seed)`` (T-01-08)
and SUMO is seeded with the same ``seed``, so a regenerated CSV is byte-identical.
A ``manifest.json`` records each file's deterministic ``data_version`` (provenance).

NOTE (surfaced, not silently resolved): the backlog DoD says "each row in a
data_version table", but no such SQLite table exists (``data_version`` is only a
column on ``experiment_run``, which models a *run*, not a *dataset*). Provenance is
recorded in the manifest here; committing golden hashes is T-01-10's scope.

Outputs land in ``data/lstm/`` (gitignored, regenerable). Run::

    python -m scripts.generate_lstm_data                       # all 50
    python -m scripts.generate_lstm_data --scenario SCN-01 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.baselines.webster import WebsterController, webster_plan_for_scenario
from src.env.intersection import N_MOVEMENTS
from src.env.sumo_env import SUMOEnv
from src.provenance.versions import data_version, git_sha, hash_files, sumo_version
from src.scenarios.config import SCENARIO_DIR, Scenario, load_all, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "data" / "lstm"

CONTROLLER = "webster"  # locked for T-01-05 (backlog dependency); see module docstring
_HEADER = (
    ["step", "sim_time"]
    + [f"q_M{i}" for i in range(N_MOVEMENTS)]
    + [f"c_M{i}" for i in range(N_MOVEMENTS)]
)


def generate_one(
    scenario: Scenario,
    seed: int,
    *,
    out_dir: Path = _OUT_DIR,
    episode_length_s: int | None = None,
) -> tuple[Path, int]:
    """Generate one CSV for ``(scenario, seed)``; return ``(path, n_rows)``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    route = write_routes(scenario, seed)
    ctrl = WebsterController(webster_plan_for_scenario(scenario))
    env = SUMOEnv(
        route,
        episode_length_s=episode_length_s or scenario.duration_s,
        sumo_seed=seed,
    )
    rows: list[str] = [",".join(_HEADER)]
    n_rows = 0
    try:
        obs, info = env.reset()
        ctrl.reset(env)
        done = False
        step = 0
        while not done:
            action = ctrl.select_action(obs, info["mask"])
            obs, _reward, terminated, truncated, info = env.step(action)
            queue, count = env.movement_features()
            step += 1
            cells = [str(step), f"{info['sim_time']:.0f}"]
            cells += [str(int(v)) for v in queue]
            cells += [str(int(v)) for v in count]
            rows.append(",".join(cells))
            n_rows += 1
            done = terminated or truncated
    finally:
        env.close()

    scn_num = scenario.id.split("-")[1]
    out_path = out_dir / f"scn_{scn_num}_seed_{seed:02d}.csv"
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out_path, n_rows


def _scenario_configs_hash() -> str:
    """Stable hash of all scenario YAMLs (a data_version input)."""
    return hash_files(list(SCENARIO_DIR.glob("scn_*.yaml")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", help="single scenario id (e.g. SCN-01); default all")
    parser.add_argument("--seed", type=int, help="single seed; default all in the scenario")
    parser.add_argument("--episode-length", type=int, help="override episode length (s), for smoke runs")
    args = parser.parse_args()

    build_net()  # ensure the net matches current sources
    scenarios = (
        [load_scenario(SCENARIO_DIR / f"scn_{args.scenario.split('-')[1]}.yaml")]
        if args.scenario
        else load_all()
    )

    cfg_hash = _scenario_configs_hash()
    code_sha = git_sha() or "unknown"
    sumo_v = sumo_version() or "unknown"

    manifest: list[dict] = []
    total = 0
    for scenario in scenarios:
        seeds = [args.seed] if args.seed is not None else list(scenario.seeds)
        for seed in seeds:
            path, n_rows = generate_one(scenario, seed, episode_length_s=args.episode_length)
            dv = data_version(
                scenario_configs_hash=cfg_hash,
                generator_git_sha=code_sha,
                generation_seed=seed,
                sumo_version=sumo_v,
            )
            manifest.append(
                {
                    "file": path.name,
                    "scenario": scenario.id,
                    "seed": seed,
                    "controller": CONTROLLER,
                    "n_rows": n_rows,
                    "data_version": dv,
                    "generator_git_sha": code_sha,
                    "sumo_version": sumo_v,
                    "scenario_configs_hash": cfg_hash,
                }
            )
            print(f"[lstm-data] {scenario.id} seed {seed:02d} -> {path.name} ({n_rows} rows, {dv})")
            total += 1

    manifest_path = _OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[lstm-data] OK - wrote {total} CSV(s) + manifest -> {manifest_path.relative_to(_REPO_ROOT)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
