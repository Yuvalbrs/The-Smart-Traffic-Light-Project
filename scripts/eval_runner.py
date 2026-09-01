"""T-04-01 - Evaluation runner: the real, paired, persisted comparison.

Runs every algorithm through the same held-out eval episodes and writes one fully-provenanced
SQLite row per episode, so T-04-02 (Wilcoxon + Holm-Bonferroni) has clean paired data.

Algorithms (12): the 3 baselines (webster, max_pressure, actuated) + the 9 trained DQN models
({plain, hybrid, random-lstm} x train-seeds {42,123,2024}). DQN models are evaluated GREEDY
(epsilon=0) from their **final** checkpoint (``ep299.pt``) - NOT ``best.pt``, which is chosen by
the noisy single-draw validation metric (sess14 finding).

Pairing (evaluation-methodology.md / pre-registration): traffic for an eval episode is fixed by
``(scenario, eval_seed)`` and is **independent of the algorithm**, so every algorithm sees
byte-identical traffic for a given (scenario, eval_seed) - i.e. the samples are paired exactly as
the signed-rank test needs. Eval seeds are held-out (distinct from the training/validation route
seeds), so this measures generalization, including to SCN-04/05 the DQN never trained on
(SCN-05 = the designated test scenario).

Each episode -> ``episode`` + ``episode_kpi`` rows under one ``experiment_run`` per algorithm
(mode="eval", full version chain), plus a flat ``data/eval/eval_results.csv`` for T-04-02.

Run::

    LIBSUMO_AS_TRACI=1 python -m scripts.eval_runner                      # full: 5 scn x 5 seeds x 12 algo
    LIBSUMO_AS_TRACI=1 python -m scripts.eval_runner --scenarios SCN-05   # one scenario
    LIBSUMO_AS_TRACI=1 python -m scripts.eval_runner --scenarios SCN-01 --eval-seeds 7000 --episode-length 600  # smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sqlalchemy.orm import Session

from scripts.build_actuated import build_actuated_add
from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.baselines.max_pressure import MaxPressureController
from src.baselines.webster import WebsterController, webster_plan_for_scenario
from src.db.engine import create_db_engine, init_db
from src.db.models import Episode, EpisodeKpi
from src.env.sumo_env import SUMOEnv
from src.metrics.kpi_extractor import EpisodeKPIs, extract_kpis
from src.ml.dqn import OBS_DIM, DQNAgent
from src.ml.hybrid_wrapper import HYBRID_OBS_DIM, load_forecaster, random_forecaster
from src.provenance.official import official_lstm_checked, official_lstm_filename
from src.provenance.records import record_experiment_run
from src.provenance.versions import git_sha, sumo_version
from src.scenarios.config import SCENARIO_DIR, Scenario, load_all, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _REPO_ROOT / "runs"
_OUT_DIR = _REPO_ROOT / "data" / "eval"
_DEFAULT_DB = _REPO_ROOT / "data" / "traffic.db"
# The pre-registered confirmatory scenario set (prereg s2). SCN-05 is the held-out
# test regime; SCN-01..03 are the training regimes; SCN-04 is the interpolation check.
CONFIRMATORY_SCENARIOS = ("SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05")
TRAIN_SEEDS = (42, 123, 2024)
DQN_VARIANTS = ("plain", "hybrid", "random-lstm")
DEFAULT_EVAL_SEEDS = tuple(range(7000, 7015))  # prereg A1.3 (2026-09-01): 15 held-out seeds.
# n=5 (the pre-amendment block 7000-7004) cannot reject at alpha=0.05 under ANY data once train
# seeds stop being counted as replication: min attainable two-sided signed-rank p = 2/2**5 = 0.0625.

# (EpisodeKPIs attribute, episode_kpi column) - scalar KPIs only; the [12] lists are stored too.
_SCALAR_KPIS = (
    "avg_waiting_time", "avg_queue_length", "throughput", "num_stops",
    "wait_p95", "fairness_std", "worst_movement_max_wait",
)


@dataclass
class Algo:
    """One thing to evaluate: a baseline controller or a loaded DQN agent."""

    name: str                       # webster | max_pressure | actuated | dqn-<variant>-s<seed>
    controller_kind: str            # baseline | dqn
    controller: object | None = None    # baseline controller (webster/max_pressure) or None
    agent: DQNAgent | None = None       # loaded DQN (greedy) or None
    forecaster: object | None = None    # frozen LSTM for hybrid/random-lstm, else None
    signal_mode: str = "rl"             # "actuated" only for the actuated baseline
    variant: str | None = None
    train_seed: int | None = None
    lstm_version: str | None = None
    ckpt: str | None = None


def _load_agent(ckpt: Path, obs_dim: int, *, cvar_alpha: float | None = None) -> DQNAgent:
    """Load a trained DQN's online net (greedy eval); the target/optimizer are not needed.

    Reads the checkpoint's own ``config`` so a distributional (IQN) checkpoint rebuilds the right
    architecture automatically. ``cvar_alpha`` overrides the saved risk level - the lever for the
    eval-time alpha sweep (one trained IQN, many risk levels) since CVaR is an action-selection knob.
    """
    state = torch.load(ckpt, map_location="cpu")
    cfg = state.get("config", {}) or {}
    distributional = bool(cfg.get("distributional", False))
    alpha = cvar_alpha if cvar_alpha is not None else float(cfg.get("cvar_alpha", 1.0))
    agent = DQNAgent(obs_dim, distributional=distributional, cvar_alpha=alpha)
    agent.online.load_state_dict(state["online"])
    agent.online.eval()
    return agent


def _forecaster_for_variant(variant: str, seed: int):
    """``(forecaster, lstm_version)`` for one variant at one training seed.

    Shared by the campaign's nine cells and by a single user-trained model, because these two
    paths choosing their forecaster independently is exactly how the arms drift apart: the pin is
    per training seed (A6.4), and a copy of this logic that forgets that would evaluate a seed-123
    agent against seed 42's forecaster and record the wrong lstm_version beside the result.
    """
    if variant == "plain":
        return None, None
    if variant == "hybrid":
        return load_forecaster(str(official_lstm_checked(seed))), official_lstm_filename(seed)
    if variant == "random-lstm":
        # seed-matched AND stats-matched: both must mirror train_matrix._forecaster_for exactly,
        # or the agent is evaluated on inputs it never saw.
        return (
            random_forecaster(seed=seed, stats_from=load_forecaster(str(official_lstm_checked(seed)))),
            "random-lstm",
        )
    raise ValueError(f"unknown variant {variant!r}")


def user_model_algo(run_dir: Path, label: str) -> Algo:
    """Build ONE evaluee from a run directory the user trained in the application.

    The controller name is prefixed ``ui:`` so these rows are distinguishable from the campaign's
    in every downstream query. They are NOT campaign rows: the model was trained with a different
    episode budget, at a different time, on whatever code was current, and merging it into a
    campaign average would silently corrupt a pre-registered result.
    """
    import yaml

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"{run_dir} has no config.yaml - not a run directory produced by train_dqn")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    variant = str(cfg.get("variant") or "plain")
    seed = int(cfg.get("seed", 42))

    ckpts = sorted((run_dir / "checkpoints").glob("ep*.pt"), key=lambda q: int(q.stem[2:]))
    if not ckpts:
        raise SystemExit(f"{run_dir} has no checkpoints/ep*.pt - train it before evaluating")
    ckpt = ckpts[-1]  # the furthest-trained checkpoint, which for a finished run is the final one

    obs_dim = OBS_DIM if variant == "plain" else HYBRID_OBS_DIM
    forecaster, lstm_version = _forecaster_for_variant(variant, seed)
    return Algo(
        name=f"ui:{label}", controller_kind="dqn", agent=_load_agent(ckpt, obs_dim),
        forecaster=forecaster, variant=f"ui:{label}", train_seed=seed,
        lstm_version=lstm_version, ckpt=str(ckpt),
    )


def _dqn_algos() -> list[Algo]:
    """Build the 9 DQN evaluees from their final (ep299) checkpoints; skip any missing."""
    algos: list[Algo] = []
    for variant in DQN_VARIANTS:
        for seed in TRAIN_SEEDS:
            ckpt = _RUNS_DIR / f"{variant}_seed{seed}" / "checkpoints" / "ep299.pt"
            if not ckpt.exists():
                print(f"[eval] WARN missing checkpoint, skipping: {ckpt}")
                continue
            obs_dim = OBS_DIM if variant == "plain" else HYBRID_OBS_DIM
            # A6.4: the pin is per DQN training seed - loading seed 42's checkpoint for every
            # hybrid row (the old bug) recorded identical lstm_version for three arms that were
            # actually trained against three different forecasters.
            forecaster, lstm_version = _forecaster_for_variant(variant, seed)
            algos.append(Algo(
                name=f"dqn-{variant}-s{seed}", controller_kind="dqn",
                agent=_load_agent(ckpt, obs_dim), forecaster=forecaster,
                variant=variant, train_seed=seed, lstm_version=lstm_version, ckpt=str(ckpt),
            ))
    return algos


def _baseline_algos(scenario: Scenario) -> list[Algo]:
    """Baselines for one scenario (webster's plan is scenario-specific)."""
    return [
        Algo("webster", "baseline", controller=WebsterController(webster_plan_for_scenario(scenario))),
        Algo("max_pressure", "baseline", controller=MaxPressureController.from_spec()),
        Algo("actuated", "baseline", signal_mode="actuated"),
    ]


def run_eval_episode(
    scenario: Scenario, eval_seed: int, algo: Algo, *,
    work_dir: Path, episode_length_s: int, warmup_s: float,
) -> tuple[EpisodeKPIs, float]:
    """Run one (scenario, eval_seed, algo) episode greedily; return (KPIs, total_reward)."""
    route = write_routes(scenario, eval_seed)
    stem = f"{scenario.id}_seed{eval_seed}_{algo.name}"
    jsonl = work_dir / f"{stem}.jsonl"
    tripinfo = work_dir / f"{stem}.tripinfo.xml"

    env = SUMOEnv(
        route, episode_length_s=episode_length_s, sumo_seed=eval_seed,
        signal_mode=algo.signal_mode, trace_path=jsonl, tripinfo_path=tripinfo,
    )
    if algo.forecaster is not None:
        from src.ml.hybrid_wrapper import HybridStateWrapper
        env = HybridStateWrapper(env, algo.forecaster)

    total_reward = 0.0
    counters = None
    try:
        obs, info = env.reset()
        if algo.controller is not None:
            algo.controller.reset(env)
        done = False
        while not done:
            mask = info["mask"]
            if algo.agent is not None:
                action = algo.agent.act(obs, mask, epsilon=0.0)  # greedy
            elif algo.controller is not None:
                action = algo.controller.select_action(obs, mask)
            else:
                action = 0  # actuated: SUMO drives the lights, action is ignored
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        counters = info.get("episode")
    finally:
        env.close()  # SUMO finalizes the trip-info XML here

    kpis = extract_kpis(
        jsonl, tripinfo, episode_counters=counters,
        warmup_s=warmup_s, episode_length_s=episode_length_s,
    )
    return kpis, total_reward


def _persist(session: Session, run, *, index: int, eval_seed: int, scenario_id: str,
             total_reward: float, kpis: EpisodeKPIs) -> None:
    """Write one episode + its 1:1 KPI row (full KPI set incl. the E5 columns)."""
    ep = Episode(
        run=run, index_in_run=index, seed=eval_seed, scenario=scenario_id,
        total_reward=total_reward,
        insertion_backlog_fraction=kpis.insertion_backlog_fraction,
        gridlock_censored=bool(kpis.gridlock_censored),
    )
    ep.kpi = EpisodeKpi(
        avg_waiting_time=kpis.avg_waiting_time, avg_queue_length=kpis.avg_queue_length,
        throughput=kpis.throughput, num_stops=kpis.num_stops, wait_p95=kpis.wait_p95,
        fairness_std=kpis.fairness_std, per_movement_max_wait=kpis.per_movement_max_wait,
        per_movement_p95_wait=kpis.per_movement_p95_wait,
        worst_movement_max_wait=kpis.worst_movement_max_wait,
    )
    session.add(ep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenarios", nargs="+", default=None, help="scenario ids; default all 5")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--episode-length", type=int, default=None, help="override (for smoke)")
    parser.add_argument("--warmup", type=float, default=300.0)
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--no-dqn", action="store_true", help="baselines only")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="evaluate ONE user-trained run directory instead of the 9 campaign cells")
    parser.add_argument("--model-label", default=None,
                        help="name for the --model-dir evaluee; defaults to the directory name")
    args = parser.parse_args()

    build_net()
    build_actuated_add()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The confirmatory set is FIXED, not a directory glob: load_all() returns all 14
    # scenarios including SCN-A1..A4, which use the arterial two-junction schema while
    # this runner builds the single-intersection net and single-junction Webster plans.
    # The default campaign is 5 scenarios x 5 seeds x 12 algos = 300 rows (prereg s2).
    scenarios = (
        [load_scenario(SCENARIO_DIR / f"scn_{s.split('-')[1]}.yaml") for s in args.scenarios]
        if args.scenarios
        else [s for s in load_all() if s.id in CONFIRMATORY_SCENARIOS]
    )
    missing = {s.id for s in scenarios} - set(CONFIRMATORY_SCENARIOS)
    if not args.scenarios and missing:
        raise SystemExit(f"confirmatory set resolved to unexpected scenarios: {missing}")
    if any(s.is_arterial for s in scenarios):
        raise SystemExit(
            "arterial scenarios need the two-junction net and per-junction Webster plans; "
            "this runner is single-intersection only."
        )
    if args.model_dir is not None:
        label = args.model_label or Path(args.model_dir).name
        dqn_algos = [user_model_algo(Path(args.model_dir), label)]
        print(f"[eval] single user model: {dqn_algos[0].name} <- {args.model_dir}")
    else:
        dqn_algos = [] if args.no_dqn else _dqn_algos()

    engine = create_db_engine(args.db)
    init_db(engine)
    sha, sv = git_sha(short=True), sumo_version()

    csv_path = _OUT_DIR / "eval_results.csv"
    # A scoped/smoke run used to truncate the full campaign CSV in place, leaving the
    # analysis to build confident tables from 12 rows. Partial runs write elsewhere.
    is_full_run = not args.scenarios and not args.no_dqn and args.model_dir is None
    if not is_full_run:
        tag = (
            f"model-{args.model_label or Path(args.model_dir).name}" if args.model_dir is not None
            else "-".join(args.scenarios) if args.scenarios
            else "no-dqn"
        )
        csv_path = _OUT_DIR / f"eval_results_partial_{tag}.csv"
        print(f"[eval] partial run -> {csv_path.name} (the campaign CSV is left intact)")
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    # Provenance travels WITH the rows: the CSV is the only artifact the analysis reads,
    # so a chain that lives only in SQLite cannot satisfy prereg s9, and cannot stop the
    # analysis from silently running on a pre-fix campaign (it did, for months).
    csv_w.writerow(["algo", "variant", "train_seed", "scenario", "eval_seed", "total_reward",
                    *(_SCALAR_KPIS), "gridlock_censored",
                    "gridlock_network", "network_jam_fraction", "insertion_backlog_fraction",
                    "loaded_count", "departed_count", "arrived_count",
                    "n_vehicles_after_warmup", "n_unfinished_after_warmup",
                    "avg_waiting_time_completed", "wait_p95_completed",
                    "lstm_version", "git_sha", "sumo_version"])

    n_rows = 0
    with Session(engine) as session:
        # one experiment_run per algorithm name (provenance chain); reused across scenarios.
        runs: dict[str, object] = {}

        def _run_for(algo: Algo):
            if algo.name not in runs:
                runs[algo.name] = record_experiment_run(
                    session, name=f"eval-{algo.name}", mode="eval",
                    controller=algo.variant or algo.name,
                    config={"variant": algo.variant, "train_seed": algo.train_seed,
                            "ckpt": algo.ckpt, "eval_seeds": args.eval_seeds},
                    run_id=str(uuid.uuid4()), lstm_version=algo.lstm_version,
                    git_sha=sha, sumo_version=sv,
                )
            return runs[algo.name]

        index: dict[str, int] = {}
        for scenario in scenarios:
            episode_length_s = args.episode_length or scenario.duration_s
            warmup = args.warmup if args.warmup < episode_length_s else 0.0
            algos = _baseline_algos(scenario) + dqn_algos
            print(f"\n[eval] === {scenario.id} : {len(algos)} algos x {len(args.eval_seeds)} seeds ===")
            for eval_seed in args.eval_seeds:
                for algo in algos:  # shared (scenario, eval_seed) traffic -> paired samples
                    kpis, total_reward = run_eval_episode(
                        scenario, eval_seed, algo, work_dir=_OUT_DIR,
                        episode_length_s=episode_length_s, warmup_s=warmup,
                    )
                    run = _run_for(algo)
                    idx = index.get(algo.name, 0)
                    index[algo.name] = idx + 1
                    _persist(session, run, index=idx, eval_seed=eval_seed,
                             scenario_id=scenario.id, total_reward=total_reward, kpis=kpis)
                    csv_w.writerow([
                        algo.name, algo.variant or "", algo.train_seed if algo.train_seed else "",
                        scenario.id, eval_seed, round(total_reward, 1),
                        *[getattr(kpis, k) for k in _SCALAR_KPIS], int(kpis.gridlock_censored),
                        int(kpis.gridlock_network), kpis.network_jam_fraction,
                        kpis.insertion_backlog_fraction,
                        kpis.loaded_count, kpis.departed_count, kpis.arrived_count,
                        kpis.n_vehicles_after_warmup, kpis.n_unfinished_after_warmup,
                        kpis.avg_waiting_time_completed, kpis.wait_p95_completed,
                        algo.lstm_version or "", sha, sv,
                    ])
                    csv_f.flush()
                    n_rows += 1
                    flag = " CENSORED" if kpis.gridlock_censored else ""
                    print(f"[eval] {scenario.id} seed{eval_seed} {algo.name:18} "
                          f"wait={kpis.avg_waiting_time:6.1f} thru={kpis.throughput:7.1f}{flag}")
                session.commit()  # persist after each (scenario, eval_seed) group

    csv_f.close()
    print(f"\n[eval] OK - {n_rows} episodes -> {args.db} + {csv_path.relative_to(_REPO_ROOT)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
