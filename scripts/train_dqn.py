"""T-03-06 - CLI entry point for DQN training.

Wires the real simulation stack into the env factories the training loop
(:mod:`src.ml.train_loop`) consumes, then runs it. The loop itself is stack-agnostic
(it never imports ``scripts``); this script is the only place that knows about
``write_routes`` / ``SUMOEnv`` / the hybrid wrapper, so the loop stays unit-testable.

Scenario rotation is by route file (no ``set_scenario`` on the real env): each episode the
loop asks for ``make_train_env(scenario_id, route_seed)`` and we generate that scenario's
deterministic ``.rou.xml`` and open a fresh ``SUMOEnv`` on it - the ``eval_baselines.py``
pattern. With ``--forecast-ckpt`` set, the env is wrapped in the 56-dim
:class:`HybridStateWrapper` (the locked with/without-forecast ablation: omit it for the
20-dim plain DQN).

Run::

    # plain 20-dim DQN, full 300-episode run, seed 42
    python -m scripts.train_dqn --seed 42

    # 56-dim hybrid run with the frozen forecaster
    python -m scripts.train_dqn --seed 42 --variant hybrid --forecast-ckpt checkpoints/lstm-XXXX.pt

    # quick smoke (T-03-06 build verification: a few short episodes)
    python -m scripts.train_dqn --episodes 3 --episode-length 200 --validation-every 0 --no-log-steps

    # resume a crashed run from a checkpoint
    python -m scripts.train_dqn --seed 42 --resume runs/plain_seed42/checkpoints/ep200.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.env.sumo_env import SUMOEnv
from src.ml.hybrid_wrapper import HybridStateWrapper, load_forecaster
from src.ml.train_loop import TrainConfig, train
from src.provenance.versions import git_sha
from src.scenarios.config import SCENARIO_DIR, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _REPO_ROOT / "runs"


def _scenario(scenario_id: str):
    """Load a scenario by id (e.g. ``"SCN-01"`` -> ``config/scenarios/scn_01.yaml``)."""
    return load_scenario(SCENARIO_DIR / f"scn_{scenario_id.split('-')[1]}.yaml")


def _make_env(
    scenario_id: str, route_seed: int, *, episode_length_s: int | None, forecaster
):
    """Build one ``SUMOEnv`` (optionally hybrid-wrapped) on a fresh deterministic route file."""
    scenario = _scenario(scenario_id)
    route = write_routes(scenario, route_seed)
    env = SUMOEnv(
        route,
        episode_length_s=episode_length_s or scenario.duration_s,
        decision_interval_s=10,
        switch_penalty=0.1,
        sumo_seed=route_seed,
        signal_mode="rl",
    )
    if forecaster is not None:
        env = HybridStateWrapper(env, forecaster)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42, help="run seed (default 42)")
    parser.add_argument("--variant", default=None,
                        help="run label for the run dir; default 'hybrid' if --forecast-ckpt else 'plain'")
    parser.add_argument("--episodes", type=int, default=300, help="number of episodes (default 300)")
    parser.add_argument("--episode-length", type=int, default=None,
                        help="override episode length in sim-seconds (for smoke runs)")
    parser.add_argument("--forecast-ckpt", default=None,
                        help="path to the frozen LSTM checkpoint -> 56-dim hybrid run")
    parser.add_argument("--validation-every", type=int, default=25,
                        help="validate every N episodes (0 disables; default 25)")
    parser.add_argument("--validation-episodes", type=int, default=5,
                        help="greedy validation episodes per check (default 5)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="checkpoint every N episodes (default 50)")
    parser.add_argument("--no-log-steps", action="store_true",
                        help="skip the per-step diagnostics CSV (smaller output)")
    parser.add_argument("--run-dir", default=None,
                        help="output dir; default runs/{variant}_seed{seed}")
    parser.add_argument("--resume", default=None, help="resume from this checkpoint .pt")
    args = parser.parse_args()

    variant = args.variant or ("hybrid" if args.forecast_ckpt else "plain")
    run_dir = Path(args.run_dir) if args.run_dir else _RUNS_DIR / f"{variant}_seed{args.seed}"

    cfg = TrainConfig(
        variant=variant,
        seed=args.seed,
        n_episodes=args.episodes,
        episode_length_s=args.episode_length or 3600,
        validation_every=args.validation_every,
        validation_episodes=args.validation_episodes,
        checkpoint_every=args.checkpoint_every,
        forecast_ckpt=args.forecast_ckpt,
        log_steps=not args.no_log_steps,
        git_sha=git_sha(short=True) or "",
    )

    build_net()  # ensure the network file matches current sources (idempotent)
    forecaster = load_forecaster(args.forecast_ckpt) if args.forecast_ckpt else None

    def make_train_env(scenario_id: str, route_seed: int):
        return _make_env(
            scenario_id, route_seed,
            episode_length_s=args.episode_length, forecaster=forecaster,
        )

    def make_val_env(route_seed: int):
        return _make_env(
            cfg.val_scenario, route_seed,
            episode_length_s=args.episode_length, forecaster=forecaster,
        )

    print(f"[train] variant={variant} seed={args.seed} obs_dim={cfg.obs_dim} "
          f"episodes={cfg.n_episodes} eps_decay_steps={cfg.eps_decay_steps} -> {run_dir}")
    result = train(
        cfg,
        make_train_env=make_train_env,
        make_val_env=make_val_env,
        run_dir=run_dir,
        resume=args.resume,
    )
    print(f"[train] OK - {result.episodes_completed} episodes, {result.total_steps} steps, "
          f"best_val_reward={result.best_val_reward:.1f} -> {result.run_dir}")
    sys.exit(0)


if __name__ == "__main__":
    main()
