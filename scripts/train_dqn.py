"""T-03-06 - CLI entry point for DQN training.

Wires the real simulation stack into the env factories the training loop
(:mod:`src.ml.train_loop`) consumes, then runs it. The loop itself is stack-agnostic
(it never imports ``scripts``); this script is the only place that knows about
``SUMOEnv`` / the hybrid wrapper (via :mod:`scripts.env_factory`), so the loop stays
unit-testable.

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

    # switch-penalty ablation (T-04-03), lambda = 0.5
    python -m scripts.train_dqn --seed 42 --switch-penalty 0.5 --variant lambda050

    # quick smoke (T-03-06 build verification: a few short episodes)
    python -m scripts.train_dqn --episodes 3 --episode-length 200 --validation-every 0 --no-log-steps

    # resume a crashed run from a checkpoint
    python -m scripts.train_dqn --seed 42 --resume runs/plain_seed42/checkpoints/ep200.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from scripts.build_network import build_net
from scripts.env_factory import build_env, load_scenario_by_id
from src.baselines.webster import WebsterController, webster_plan_for_scenario
from src.ml.hybrid_wrapper import load_forecaster, random_forecaster
from src.ml.train_loop import TrainConfig, train
from src.provenance.versions import git_sha

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _REPO_ROOT / "runs"


def _collect_webster_transitions(scenarios, seeds, episode_length):
    """Roll Webster through each (scenario, seed) and log (obs, webster_action, mask) per step.

    The behavior-cloning dataset for the Webster warm-start: cloning these (state -> Webster action)
    pairs starts the agent near Webster's robust policy. Covers the train scenarios + the (light)
    val scenario so the clone learns Webster's behaviour across demand regimes.
    """
    states, actions, masks = [], [], []
    for scn_id in scenarios:
        plan = webster_plan_for_scenario(load_scenario_by_id(scn_id))
        for seed in seeds:
            env = build_env(scn_id, seed, episode_length_s=episode_length)
            ctrl = WebsterController(plan)
            obs, info = env.reset()
            ctrl.reset(env)
            done = False
            while not done:
                mask = info["mask"]
                a = ctrl.select_action(obs, mask)
                states.append(np.asarray(obs, dtype=np.float32))
                actions.append(int(a))
                masks.append(np.asarray(mask, dtype=bool))
                obs, _r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
            env.close()
    return (np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(masks, dtype=bool))


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
    parser.add_argument("--switch-penalty", type=float, default=0.1,
                        help="reward switch penalty lambda (T-04-03 sweep; default 0.1)")
    parser.add_argument("--gridlock-penalty", type=float, default=0.0,
                        help="v2 anti-gridlock reward weight mu (0 = off, locked reward)")
    parser.add_argument("--gridlock-threshold", type=float, default=20.0,
                        help="per-movement queue threshold for the anti-gridlock penalty")
    parser.add_argument("--forecast-ckpt", default=None,
                        help="path to the frozen LSTM checkpoint -> 56-dim hybrid run")
    parser.add_argument("--random-lstm", action="store_true",
                        help="56-dim run with a frozen UNTRAINED forecaster (random-LSTM control)")
    parser.add_argument("--distributional", action="store_true",
                        help="T-03-09: train an IQN (risk-sensitive DQN); CVaR set at eval via --cvar-alpha")
    parser.add_argument("--cvar-alpha", type=float, default=1.0,
                        help="CVaR risk level for action-selection AND the bootstrap (1.0=risk-neutral; "
                             "<1 trains risk-averse end to end)")
    parser.add_argument("--warm-start-from", default=None,
                        help="plain-DQN checkpoint to transfer-init the IQN trunk/head (fixes undertraining)")
    parser.add_argument("--bc-warmstart-webster", action="store_true",
                        help="behavior-clone Webster into the agent before online RL (start at "
                             "Webster's robustness, then improve - the sess16 best-tradeoff pick)")
    parser.add_argument("--train-scenarios", nargs="+", default=None,
                        help="override the training scenario rotation (default SCN-01 SCN-02 SCN-03); "
                             "e.g. add a shifting scenario: --train-scenarios SCN-01 SCN-02 SCN-03 SCN-10")
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

    if args.random_lstm and args.forecast_ckpt:
        parser.error("--random-lstm and --forecast-ckpt are mutually exclusive")
    if args.random_lstm:
        variant = args.variant or "random-lstm"
        forecaster, fc_label = random_forecaster(seed=args.seed), "random-lstm"
    elif args.forecast_ckpt:
        variant = args.variant or "hybrid"
        forecaster, fc_label = load_forecaster(args.forecast_ckpt), args.forecast_ckpt
    else:
        default = "iqn-webster" if args.bc_warmstart_webster else ("iqn" if args.distributional else "plain")
        variant = args.variant or default
        forecaster, fc_label = None, None
    run_dir = (Path(args.run_dir) if args.run_dir
               else _RUNS_DIR / f"{variant}_seed{args.seed}").resolve()

    cfg = TrainConfig(
        variant=variant,
        seed=args.seed,
        n_episodes=args.episodes,
        episode_length_s=args.episode_length or 3600,
        switch_penalty=args.switch_penalty,
        gridlock_penalty_mu=args.gridlock_penalty,
        gridlock_queue_threshold=args.gridlock_threshold,
        train_scenarios=tuple(args.train_scenarios) if args.train_scenarios else ("SCN-01", "SCN-02", "SCN-03"),
        validation_every=args.validation_every,
        validation_episodes=args.validation_episodes,
        checkpoint_every=args.checkpoint_every,
        forecast=forecaster is not None,
        forecast_ckpt=fc_label,
        distributional=args.distributional,
        cvar_alpha=args.cvar_alpha,
        warm_start_ckpt=args.warm_start_from,
        bc_warmstart_controller="webster" if args.bc_warmstart_webster else None,
        log_steps=not args.no_log_steps,
        git_sha=git_sha(short=True) or "",
    )

    build_net()  # ensure the network file matches current sources (idempotent)

    def make_train_env(scenario_id: str, route_seed: int):
        return build_env(
            scenario_id, route_seed, forecaster=forecaster,
            episode_length_s=args.episode_length, switch_penalty=cfg.switch_penalty,
            gridlock_penalty_mu=cfg.gridlock_penalty_mu,
            gridlock_queue_threshold=cfg.gridlock_queue_threshold,
        )

    def make_val_env(route_seed: int):
        return build_env(
            cfg.val_scenario, route_seed, forecaster=forecaster,
            episode_length_s=args.episode_length, switch_penalty=cfg.switch_penalty,
            gridlock_penalty_mu=cfg.gridlock_penalty_mu,
            gridlock_queue_threshold=cfg.gridlock_queue_threshold,
        )

    bc_dataset = None
    if args.bc_warmstart_webster:
        scns = list(cfg.train_scenarios) + [cfg.val_scenario]
        print(f"[train] collecting Webster BC dataset over {scns} ...", flush=True)
        bc_dataset = _collect_webster_transitions(scns, seeds=(1, 2), episode_length=args.episode_length)
        print(f"[train] Webster BC dataset: {len(bc_dataset[0])} transitions", flush=True)

    print(f"[train] variant={variant} seed={args.seed} obs_dim={cfg.obs_dim} "
          f"episodes={cfg.n_episodes} lambda={cfg.switch_penalty} "
          f"eps_decay_steps={cfg.eps_decay_steps} -> {run_dir}")
    result = train(
        cfg,
        make_train_env=make_train_env,
        make_val_env=make_val_env,
        run_dir=run_dir,
        resume=args.resume,
        bc_dataset=bc_dataset,
    )
    print(f"[train] OK - {result.episodes_completed} episodes, {result.total_steps} steps, "
          f"best_val_reward={result.best_val_reward:.1f} -> {result.run_dir}")
    sys.exit(0)


if __name__ == "__main__":
    main()
