"""T-03-06 - The DQN training loop: epsilon-decay, target sync, validation, checkpointing.

The ignition for the assembled Phase-3 machine: it drives ``agent.act -> buffer.push ->
buffer.sample -> agent.learn -> agent.sync_target`` over 300 episodes, decays exploration,
validates on the held-out scenario, and writes resumable checkpoints + CSV diagnostics.

This module is the reconciliation of the **stale** ``training-infrastructure.md`` pseudocode
against the real code (the divergence hot.md / the T-03-05 DoD flagged). The notes are wrong
in several load-bearing places; what actually holds:

* The env is **route-file bound** - there is no ``SUMOEnv(scenario_id=...)`` / ``set_scenario()``.
  Scenario rotation = a fresh env per episode (built by the injected ``make_train_env``), exactly
  the ``eval_baselines.py`` pattern. This module never imports ``scripts`` (correct layering);
  the caller passes env factories, which also makes the loop unit-testable with a dummy env.
* The agent has **no** ``epsilon`` / ``decay_epsilon`` / ``select_action`` / ``update`` / ``save``.
  Exploration epsilon lives entirely here (:func:`epsilon_at`); the agent exposes ``act`` /
  ``learn`` / ``sync_target`` only, and checkpointing is this module's job.
* ``reset()`` returns ``(obs, info)`` and ``step()`` the gymnasium 5-tuple. ``info`` carries the
  action ``mask`` (also via ``env.get_action_mask()``) but **not** ``mean_queue`` / ``mean_wait`` /
  ``throughput`` / ``ep_length`` - those come from the KPI extractor over a trace, not from the
  loop. So validation here is reward-only (the locked "best-by-validation-reward" rule); the rich
  KPIs are T-04-01's job.

Two correctness details the stale single-``done`` pseudocode gets wrong and we do not:

* **Terminated vs truncated.** The Bellman bootstrap must be zeroed only on a *true* terminal
  (all vehicles cleared, ``getMinExpectedNumber()==0`` -> ``terminated``), NOT on the time-limit
  cutoff (``truncated``), whose successor still has value. We push ``terminated`` as the done flag
  but end the episode on ``terminated or truncated``.
* **Mask the next state.** Already handled in :class:`~src.ml.dqn.DQNAgent` (target mask before the
  max); here we only store ``next_mask`` and assert ``mask.any()`` every step (the DoD guard).

Diagnostics (DoD): per-step ``loss`` / ``grad_norm`` / ``q_mean`` / ``q_max`` (from
``agent.learn(..., return_diagnostics=True)``) and a rolling forecast skill score ``SS_rolling``
(:class:`ForecastSkillTracker`) are logged to CSV. Checkpoints carry optimizer + epsilon-via-step
+ RNG state so ``--resume`` continues a crashed run instead of restarting from zero. The replay
buffer is intentionally NOT checkpointed (50 MB; standard DQN-resume practice) - it warms up again
on resume while the agent weights/optimizer/step-count carry the learned state.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from src.ml.dqn import N_PHASES, OBS_DIM, DQNAgent
from src.ml.lstm_data import DEFAULT_TARGET_OFFSETS
from src.ml.replay_buffer import CAPACITY, MIN_REPLAY, ReplayBuffer

# An env factory: ``make_train_env(scenario_id, route_seed) -> gym.Env`` (and the validation
# variant ``make_val_env(route_seed) -> gym.Env``). The caller (scripts/train_dqn.py) binds
# write_routes + SUMOEnv + the optional HybridStateWrapper into these.
TrainEnvFactory = Callable[[str, int], Any]
ValEnvFactory = Callable[[int], Any]

_VAL_SEED_BASE = 990_000  # validation traffic seeds (fixed -> comparable across checkpoints)
_ROUTE_SEED_STRIDE = 10_000  # route_seed = run_seed * stride + episode -> per-run-disjoint traffic


@dataclass
class TrainConfig:
    """All hyperparameters for one training run. Serialized to ``config.yaml`` per the DoD.

    Defaults are the locked values (training-infrastructure.md hyperparameter table /
    decisions.md). ``forecast_ckpt`` set => 56-dim hybrid run; ``None`` => 20-dim plain DQN
    (the locked with/without-forecast ablation). ``obs_dim`` is derived in ``__post_init__``.
    """

    variant: str = "plain"  # informational label for the run dir / config record
    seed: int = 42  # run seed: agent init + exploration RNG + per-episode route seeds
    n_episodes: int = 300
    episode_length_s: int = 3600
    decision_interval_s: int = 10  # used only to estimate steps/episode for the eps schedule

    # agent / optimization (forwarded to DQNAgent)
    gamma: float = 0.99
    lr: float = 1e-4
    grad_clip: float = 10.0
    batch_size: int = 64

    # distributional / risk-sensitive (T-03-09): IQN + CVaR action-selection
    distributional: bool = False  # True -> IQN agent (quantile-Huber loss, CVaR act)
    cvar_alpha: float = 1.0        # CVaR risk level for act() + bootstrap; 1.0 = risk-neutral.
    warm_start_ckpt: str | None = None  # plain-DQN checkpoint to transfer-init the IQN trunk/head
    bc_warmstart_controller: str | None = None  # guide cloned before online RL (e.g. "webster")
    bc_epochs: int = 10            # behavior-cloning pretrain epochs when a guide dataset is given

    # replay
    buffer_capacity: int = CAPACITY
    min_replay: int = MIN_REPLAY
    target_update_freq: int = 1000

    # exploration (linear eps_start -> eps_end over eps_decay_steps GLOBAL steps)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int | None = None  # None -> 50% of n_episodes * steps/episode

    # env reward shaping (informational here; the factory applies it to SUMOEnv)
    switch_penalty: float = 0.1
    gridlock_penalty_mu: float = 0.0       # v2 anti-gridlock shaping; 0 = locked reward unchanged
    gridlock_queue_threshold: float = 20.0  # per-movement queue beyond which the penalty applies

    # scenarios / validation / checkpointing
    train_scenarios: tuple[str, ...] = ("SCN-01", "SCN-02", "SCN-03")
    val_scenario: str = "SCN-04"
    validation_every: int = 25
    validation_episodes: int = 5
    checkpoint_every: int = 50

    # forecast - path/label of the frozen LSTM, or None for plain DQN
    forecast_ckpt: str | None = None
    # True for any 56-dim forecast run (hybrid OR the random-LSTM control); auto-set from
    # forecast_ckpt for back-compat. Drives obs_dim (56 vs 20) and the SS_rolling tracker.
    forecast: bool = False

    # logging
    log_steps: bool = True  # write per-step diagnostics to steps.csv

    # provenance (best-effort; recorded into config.yaml + checkpoints, T-01-06 chain)
    git_sha: str = ""
    lstm_version: str = ""

    obs_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.forecast = bool(self.forecast or self.forecast_ckpt)
        self.obs_dim = 56 if self.forecast else OBS_DIM
        if self.eps_decay_steps is None:
            steps_per_ep = max(1, self.episode_length_s // self.decision_interval_s)
            self.eps_decay_steps = int(0.5 * self.n_episodes * steps_per_ep)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view for YAML serialization (tuples -> lists)."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RunResult:
    """Summary of a finished (or smoke) run."""

    run_dir: Path
    episodes_completed: int
    total_steps: int
    best_val_reward: float


def epsilon_at(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """Linear epsilon schedule: ``eps_start`` -> ``eps_end`` over ``decay_steps`` global steps."""
    if decay_steps <= 0:
        return eps_end
    frac = min(1.0, step / decay_steps)
    return eps_start + (eps_end - eps_start) * frac


class ForecastSkillTracker:
    """Rolling forecast skill score ``SS = 1 - MSE_forecast / MSE_persistence`` during training.

    For each step the forecaster predicts the queue at ``offsets`` decision steps ahead; we hold
    each prediction until its target step arrives, then score it against the realized queue and
    against the persistence baseline (predict "future queue = queue now"). The same definition as
    :func:`src.ml.lstm_model.skill_scores`, but computed online over a moving window so the loop
    can log whether the *frozen* forecaster still has skill under the DQN-driven distribution.

    ``SS > 0`` means the forecast beats persistence; ``<= 0`` means it does not (the open-items E1
    failure mode). Returns ``None`` until at least one prediction has resolved.
    """

    def __init__(self, offsets: tuple[int, ...] = DEFAULT_TARGET_OFFSETS, *, window: int = 200) -> None:
        self._offsets = offsets
        self._pending: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._fse: deque[float] = deque(maxlen=window)  # forecast squared errors
        self._pse: deque[float] = deque(maxlen=window)  # persistence squared errors
        self._step = 0

    def reset_episode(self) -> None:
        """Drop pending predictions at an episode boundary (they cannot resolve across reset)."""
        self._pending.clear()
        self._step = 0

    def update(self, realized_queue: np.ndarray, forecast: np.ndarray | None) -> float | None:
        """Record realized queue + a new ``(3, 12)`` raw forecast (or ``None`` at cold start).

        Resolves any prediction whose horizon lands on the current step, then returns the current
        rolling SS (or ``None`` if nothing has resolved yet).
        """
        s = self._step
        realized = np.asarray(realized_queue, dtype=np.float64)
        for origin_q, pred_q in self._pending.pop(s, []):
            self._fse.append(float(np.mean((pred_q - realized) ** 2)))
            self._pse.append(float(np.mean((origin_q - realized) ** 2)))  # persistence = origin
        if forecast is not None:
            for i, off in enumerate(self._offsets):
                self._pending.setdefault(s + off, []).append(
                    (realized.copy(), np.asarray(forecast[i], dtype=np.float64))
                )
        self._step += 1
        return self.value()

    def value(self) -> float | None:
        """Current rolling skill score, or ``None`` if no prediction has resolved yet."""
        if not self._pse:
            return None
        sp = sum(self._pse)
        return 1.0 - sum(self._fse) / sp if sp > 1e-12 else None


def _set_global_seeds(seed: int) -> None:
    """Seed python/numpy/torch for end-to-end reproducibility (cudnn flags are CPU no-ops)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    path: str | Path,
    agent: DQNAgent,
    *,
    episode: int,
    total_steps: int,
    best_val_reward: float,
    epsilon: float,
    cfg: TrainConfig,
) -> None:
    """Write a resumable checkpoint: online+target+optimizer weights, step/episode, RNG, config.

    Carries optimizer + epsilon-via-``total_steps`` + RNG state so ``--resume`` continues exactly
    (a crash at ep 249 resumes at ep 250). The frozen LSTM is a separate artifact and is NOT
    embedded (training-infrastructure.md); the replay buffer is intentionally not saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "total_steps": total_steps,
            "best_val_reward": best_val_reward,
            "epsilon": epsilon,  # informational; the live value is recomputed from total_steps
            "online": agent.online.state_dict(),
            "target": agent.target.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "agent_rng": agent._rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "config": cfg.to_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, agent: DQNAgent) -> dict[str, Any]:
    """Restore agent weights/optimizer/RNG from a checkpoint; return resume metadata.

    Returns ``{"episode", "total_steps", "best_val_reward", "config"}``. The caller resumes the
    loop at ``episode + 1``; the replay buffer warms up again from empty.
    """
    ckpt = torch.load(path, map_location=agent.device)
    agent.online.load_state_dict(ckpt["online"])
    agent.target.load_state_dict(ckpt["target"])
    agent.optimizer.load_state_dict(ckpt["optimizer"])
    agent._rng.setstate(ckpt["agent_rng"])
    torch.set_rng_state(ckpt["torch_rng"])
    return {
        "episode": ckpt["episode"],
        "total_steps": ckpt["total_steps"],
        "best_val_reward": ckpt["best_val_reward"],
        "config": ckpt["config"],
    }


def validate(agent: DQNAgent, cfg: TrainConfig, make_val_env: ValEnvFactory) -> tuple[float, float]:
    """Greedy (epsilon=0) evaluation on the held-out scenario; return ``(mean_reward, std_reward)``.

    Reward-only by design (the locked best-by-validation-reward rule): fixed validation seeds so
    the traffic is identical across checkpoints and the comparison is apples-to-apples.
    """
    rewards: list[float] = []
    for i in range(cfg.validation_episodes):
        env = make_val_env(_VAL_SEED_BASE + i)
        try:
            obs, info = env.reset()
            mask = info["mask"]
            done = False
            total = 0.0
            while not done:
                action = agent.act(obs, mask, epsilon=0.0)  # greedy
                obs, reward, terminated, truncated, info = env.step(action)
                mask = info["mask"]
                total += reward
                done = terminated or truncated
            rewards.append(total)
        finally:
            env.close()
    arr = np.asarray(rewards, dtype=float)
    return float(arr.mean()), float(arr.std())


def _mean(values: list[float]) -> float | str:
    """Mean of a list, or ``""`` (empty CSV cell) when nothing was logged yet (pre-warmup)."""
    return float(np.mean(values)) if values else ""


def _ss_cell(ss: float | None) -> float | str:
    return ss if ss is not None else ""


def train(
    cfg: TrainConfig,
    *,
    make_train_env: TrainEnvFactory,
    make_val_env: ValEnvFactory,
    run_dir: str | Path,
    resume: str | Path | None = None,
    bc_dataset: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> RunResult:
    """Run the full DQN training loop and return a :class:`RunResult`.

    Writes ``config.yaml`` + ``episodes.csv`` + ``steps.csv`` + ``validation.csv`` and checkpoints
    under ``run_dir/checkpoints``. ``resume`` continues a crashed run from a checkpoint (the loop
    restarts at the saved ``episode + 1``; CSVs are appended, the buffer warms up again).
    """
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _set_global_seeds(cfg.seed)

    agent = DQNAgent(
        cfg.obs_dim, gamma=cfg.gamma, lr=cfg.lr, grad_clip=cfg.grad_clip, seed=cfg.seed,
        distributional=cfg.distributional, cvar_alpha=cfg.cvar_alpha,
    )
    # the knob must reach the machine (sess14 scar): config records distributional/cvar_alpha,
    # so assert the constructed agent actually matches what config.yaml will claim.
    assert agent.distributional == cfg.distributional, "distributional flag did not reach the agent"
    if cfg.distributional and cfg.warm_start_ckpt:  # transfer-init from a trained plain DQN
        warm = torch.load(cfg.warm_start_ckpt, map_location=agent.device)
        agent.warm_start_from_scalar(warm["online"])
    if bc_dataset is not None and resume is None:  # behavior-clone a guide (e.g. Webster) first
        bc_loss = agent.pretrain_bc(*bc_dataset, epochs=cfg.bc_epochs)
        print(f"[train] BC warm-start from {cfg.bc_warmstart_controller}: "
              f"{len(bc_dataset[0])} transitions, final loss {bc_loss:.4f}", flush=True)
    # Decorrelate the sampling RNG from the exploration RNG (both would otherwise be Random(seed)).
    buffer = ReplayBuffer(cfg.buffer_capacity, n_actions=N_PHASES, seed=cfg.seed + 1_000_000)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
    )

    start_ep, total_steps, best_val_reward = 0, 0, -math.inf
    if resume is not None:
        meta = load_checkpoint(resume, agent)
        start_ep = meta["episode"] + 1
        total_steps = meta["total_steps"]
        best_val_reward = meta["best_val_reward"]

    append = resume is not None
    ep_w, ep_f = _open_csv(
        run_dir / "episodes.csv", append,
        ["episode", "total_steps", "epsilon", "scenario", "ep_reward", "ep_steps",
         "terminated", "truncated", "mean_loss", "mean_grad_norm", "mean_q_mean",
         "mean_q_max", "ss_rolling", "mask_legal_mean"],
    )
    step_w, step_f = _open_csv(
        run_dir / "steps.csv", append,
        ["total_step", "episode", "epsilon", "loss", "grad_norm", "q_mean", "q_max", "ss_rolling"],
    )
    val_w, val_f = _open_csv(
        run_dir / "validation.csv", append,
        ["episode", "total_steps", "val_mean_reward", "val_std_reward"],
    )
    skill = ForecastSkillTracker() if cfg.forecast else None

    try:
        for ep in range(start_ep, cfg.n_episodes):
            scenario_id = cfg.train_scenarios[ep % len(cfg.train_scenarios)]
            route_seed = cfg.seed * _ROUTE_SEED_STRIDE + ep  # per-(run, episode) disjoint traffic
            env = make_train_env(scenario_id, route_seed)
            if skill is not None:
                skill.reset_episode()

            obs, info = env.reset()
            mask = info["mask"]
            done = terminated = truncated = False
            ep_reward, ep_steps = 0.0, 0
            losses: list[float] = []
            grad_norms: list[float] = []
            q_means: list[float] = []
            q_maxes: list[float] = []
            mask_legal_total = 0  # sum of legal-action counts -> mask fire-rate (T-03-08 gate)
            eps = epsilon_at(total_steps, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
            try:
                while not done:
                    assert mask.any(), "all actions masked at s - env masking is broken"  # DoD guard
                    mask_legal_total += int(np.asarray(mask).sum())
                    eps = epsilon_at(total_steps, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
                    action = agent.act(obs, mask, eps)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    next_mask = info["mask"]
                    # done flag for the bootstrap = terminated ONLY (truncation is a time cutoff).
                    buffer.push(obs, action, reward, next_obs, terminated, next_mask)
                    ep_reward += reward
                    ep_steps += 1
                    total_steps += 1

                    ss: float | None = None
                    if skill is not None:
                        # reuse the queue the hybrid wrapper already read this step (no 2nd TraCI
                        # call); fall back to movement_features() only if the attr is absent.
                        queue = getattr(env, "last_queue", None)
                        if queue is None:
                            queue, _count = env.movement_features()
                        ss = skill.update(queue, getattr(env, "last_forecast", None))

                    if total_steps % cfg.target_update_freq == 0:
                        agent.sync_target()

                    if len(buffer) >= cfg.min_replay:
                        diag = agent.learn(buffer.sample(cfg.batch_size), return_diagnostics=True)
                        losses.append(diag["loss"])
                        grad_norms.append(diag["grad_norm"])
                        q_means.append(diag["q_mean"])
                        q_maxes.append(diag["q_max"])
                        if cfg.log_steps:
                            step_w.writerow([
                                total_steps, ep, round(eps, 5), diag["loss"], diag["grad_norm"],
                                diag["q_mean"], diag["q_max"], _ss_cell(ss),
                            ])

                    obs, mask = next_obs, next_mask
                    done = terminated or truncated
            finally:
                env.close()

            ep_w.writerow([
                ep, total_steps, round(eps, 5), scenario_id, ep_reward, ep_steps,
                int(terminated), int(truncated), _mean(losses), _mean(grad_norms),
                _mean(q_means), _mean(q_maxes), _ss_cell(skill.value() if skill else None),
                (mask_legal_total / ep_steps) if ep_steps else "",
            ])
            ep_f.flush()
            # live one-line-per-episode progress so a foreground run is watchable in the console.
            print(f"  ep {ep + 1:3d}/{cfg.n_episodes}  {scenario_id}  "
                  f"reward={ep_reward:12,.1f}  eps={eps:.3f}  buffer={len(buffer)}", flush=True)

            if ep % cfg.checkpoint_every == 0 or ep == cfg.n_episodes - 1:
                save_checkpoint(
                    run_dir / "checkpoints" / f"ep{ep}.pt", agent,
                    episode=ep, total_steps=total_steps,
                    best_val_reward=best_val_reward, epsilon=eps, cfg=cfg,
                )

            if cfg.validation_every and ep > 0 and ep % cfg.validation_every == 0:
                v_mean, v_std = validate(agent, cfg, make_val_env)
                val_w.writerow([ep, total_steps, v_mean, v_std])
                val_f.flush()
                if v_mean > best_val_reward:
                    best_val_reward = v_mean
                    save_checkpoint(
                        run_dir / "checkpoints" / "best.pt", agent,
                        episode=ep, total_steps=total_steps,
                        best_val_reward=best_val_reward, epsilon=eps, cfg=cfg,
                    )
    finally:
        ep_f.close()
        step_f.close()
        val_f.close()

    return RunResult(
        run_dir=run_dir,
        episodes_completed=cfg.n_episodes - start_ep,
        total_steps=total_steps,
        best_val_reward=best_val_reward,
    )


def _open_csv(path: Path, append: bool, header: list[str]):
    """Open a CSV writer; write the header unless appending to an existing file."""
    exists = path.exists()
    f = path.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not (append and exists):
        writer.writerow(header)
    return writer, f
