"""T-03-06 - Tests for the DQN training loop, against its known failure modes.

No SUMO here: a :class:`DummyEnv` stands in for ``SUMOEnv`` so the loop's control flow,
diagnostics, resume, and guards are tested fast and deterministically. The real env wiring
(scenario rotation by route file, the hybrid wrapper) lives in ``scripts/train_dqn.py`` and
is exercised by the smoke run there; here we verify the loop logic the script depends on.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.ml.train_loop import (
    ForecastSkillTracker,
    TrainConfig,
    epsilon_at,
    train,
)


class DummyEnv:
    """Minimal gymnasium-shaped env: fixed-length episodes, full (or custom) mask."""

    def __init__(self, obs_dim: int, *, n_steps: int = 6, terminate: bool = True, mask=None):
        self._obs_dim = obs_dim
        self._n_steps = n_steps
        self._terminate = terminate
        self._mask = np.ones(8, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        self.last_forecast = None
        self._t = 0

    def _obs(self):
        return np.zeros(self._obs_dim, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {"mask": self._mask.copy()}

    def step(self, action):
        self._t += 1
        last = self._t >= self._n_steps
        terminated = last and self._terminate
        truncated = last and not self._terminate
        return self._obs(), -1.0, terminated, truncated, {"mask": self._mask.copy()}

    def movement_features(self):
        return np.zeros(12, dtype=np.float32), np.zeros(12, dtype=np.float32)

    def close(self):
        pass


def _smoke_cfg(**overrides) -> TrainConfig:
    base = dict(
        seed=0, n_episodes=4, episode_length_s=200, decision_interval_s=10,
        min_replay=2, batch_size=2, validation_every=2, validation_episodes=2,
        checkpoint_every=2, log_steps=True,
    )
    base.update(overrides)
    return TrainConfig(**base)


def _factories(obs_dim: int):
    def make_train_env(scenario_id: str, route_seed: int):
        return DummyEnv(obs_dim, n_steps=6)

    def make_val_env(route_seed: int):
        return DummyEnv(obs_dim, n_steps=4)

    return make_train_env, make_val_env


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


# --- epsilon schedule ---

def test_epsilon_linear_decay_and_clamp():
    assert epsilon_at(0, 1.0, 0.05, 100) == 1.0
    assert epsilon_at(50, 1.0, 0.05, 100) == pytest.approx(0.525)
    assert epsilon_at(100, 1.0, 0.05, 100) == pytest.approx(0.05)
    assert epsilon_at(500, 1.0, 0.05, 100) == pytest.approx(0.05)  # clamps at the floor
    assert epsilon_at(5, 1.0, 0.05, 0) == 0.05  # degenerate decay window -> floor


def test_eps_decay_steps_default_is_half_of_training():
    cfg = TrainConfig(n_episodes=300, episode_length_s=3600, decision_interval_s=10)
    assert cfg.eps_decay_steps == 54_000  # 0.5 * 300 * 360


# --- forecast skill tracker ---

def test_skill_tracker_perfect_forecast_scores_one():
    t = ForecastSkillTracker(offsets=(1,))
    q0 = np.full(12, 5.0)
    q1 = np.full(12, 8.0)
    assert t.update(q0, np.stack([q1])) is None  # nothing resolved yet
    ss = t.update(q1, None)  # resolves: forecast err 0, persistence err 9 -> SS = 1
    assert ss == pytest.approx(1.0)


def test_skill_tracker_persistence_equal_forecast_scores_zero():
    t = ForecastSkillTracker(offsets=(1,))
    q0 = np.full(12, 5.0)
    q1 = np.full(12, 8.0)
    t.update(q0, np.stack([q0]))  # forecast == persistence
    assert t.update(q1, None) == pytest.approx(0.0)


def test_skill_tracker_resets_pending_across_episodes():
    t = ForecastSkillTracker(offsets=(3,))
    t.update(np.zeros(12), np.zeros((1, 12)))  # enqueues a far-horizon prediction
    t.reset_episode()
    # after reset the pending prediction is gone, so a same-step resolve never fires
    assert t.value() is None


# --- end-to-end smoke run ---

def test_smoke_run_writes_outputs_and_checkpoints(tmp_path):
    cfg = _smoke_cfg()
    make_train_env, make_val_env = _factories(cfg.obs_dim)
    result = train(cfg, make_train_env=make_train_env, make_val_env=make_val_env, run_dir=tmp_path)

    assert result.episodes_completed == 4
    assert (tmp_path / "config.yaml").exists()

    ep_rows = _rows(tmp_path / "episodes.csv")
    assert ep_rows[0][:2] == ["episode", "total_steps"]  # header
    assert len(ep_rows) == 1 + 4  # header + one row per episode

    # diagnostics: learning kicked in (min_replay=2) so step rows carry numeric grad_norm/q stats
    step_rows = _rows(tmp_path / "steps.csv")
    assert step_rows[0] == ["total_step", "episode", "epsilon", "loss", "grad_norm",
                            "q_mean", "q_max", "ss_rolling"]
    assert len(step_rows) > 1
    float(step_rows[1][4])  # grad_norm parses as a float

    # checkpoints: every 2 episodes (0, 2) + the last (3); validation at ep 2 wrote best.pt
    ck = tmp_path / "checkpoints"
    assert (ck / "ep0.pt").exists() and (ck / "ep2.pt").exists() and (ck / "ep3.pt").exists()
    assert (ck / "best.pt").exists()
    val_rows = _rows(tmp_path / "validation.csv")
    assert len(val_rows) == 1 + 1  # header + the single ep-2 validation


def test_resume_continues_from_checkpoint(tmp_path):
    make_train_env, make_val_env = _factories(20)

    cfg1 = _smoke_cfg(n_episodes=2, validation_every=0, checkpoint_every=1, log_steps=False)
    train(cfg1, make_train_env=make_train_env, make_val_env=make_val_env, run_dir=tmp_path)
    assert (tmp_path / "checkpoints" / "ep1.pt").exists()

    cfg2 = _smoke_cfg(n_episodes=4, validation_every=0, checkpoint_every=1, log_steps=False)
    result = train(
        cfg2, make_train_env=make_train_env, make_val_env=make_val_env,
        run_dir=tmp_path, resume=tmp_path / "checkpoints" / "ep1.pt",
    )

    assert result.episodes_completed == 2  # only episodes 2 and 3 ran
    ep_rows = _rows(tmp_path / "episodes.csv")
    assert len(ep_rows) == 1 + 4  # header + 2 (first run) + 2 (resume, appended)
    assert [r[0] for r in ep_rows[1:]] == ["0", "1", "2", "3"]  # episode index continued, not reset


def test_mask_guard_fires_when_all_actions_masked(tmp_path):
    cfg = _smoke_cfg(n_episodes=1, validation_every=0)

    def make_train_env(scenario_id, route_seed):
        return DummyEnv(cfg.obs_dim, mask=np.zeros(8, dtype=bool))  # illegal: nothing legal

    def make_val_env(route_seed):
        return DummyEnv(cfg.obs_dim)

    with pytest.raises(AssertionError, match="masked"):
        train(cfg, make_train_env=make_train_env, make_val_env=make_val_env, run_dir=tmp_path)
