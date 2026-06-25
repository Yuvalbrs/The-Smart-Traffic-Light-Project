"""T-03-04 - Tests for the uniform replay buffer.

DoD focus: the buffer fills correctly, evicts past capacity (ring), samples uniformly, and
``sample()`` returns a :class:`Batch` whose ``next_mask`` has the right shape AND bool dtype,
matching the next-state validity recorded at insert time (no transition/mask scrambling).
Plus: seed reproducibility, batch-too-large guard, and an end-to-end smoke that the sampled
Batch drops straight into ``DQNAgent.learn``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.ml.dqn import N_PHASES, OBS_DIM, DQNAgent
from src.ml.replay_buffer import CAPACITY, MIN_REPLAY, ReplayBuffer


def _mask_with_only(action: int, n: int = N_PHASES) -> np.ndarray:
    """A length-n bool mask with exactly ``action`` legal - a fingerprint tying mask to row."""
    m = np.zeros(n, dtype=bool)
    m[action] = True
    return m


def _push_n(buf: ReplayBuffer, n: int, *, obs_dim: int = OBS_DIM) -> None:
    """Push ``n`` transitions whose reward == index and next_mask fingerprints index % 8."""
    for i in range(n):
        buf.push(
            state=np.full(obs_dim, float(i), dtype=np.float32),
            action=i % N_PHASES,
            reward=float(i),
            next_state=np.full(obs_dim, float(i) + 0.5, dtype=np.float32),
            done=False,
            next_mask=_mask_with_only(i % N_PHASES),
        )


# --- fill / capacity ---

def test_len_grows_then_caps_at_capacity() -> None:
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0
    _push_n(buf, 7)
    assert len(buf) == 7
    _push_n(buf, 8)  # 15 pushed into capacity 10 -> evict oldest
    assert len(buf) == 10


def test_ring_buffer_evicts_oldest() -> None:
    buf = ReplayBuffer(capacity=3, seed=0)
    _push_n(buf, 5)  # rewards 0..4 pushed; only the last 3 (2,3,4) survive
    batch = buf.sample(3)
    assert set(int(r) for r in batch.reward.tolist()) == {2, 3, 4}


def test_default_capacity_is_locked_100k() -> None:
    assert CAPACITY == 100_000 and MIN_REPLAY == 1_000
    assert ReplayBuffer().capacity == 100_000


# --- sample shapes / dtypes (DoD) ---

def test_sample_shapes_and_dtypes() -> None:
    buf = ReplayBuffer(capacity=100, seed=1)
    _push_n(buf, 50)
    b = buf.sample(16)
    assert b.obs.shape == (16, OBS_DIM) and b.obs.dtype == torch.float32
    assert b.action.shape == (16,) and b.action.dtype == torch.long
    assert b.reward.shape == (16,) and b.reward.dtype == torch.float32
    assert b.next_obs.shape == (16, OBS_DIM) and b.next_obs.dtype == torch.float32
    assert b.done.shape == (16,) and b.done.dtype == torch.float32
    # the DoD: next_mask is (B, n_actions) and bool
    assert b.next_mask.shape == (16, N_PHASES) and b.next_mask.dtype == torch.bool


def test_next_mask_matches_inserted_transition() -> None:
    """Each row's next_mask must still belong to its own transition - no scrambling."""
    buf = ReplayBuffer(capacity=64, seed=2)
    _push_n(buf, 40)
    b = buf.sample(40)  # draw them all
    for i in range(40):
        reward_i = int(b.reward[i].item())
        legal = torch.nonzero(b.next_mask[i]).flatten()
        assert legal.numel() == 1  # the fingerprint mask had exactly one legal action
        assert int(legal.item()) == reward_i % N_PHASES  # ... and it matches that row's reward


def test_push_rejects_wrong_mask_shape() -> None:
    buf = ReplayBuffer(capacity=10)
    with pytest.raises(ValueError):
        buf.push(
            state=np.zeros(OBS_DIM, dtype=np.float32), action=0, reward=0.0,
            next_state=np.zeros(OBS_DIM, dtype=np.float32), done=False,
            next_mask=np.ones(N_PHASES + 1, dtype=bool),  # wrong length
        )


def test_buffer_does_not_alias_caller_arrays() -> None:
    """Mutating the arrays passed to push must not change what the buffer stored."""
    buf = ReplayBuffer(capacity=4, seed=0)
    state = np.zeros(OBS_DIM, dtype=np.float32)
    mask = _mask_with_only(3)
    buf.push(state=state, action=3, reward=1.0,
             next_state=state, done=False, next_mask=mask)
    state[:] = 99.0  # mutate after push
    mask[:] = True
    b = buf.sample(1)
    assert torch.all(b.obs == 0.0)  # stored copy unaffected
    assert int(torch.nonzero(b.next_mask[0]).item()) == 3  # mask copy unaffected


# --- uniformity + reproducibility ---

def test_sampling_is_uniform() -> None:
    """Over many single draws every transition appears, with roughly even frequency."""
    buf = ReplayBuffer(capacity=10, seed=123)
    _push_n(buf, 10)  # rewards 0..9
    counts = np.zeros(10, dtype=int)
    draws = 5000
    for _ in range(draws):
        counts[int(buf.sample(1).reward.item())] += 1
    assert (counts > 0).all()  # every item drawn at least once
    expected = draws / 10
    assert counts.max() < 1.5 * expected and counts.min() > 0.5 * expected  # roughly even


def test_seed_reproducible_sampling() -> None:
    b1 = ReplayBuffer(capacity=100, seed=7)
    b2 = ReplayBuffer(capacity=100, seed=7)
    _push_n(b1, 50)
    _push_n(b2, 50)
    s1 = b1.sample(16)
    s2 = b2.sample(16)
    assert torch.equal(s1.reward, s2.reward)
    assert torch.equal(s1.next_mask, s2.next_mask)


def test_sample_larger_than_buffer_raises() -> None:
    buf = ReplayBuffer(capacity=100, seed=0)
    _push_n(buf, 5)
    with pytest.raises(ValueError):
        buf.sample(16)


# --- end-to-end: the Batch contract feeds the agent ---

def test_sampled_batch_feeds_agent_learn() -> None:
    buf = ReplayBuffer(capacity=200, seed=0)
    _push_n(buf, 100)
    agent = DQNAgent(obs_dim=OBS_DIM, seed=0)
    loss = agent.learn(buf.sample(32))  # shapes/dtypes must line up with no conversion
    assert isinstance(loss, float) and loss >= 0.0
