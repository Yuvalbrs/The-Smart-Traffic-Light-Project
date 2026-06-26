"""Tests for the SafetySupervisor hysteresis state machine (pure logic, no SUMO)."""

from __future__ import annotations

import numpy as np

from src.ml.supervisor import SafetySupervisor


class _FakeEnv:
    """Env stub whose saturation (sum of queues) we drive directly."""

    def __init__(self) -> None:
        self.q_sum = 0.0

    def movement_features(self):
        return np.full(12, self.q_sum / 12.0, dtype=np.float32), np.zeros(12, dtype=np.float32)


class _FakeAgent:
    def act(self, obs, mask, epsilon=0.0) -> int:
        return 1  # the "DQN" action


class _FakeFallback:
    def __init__(self) -> None:
        self.reset_called = False

    def reset(self, env) -> None:
        self.reset_called = True

    def select_action(self, obs, mask) -> int:
        return 7  # the "fallback" action


def _make(threshold=60.0, hysteresis=3):
    env = _FakeEnv()
    sup = SafetySupervisor(_FakeAgent(), _FakeFallback(), threshold=threshold,
                           hysteresis=hysteresis, exit_ratio=0.5)
    sup.reset(env)
    return sup, env


def test_uses_agent_below_threshold():
    sup, env = _make()
    env.q_sum = 10.0
    assert all(sup.select_action(None, None) == 1 for _ in range(10))
    assert sup.active_frac == 0.0


def test_switches_to_fallback_only_after_hysteresis():
    sup, env = _make(threshold=60, hysteresis=3)
    env.q_sum = 100.0  # above threshold
    assert sup.select_action(None, None) == 1  # step 1: still agent (hi=1)
    assert sup.select_action(None, None) == 1  # step 2: still agent (hi=2)
    assert sup.select_action(None, None) == 7  # step 3: fallback engages (hi=3)
    assert sup.select_action(None, None) == 7  # stays in fallback


def test_switches_back_only_after_clearing_for_hysteresis():
    sup, env = _make(threshold=60, hysteresis=3)
    env.q_sum = 100.0
    for _ in range(3):
        sup.select_action(None, None)  # now in fallback
    env.q_sum = 20.0  # below exit_threshold (0.5*60=30)
    assert sup.select_action(None, None) == 7  # lo=1, still fallback
    assert sup.select_action(None, None) == 7  # lo=2
    assert sup.select_action(None, None) == 1  # lo=3 -> back to agent


def test_no_chatter_at_boundary():
    # Oscillating around the threshold but never sustained -> never flips.
    sup, env = _make(threshold=60, hysteresis=3)
    for i in range(20):
        env.q_sum = 100.0 if i % 2 == 0 else 10.0
        sup.select_action(None, None)
    assert sup.active_frac == 0.0  # never sustained 3 in a row -> stayed with the agent


def test_active_frac_and_reset():
    sup, env = _make(threshold=60, hysteresis=1)
    env.q_sum = 100.0
    for _ in range(4):
        sup.select_action(None, None)  # immediately in fallback (hysteresis=1)
    assert sup.active_frac == 1.0
    assert sup.fallback.reset_called
    sup.reset(env)  # new episode
    assert sup.total_steps == 0 and sup.active_steps == 0 and sup.active_frac == 0.0
