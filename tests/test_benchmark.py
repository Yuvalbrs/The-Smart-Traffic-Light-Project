"""T-02-09 - Smoke tests for the throughput pilot.

Timing values are machine-dependent, so these assert structure + the projection
arithmetic (which is deterministic), not absolute speeds. A few real steps confirm
the env and gradient benchmarks actually run.
"""

from __future__ import annotations

import pytest

from scripts.benchmark_env import (
    EPISODES,
    SEEDS,
    STEPS_PER_EP,
    VARIANTS,
    benchmark_env,
    benchmark_gradient,
    project_training,
)
from scripts.build_network import build_net


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    build_net()


def test_project_training_arithmetic() -> None:
    p = project_training(env_sps=100.0, grad_sps=1000.0)
    assert p["total_env_steps"] == EPISODES * VARIANTS * SEEDS * STEPS_PER_EP == 1_296_000
    assert p["env_hours"] == pytest.approx(1_296_000 / 100 / 3600)   # 3.6 h
    assert p["grad_hours"] == pytest.approx(1_296_000 / 1000 / 3600)  # 0.36 h
    assert p["total_hours"] == pytest.approx(3.96)
    assert p["exceeds_escalation"] is False


def test_project_training_flags_escalation() -> None:
    assert project_training(env_sps=5.0, grad_sps=1000.0)["exceeds_escalation"] is True


def test_benchmark_env_runs() -> None:
    res = benchmark_env(3)
    assert res["n_steps"] == 3
    assert res["steps_per_sec"] > 0
    assert res["elapsed_s"] > 0


def test_benchmark_gradient_runs() -> None:
    res = benchmark_gradient(3)
    assert res["n_steps"] == 3
    assert res["steps_per_sec"] > 0
    assert res["device"] in ("cpu", "cuda")
