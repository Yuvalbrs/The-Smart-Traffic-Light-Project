"""Tests for the v2 anti-gridlock reward-shaping helper (pure, no SUMO)."""

from __future__ import annotations

import pytest

from src.env.sumo_env import gridlock_penalty


def test_off_by_default_mu_zero():
    # mu<=0 is the locked-reward no-op: penalty is exactly 0 regardless of queue.
    assert gridlock_penalty(100.0, 0.0, 20.0) == 0.0
    assert gridlock_penalty(100.0, -0.5, 20.0) == 0.0


def test_zero_at_or_below_threshold():
    assert gridlock_penalty(20.0, 0.1, 20.0) == 0.0
    assert gridlock_penalty(5.0, 0.1, 20.0) == 0.0


def test_linear_above_threshold():
    assert gridlock_penalty(30.0, 0.1, 20.0) == pytest.approx(1.0)   # 0.1 * (30-20)
    assert gridlock_penalty(50.0, 0.5, 20.0) == pytest.approx(15.0)  # 0.5 * (50-20)
