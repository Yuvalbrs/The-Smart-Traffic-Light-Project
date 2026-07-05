"""Tests for the multi-agent (dict-keyed) SUMOEnv API on the 2-TLS arterial.

Covers the Step-2 DoD of network-arterial-plan.md: per-TLS obs/reward/mask dicts
(20-dim obs each), independent per-TLS transition timing, local rewards, global
termination, and determinism. Uses the hand-written smoke route fixture (NOT the
real corridor demand generator).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.env.sumo_env import SUMOEnv

_ROOT = Path(__file__).resolve().parents[1]
_NET = _ROOT / "config" / "network" / "arterial.net.xml"
_BINDING = _ROOT / "config" / "network" / "arterial_link_index_binding.yaml"
_ROUTES = _ROOT / "config" / "routes" / "arterial_smoke.rou.xml"

_TLS = ["C1", "C2"]


def _make_env(**kwargs) -> SUMOEnv:
    return SUMOEnv(
        _ROUTES,
        net_file=_NET,
        binding_file=_BINDING,
        tls_ids=list(_TLS),
        episode_length_s=kwargs.pop("episode_length_s", 60),
        sumo_seed=kwargs.pop("sumo_seed", 42),
        **kwargs,
    )


@pytest.fixture()
def env():
    e = _make_env()
    yield e
    e.close()


def test_reset_returns_per_tls_dicts(env: SUMOEnv) -> None:
    obs, info = env.reset(seed=42)
    assert set(obs) == set(_TLS) and set(info) == set(_TLS)
    for t in _TLS:
        assert obs[t].shape == (20,)
        assert obs[t].dtype == np.float32


def test_step_dict_api_and_local_rewards(env: SUMOEnv) -> None:
    env.reset(seed=42)
    saw_different_rewards = False
    for _ in range(4):
        obs, rewards, terms, truncs, infos = env.step({"C1": 0, "C2": 0})
        for d in (obs, rewards, terms, truncs, infos):
            assert set(d) == set(_TLS)
        for t in _TLS:
            assert obs[t].shape == (20,)
            assert isinstance(rewards[t], float) and rewards[t] <= 0.0
        # termination/truncation are network-global: both agents agree
        assert terms["C1"] == terms["C2"]
        assert truncs["C1"] == truncs["C2"]
        if rewards["C1"] != rewards["C2"]:
            saw_different_rewards = True
    # smoke demand is asymmetric across the two junctions -> local rewards differ
    assert saw_different_rewards


def test_per_tls_transition_timing_is_independent(env: SUMOEnv) -> None:
    env.reset(seed=42)
    # accumulate green time on phase 0 at both junctions (2 windows = 20 s)
    env.step({"C1": 0, "C2": 0})
    env.step({"C1": 0, "C2": 0})
    # switch ONLY C1 (0 -> 4 crosses the NS<->EW barrier: 3 s yellow + 2 s all-red)
    env.step({"C1": 4, "C2": 0})

    mask_c1 = env.get_action_mask("C1")
    mask_c2 = env.get_action_mask("C2")
    # C1 restarted its green timer (10 - 5 = 5 s < min-green): locked to its phase
    assert mask_c1[4] and mask_c1.sum() == 1
    # C2 held for 30 s (>= min-green, < max-green): free choice, untouched by C1
    assert mask_c2.all()


def test_same_seed_same_obs_sequence() -> None:
    def rollout() -> list[np.ndarray]:
        e = _make_env(sumo_seed=7)
        obs, _ = e.reset(seed=7)
        seq = [np.concatenate([obs[t] for t in _TLS])]
        for i in range(3):
            obs, *_ = e.step({"C1": i % 2 * 4, "C2": 0})
            seq.append(np.concatenate([obs[t] for t in _TLS]))
        e.close()
        return seq

    for a, b in zip(rollout(), rollout()):
        np.testing.assert_array_equal(a, b)
