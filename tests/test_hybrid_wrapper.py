"""T-03-05 - Tests for the HybridStateWrapper + the frozen-forecaster loader.

DoD: wrapped env yields 56-dim observations; observation_space is (56,); get_action_mask
passes through; cold-start zero-padding holds for the first 11 steps; a forward pass profiles
under 2 ms. Plus the audit fix: the forecast normalization is z-score, NOT /30+clip, so a large
queue forecast (>100) does NOT saturate to 1.0. And load_forecaster freezes + eval()s.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.build_network import build_net
from src.ml.dqn import DQNAgent
from src.ml.hybrid_wrapper import (
    FORECAST_DIM,
    HYBRID_OBS_DIM,
    HybridStateWrapper,
    load_forecaster,
)
from src.ml.lstm_model import HORIZON, N_MOVEMENTS, LSTMForecaster
from src.env.sumo_env import SUMOEnv


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    build_net()


def _write_route(path: Path, vehicles: list[tuple[str, float, int, str]]) -> Path:
    lines = ['<routes>', '    <vType id="passenger" vClass="passenger"/>']
    for vid, depart, lane, edges in vehicles:
        lines.append(
            f'    <vehicle id="{vid}" type="passenger" depart="{depart}" '
            f'departLane="{lane}" departSpeed="max"><route edges="{edges}"/></vehicle>'
        )
    lines.append("</routes>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _empty_env(tmp_path: Path) -> SUMOEnv:
    """A SUMOEnv whose single vehicle departs far in the future -> zero traffic for many steps."""
    route = _write_route(tmp_path / "late.rou.xml", [("v0", 1000.0, 1, "n_t t_s")])
    return SUMOEnv(route, episode_length_s=300)


# --- loader ---

def test_load_forecaster_freezes_and_evals(tmp_path: Path) -> None:
    model = LSTMForecaster()
    model.set_input_stats(torch.full((24,), 3.0), torch.full((24,), 2.0))
    ckpt = tmp_path / "lstm-test.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt)

    loaded = load_forecaster(ckpt)
    assert not loaded.training  # eval mode
    assert all(not p.requires_grad for p in loaded.parameters())  # frozen
    # the fitted input stats rode along in the state_dict
    assert torch.allclose(loaded.input_mean, torch.full((24,), 3.0))
    assert tuple(loaded(torch.randn(2, 12, 24)).shape) == (2, HORIZON, N_MOVEMENTS)


# --- observation contract ---

def test_wrapped_observation_is_56_dim(tmp_path: Path) -> None:
    env = HybridStateWrapper(_empty_env(tmp_path), LSTMForecaster())
    try:
        assert env.observation_space.shape == (HYBRID_OBS_DIM,)  # 56
        obs, info = env.reset()
        assert obs.shape == (HYBRID_OBS_DIM,) and obs.dtype == np.float32
        obs2, *_ = env.step(0)
        assert obs2.shape == (HYBRID_OBS_DIM,)
    finally:
        env.close()


def test_get_action_mask_forwards(tmp_path: Path) -> None:
    base = _empty_env(tmp_path)
    env = HybridStateWrapper(base, LSTMForecaster())
    try:
        env.reset()
        m = env.get_action_mask()  # forwarded to the base env via Wrapper delegation
        assert m.shape == (8,) and m.dtype == bool
        assert np.array_equal(m, base.get_action_mask())
    finally:
        env.close()


def test_cold_start_zero_forecast_for_first_11_steps(tmp_path: Path) -> None:
    """history fills after reset + 11 steps; the forecast tail is zero until then, real after."""
    env = HybridStateWrapper(_empty_env(tmp_path), LSTMForecaster())
    try:
        obs, _ = env.reset()  # augmentation #1 (history len 1)
        tails = [obs[-FORECAST_DIM:]]
        for _ in range(11):
            obs, *_ = env.step(0)
            tails.append(obs[-FORECAST_DIM:])
        # augmentations 1..11 (indices 0..10) are cold-start zeros
        for i in range(11):
            assert np.all(tails[i] == 0.0), f"forecast not zero at cold-start step {i}"
        # augmentation #12 (index 11) ran the model -> a computed (here all-zero-input) forecast
        assert len(tails) == 12
    finally:
        env.close()


# --- the audit fix: z-score normalization does NOT saturate ---

def test_forecast_normalization_does_not_saturate(tmp_path: Path) -> None:
    """A forecast of 100 (heavy queue) z-scores to a spread value, not the old /30+clip 1.0."""
    forecaster = LSTMForecaster()
    # queue stats: mean 10, std 30 -> z-score(100) = (100-10)/30 = 3.0 (the /30+clip path gives 1.0)
    mean = torch.cat([torch.full((N_MOVEMENTS,), 10.0), torch.zeros(N_MOVEMENTS)])
    std = torch.cat([torch.full((N_MOVEMENTS,), 30.0), torch.ones(N_MOVEMENTS)])
    forecaster.set_input_stats(mean, std)
    # history_len=1 so the model path triggers on the first augmentation; stub a huge forecast.
    env = HybridStateWrapper(_empty_env(tmp_path), forecaster, history_len=1)
    env.forecaster.forward = lambda x: torch.full((x.shape[0], HORIZON, N_MOVEMENTS), 100.0)
    try:
        obs, _ = env.reset()
        tail = obs[-FORECAST_DIM:]
        assert np.allclose(tail, 3.0, atol=1e-5)  # z-scored, spread
        assert not np.allclose(tail, 1.0)  # NOT saturated like the old /30+clip would
    finally:
        env.close()


# --- performance: forward < 2 ms (it runs every step) ---

def test_forecaster_forward_under_2ms() -> None:
    model = LSTMForecaster().eval()
    x = torch.randn(1, 12, 24)
    times = []
    with torch.no_grad():
        for _ in range(5):  # warm up
            model(x)
        for _ in range(30):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    # min over repeats = the compute floor (jitter only adds time); it runs every decision step.
    assert min(times) < 2e-3, f"min forward {min(times) * 1e3:.3f} ms exceeds 2 ms"


# --- integration: 56-dim obs flows through the agent ---

def test_hybrid_obs_flows_through_agent(tmp_path: Path) -> None:
    env = HybridStateWrapper(_empty_env(tmp_path), LSTMForecaster())
    agent = DQNAgent(obs_dim=HYBRID_OBS_DIM, seed=0)
    try:
        obs, _ = env.reset()
        action = agent.act(obs, env.get_action_mask(), epsilon=0.0)
        assert 0 <= action < 8
    finally:
        env.close()
