"""T-03-03 - Tests for the plain-DQN agent.

Two DoD traps drive this file:

1. ``next_mask`` is applied to the target-net Q BEFORE the ``max`` (a mask applied only at
   action selection but skipped in the Bellman target still passes a "loss decreases" smoke
   test - it just learns a biased Q that only fails at eval). ``test_next_mask_before_max``.
2. Reward sanity on the real env: step-0 reward is negative in ``[-70, 0]`` and the switch
   penalty fires exactly when ``action != prev_action``. ``test_reward_sanity_*``.

Plus the ordinary agent contract: shapes, masked epsilon-greedy never picks a forbidden
action, the smoke that loss decreases, ``done`` zeroes the bootstrap, hard target sync, and
seed reproducibility. The pure-torch tests need no SUMO; the reward-sanity tests run a short
live episode and reuse ``test_env``'s net fixture + route helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.build_network import build_net
from src.ml.dqn import N_PHASES, OBS_DIM, Batch, DQNAgent, QNetwork


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    build_net()


def _write_route(path: Path, vehicles: list[tuple[str, float, int, str]]) -> Path:
    """Write a minimal .rou.xml. Each vehicle = (id, depart, lane, "in out")."""
    lines = ['<routes>', '    <vType id="passenger" vClass="passenger"/>']
    for vid, depart, lane, edges in vehicles:
        lines.append(
            f'    <vehicle id="{vid}" type="passenger" depart="{depart}" '
            f'departLane="{lane}" departSpeed="max"><route edges="{edges}"/></vehicle>'
        )
    lines.append("</routes>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- network / shapes ---

@pytest.mark.parametrize("obs_dim", [OBS_DIM, 56])
def test_qnetwork_shape(obs_dim: int) -> None:
    """The same class serves both the 20-dim base and the 56-dim hybrid state -> 8 Q-values."""
    net = QNetwork(obs_dim)
    out = net(torch.randn(4, obs_dim))
    assert tuple(out.shape) == (4, N_PHASES)


# --- DoD trap 1: the next-state mask reaches the target BEFORE the max ---

def test_next_mask_before_max() -> None:
    """The illegal next-action has the highest Q'; the target must use the best LEGAL action."""
    agent = DQNAgent(obs_dim=OBS_DIM, seed=0)
    # action 0 is the global argmax (Q'=10) but illegal at s'; best legal is action 2 (Q'=2).
    fixed_q = torch.tensor([[10.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    agent.target = lambda _x: fixed_q  # type: ignore[assignment]
    next_mask = torch.tensor([[False, False, True, False, False, False, False, False]])
    batch = Batch(
        obs=torch.zeros(1, OBS_DIM),
        action=torch.tensor([0]),
        reward=torch.tensor([0.0]),
        next_obs=torch.zeros(1, OBS_DIM),
        done=torch.tensor([0.0]),
        next_mask=next_mask,
    )
    target = agent._td_target(batch).item()
    assert target == pytest.approx(agent.gamma * 2.0)  # masked -> max over legal {2}
    assert target != pytest.approx(agent.gamma * 10.0)  # NOT the global (illegal) max


def test_done_zeros_bootstrap() -> None:
    """``done=1`` -> target is the reward alone; ``done=0`` -> reward + discounted next-Q."""
    agent = DQNAgent(seed=0)
    agent.target = lambda _x: torch.full((1, N_PHASES), 5.0)  # type: ignore[assignment]
    full_mask = torch.ones(1, N_PHASES, dtype=torch.bool)
    common = dict(
        obs=torch.zeros(1, OBS_DIM), action=torch.tensor([0]),
        reward=torch.tensor([1.0]), next_obs=torch.zeros(1, OBS_DIM), next_mask=full_mask,
    )
    terminal = agent._td_target(Batch(done=torch.tensor([1.0]), **common)).item()
    live = agent._td_target(Batch(done=torch.tensor([0.0]), **common)).item()
    assert terminal == pytest.approx(1.0)  # no bootstrap
    assert live == pytest.approx(1.0 + agent.gamma * 5.0)


# --- masked epsilon-greedy ---

def test_act_never_picks_masked_action() -> None:
    agent = DQNAgent(seed=1)
    # global best is action 0 (illegal here); legal set is {1, 3}, best legal is action 1.
    fixed_q = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]])
    agent.online = lambda _x: fixed_q  # type: ignore[assignment]
    mask = np.array([False, True, False, True, False, False, False, False])
    assert agent.act(np.zeros(OBS_DIM), mask, epsilon=0.0) == 1  # greedy = best legal
    for _ in range(50):  # epsilon=1 always explores, but only among legal actions
        assert mask[agent.act(np.zeros(OBS_DIM), mask, epsilon=1.0)]


def test_act_raises_when_mask_forbids_all() -> None:
    agent = DQNAgent(seed=0)
    with pytest.raises(ValueError):
        agent.act(np.zeros(OBS_DIM), np.zeros(N_PHASES, dtype=bool))


def test_seed_reproducible_action_stream() -> None:
    """Same seed -> identical weight init + exploration stream -> identical actions."""
    mask = np.ones(N_PHASES, dtype=bool)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    a1, a2 = DQNAgent(seed=7), DQNAgent(seed=7)
    seq1 = [a1.act(obs, mask, epsilon=1.0) for _ in range(20)]
    seq2 = [a2.act(obs, mask, epsilon=1.0) for _ in range(20)]
    assert seq1 == seq2


# --- learning ---

def test_loss_decreases_on_fixed_batch() -> None:
    """Fitting a fixed batch against the frozen target net drives the MSE TD loss down."""
    agent = DQNAgent(obs_dim=OBS_DIM, seed=0)
    b = 32
    torch.manual_seed(0)
    batch = Batch(
        obs=torch.randn(b, OBS_DIM),
        action=torch.randint(0, N_PHASES, (b,)),
        reward=torch.randn(b),
        next_obs=torch.randn(b, OBS_DIM),
        done=torch.zeros(b),
        next_mask=torch.ones(b, N_PHASES, dtype=torch.bool),
    )
    first = agent.learn(batch)
    last = first
    for _ in range(200):
        last = agent.learn(batch)
    assert last < first


def test_sync_target_copies_weights() -> None:
    agent = DQNAgent(seed=0)
    with torch.no_grad():  # perturb online so the nets differ
        for p in agent.online.parameters():
            p.add_(1.0)
    online_sd, target_sd = agent.online.state_dict(), agent.target.state_dict()
    assert any(not torch.equal(online_sd[k], target_sd[k]) for k in online_sd)
    agent.sync_target()
    for k in online_sd:
        assert torch.equal(agent.online.state_dict()[k], agent.target.state_dict()[k])


# --- DoD trap 2: reward sanity on the live env ---

def test_reward_sanity_switch_penalty_fires_iff_action_changes(tmp_path) -> None:
    """No traffic at step 0 -> reward is just the switch penalty: 0 on a hold, -0.1 on a switch."""
    from src.env.sumo_env import SUMOEnv

    # one vehicle that departs well after the first decision window -> zero pressure in [0,10).
    route = _write_route(tmp_path / "empty.rou.xml", [("v0", 100.0, 1, "n_t t_s")])

    env_hold = SUMOEnv(route, episode_length_s=120, switch_penalty=0.1)
    try:
        env_hold.reset()
        _, r_hold, _, _, _ = env_hold.step(0)  # hold phase 0 (last_action starts at 0)
    finally:
        env_hold.close()

    env_switch = SUMOEnv(route, episode_length_s=120, switch_penalty=0.1)
    try:
        env_switch.reset()
        _, r_switch, _, _, _ = env_switch.step(1)  # switch 0 -> 1
    finally:
        env_switch.close()

    assert r_hold == pytest.approx(0.0)  # no traffic, no switch -> no penalty
    assert r_switch == pytest.approx(-0.1)  # penalty fires exactly on the change
    assert -70.0 <= r_hold <= 0.0  # step-0 reward sign/bound


def test_reward_sanity_step0_bounded_under_load(tmp_path) -> None:
    """Even on the oversaturated scenario, the step-0 reward is negative and within [-70, 0]."""
    from scripts.build_routes import write_routes
    from src.env.sumo_env import SUMOEnv
    from src.scenarios.config import load_all

    scn = next(s for s in load_all() if s.id == "SCN-02")
    route = write_routes(scn, 0, out_dir=tmp_path)
    env = SUMOEnv(route, episode_length_s=120)
    try:
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert -70.0 <= reward <= 0.0
    finally:
        env.close()
