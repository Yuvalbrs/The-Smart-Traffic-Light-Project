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
from src.ml.dqn import (
    N_PHASES,
    N_QUANTILES,
    OBS_DIM,
    Batch,
    DQNAgent,
    IQNQNetwork,
    QNetwork,
)


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


# --- T-03-09: distributional IQN + CVaR (risk-sensitive DQN) ---

@pytest.mark.parametrize("obs_dim", [OBS_DIM, 56])
def test_iqn_shape(obs_dim: int) -> None:
    """IQN maps (obs, tau-grid) -> one return quantile per (sample, tau, action)."""
    net = IQNQNetwork(obs_dim)
    taus = torch.rand(4, N_QUANTILES)
    out = net(torch.randn(4, obs_dim), taus)
    assert tuple(out.shape) == (4, N_QUANTILES, N_PHASES)


def test_distributional_loss_decreases_on_fixed_batch() -> None:
    """The quantile-Huber TD loss drives down when fitting a fixed batch (IQN smoke)."""
    agent = DQNAgent(obs_dim=OBS_DIM, seed=0, distributional=True)
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


def test_cvar_selection_differs_from_mean() -> None:
    """Rigged return heads: action 0 has the higher MEAN but a heavy lower tail; action 1 is tight.

    Risk-neutral (alpha=1) must prefer the high-mean action 0; risk-averse CVaR (alpha=0.1) must
    avoid its tail and pick action 1. This is the whole point of the contribution - selecting on the
    worst-case end of the distribution changes the decision.
    """
    agent = DQNAgent(seed=0, distributional=True)

    def rigged(_obs, taus):  # (B, N) -> (B, N, A); only actions 0,1 are in play
        z = torch.full((*taus.shape, N_PHASES), -1e3)
        z[..., 0] = torch.where(taus < 0.1, torch.full_like(taus, -100.0), torch.full_like(taus, 40.0))
        z[..., 1] = 20.0
        return z

    agent.online = rigged  # type: ignore[assignment]
    mask = np.array([True, True] + [False] * (N_PHASES - 2))
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    agent.cvar_alpha = 1.0
    assert agent.act(obs, mask, epsilon=0.0) == 0  # high mean wins when risk-neutral
    agent.cvar_alpha = 0.1
    assert agent.act(obs, mask, epsilon=0.0) == 1  # heavy lower tail loses when risk-averse


def test_distributional_act_respects_mask() -> None:
    """Per-quantile CVaR selection still never returns a forbidden action (greedy or exploring)."""
    agent = DQNAgent(seed=1, distributional=True)
    mask = np.array([False, True, False, True, False, False, False, False])
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    for _ in range(20):
        assert mask[agent.act(obs, mask, epsilon=0.0)]
    for _ in range(50):
        assert mask[agent.act(obs, mask, epsilon=1.0)]


def test_distributional_greedy_is_deterministic() -> None:
    """CVaR action-selection uses a FIXED tau grid (no RNG) -> reproducible greedy eval actions."""
    agent = DQNAgent(seed=3, distributional=True)
    mask = np.ones(N_PHASES, dtype=bool)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    actions = [agent.act(obs, mask, epsilon=0.0) for _ in range(10)]
    assert len(set(actions)) == 1


def test_warm_start_copies_scalar_trunk() -> None:
    """Transfer-init copies the plain QNetwork trunk+head into the IQN and syncs the target."""
    scalar = DQNAgent(obs_dim=OBS_DIM, seed=0)
    with torch.no_grad():  # perturb so the copy is observable, not coincidental
        for p in scalar.online.parameters():
            p.add_(0.5)
    iqn = DQNAgent(obs_dim=OBS_DIM, seed=1, distributional=True)
    iqn.warm_start_from_scalar(scalar.online.state_dict())
    sd = scalar.online.state_dict()
    assert torch.equal(iqn.online.psi[0].weight, sd["net.0.weight"])
    assert torch.equal(iqn.online.psi[2].weight, sd["net.2.weight"])
    assert torch.equal(iqn.online.head[1].weight, sd["net.4.weight"])
    assert torch.equal(iqn.target.psi[0].weight, iqn.online.psi[0].weight)  # target synced


def test_warm_start_requires_distributional() -> None:
    scalar = DQNAgent(obs_dim=OBS_DIM, seed=0)
    with pytest.raises(ValueError):
        DQNAgent(obs_dim=OBS_DIM, seed=1).warm_start_from_scalar(scalar.online.state_dict())


def test_pretrain_bc_clones_guide_actions() -> None:
    """Behavior cloning makes the agent's masked greedy action match the guide's labels."""
    agent = DQNAgent(obs_dim=OBS_DIM, seed=0, distributional=True)
    rng = np.random.default_rng(0)
    n = 256
    states = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
    masks = np.ones((n, N_PHASES), dtype=bool)
    guide = np.where(states[:, 0] > 0, 3, 0).astype(np.int64)  # a learnable state->action rule
    agent.pretrain_bc(states, guide, masks, epochs=40, batch_size=64)
    preds = np.array([agent.act(states[i], masks[i], epsilon=0.0) for i in range(n)])
    assert (preds == guide).mean() > 0.85  # clone reproduces the guide on most states
