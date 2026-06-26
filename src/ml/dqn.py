"""T-03-03 - The plain-DQN agent: Q-network, masked action selection, Bellman update.

The locked algorithm (algorithm-choice.md / ADR-003): plain DQN, a 2-layer MLP
``obs_dim -> 128 -> 128 -> 8`` with ReLU, Adam(lr=1e-4), MSE on the TD error, gradient
clipping, and a hard target-network copy. No dropout, no batch norm - both are known to
hurt DQN stability; the replay buffer + target net are the regularizers.

``obs_dim`` defaults to 20 (the base state) but is a constructor argument so the *same*
network serves the 56-dim hybrid state (the locked forecast ablation, T-03-05) without a
second class.

The one detail that silently corrupts the Q-function if you get it wrong
(algorithm-choice.md "Bellman target with safety mask"): the safety mask at the NEXT
state must be applied to the target-net Q values **before** the ``max``, not at action
selection only -

    q_next[~next_mask] = -1e9          # forbid illegal next actions
    q_next_max = q_next.max(dim=1)     # ... THEN take the max

Masking only at ``act`` time but skipping it here still makes the training loss decrease
on a smoke test - it just learns a biased Q that leaks value through forbidden actions,
and only fails at eval. ``test_dqn.py`` asserts the mask reaches the target before the max.

Scope (T-03-03): the agent only. The replay buffer is T-03-04, the hybrid state wrapper is
T-03-05, and the training loop (epsilon-decay, target-sync cadence, validation) is T-03-06.
This module exposes ``sync_target()`` for the loop to call every ``TARGET_UPDATE_FREQ``
steps; it does not schedule the sync itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

N_PHASES = 8  # locked Discrete(8) NEMA action space (project-rules #3)
OBS_DIM = 20  # locked base state; the hybrid wrapper passes 56 (project-rules #2)
HIDDEN_SIZE = 128  # locked (algorithm-choice.md)

GAMMA = 0.99  # discount; swept {0.9,0.95,0.99}, 0.99 locked (decisions.md / reward-function.md)
LR = 1e-4  # Adam learning rate, locked (algorithm-choice.md "Optimizer")
GRAD_CLIP_NORM = 10.0  # max global grad norm (backlog T-03-03 "grad clip"; not in the ledger)
# hard-copy cadence in agent steps; the training LOOP enforces it (algorithm-choice.md).
TARGET_UPDATE_FREQ = 1000

_MASK_FILL = -1e9  # forbidden-action sentinel (algorithm-choice.md uses -1e9, not -inf -> no NaNs)


class QNetwork(nn.Module):
    """2-layer MLP ``obs_dim -> 128 -> 128 -> n_actions`` with ReLU. No dropout/batch norm."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        *,
        hidden_size: int = HIDDEN_SIZE,
        n_actions: int = N_PHASES,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map a ``(B, obs_dim)`` state batch to ``(B, n_actions)`` Q-values."""
        return self.net(x)


@dataclass
class Batch:
    """One sampled minibatch of transitions, as batched tensors.

    Defined here because the agent is what consumes it; the replay buffer (T-03-04) builds
    and emits this exact type, so the two tasks share one contract without overlapping DoDs.

    Attributes
    ----------
    obs : Tensor
        ``(B, obs_dim)`` float32 states ``s``.
    action : Tensor
        ``(B,)`` int64 actions ``a`` taken at ``s``.
    reward : Tensor
        ``(B,)`` float32 rewards ``r``.
    next_obs : Tensor
        ``(B, obs_dim)`` float32 next states ``s'``.
    done : Tensor
        ``(B,)`` float32 terminal flags (1.0 if ``s'`` is terminal else 0.0); zeros the bootstrap.
    next_mask : Tensor
        ``(B, n_actions)`` bool legal-action mask at ``s'`` - applied to the target before the max.
    """

    obs: Tensor
    action: Tensor
    reward: Tensor
    next_obs: Tensor
    done: Tensor
    next_mask: Tensor


class DQNAgent:
    """Plain-DQN agent: online + frozen target Q-net, masked epsilon-greedy, MSE Bellman update.

    Parameters
    ----------
    obs_dim : int, optional
        State dimensionality. 20 (base) by default; pass 56 for the hybrid state.
    gamma, lr, grad_clip, hidden_size, n_actions :
        Locked hyperparameters; see module constants.
    device : str, optional
        Torch device. Default ``"cpu"`` (the pilot showed the project is SUMO-bound, T-02-09).
    seed : int, optional
        Seeds weight init AND the exploration RNG, so a fixed seed gives reproducible behavior.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        *,
        gamma: float = GAMMA,
        lr: float = LR,
        grad_clip: float = GRAD_CLIP_NORM,
        hidden_size: int = HIDDEN_SIZE,
        n_actions: int = N_PHASES,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.grad_clip = grad_clip
        net_kw = dict(hidden_size=hidden_size, n_actions=n_actions)
        self.online = QNetwork(obs_dim, **net_kw).to(self.device)
        self.target = QNetwork(obs_dim, **net_kw).to(self.device)
        self.sync_target()  # target starts == online
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        # Exploration RNG owned by the agent (independent of torch's global stream) so the
        # epsilon coin + the random legal action are reproducible from `seed` alone.
        self._rng = random.Random(seed)

    def act(
        self, obs: np.ndarray | Tensor, mask: np.ndarray | Tensor, epsilon: float = 0.0
    ) -> int:
        """Pick an action by masked epsilon-greedy: explore among LEGAL actions, else masked-argmax.

        With probability ``epsilon`` returns a uniformly random *legal* action; otherwise the
        greedy ``argmax`` over Q with forbidden actions set to ``-1e9``. Never returns a masked-out
        action. Raises if the mask forbids everything (the env guarantees ``mask.any()``).
        """
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device)
        legal = torch.nonzero(mask_t, as_tuple=False).flatten()
        if legal.numel() == 0:
            raise ValueError("action mask forbids all actions")
        if self._rng.random() < epsilon:
            return int(legal[self._rng.randrange(legal.numel())].item())
        with torch.no_grad():
            obs_a = np.asarray(obs)
            obs_t = torch.as_tensor(obs_a, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(obs_t).squeeze(0)
            q = q.masked_fill(~mask_t, _MASK_FILL)
            return int(q.argmax().item())

    def _td_target(self, batch: Batch) -> Tensor:
        """Bellman target ``r + gamma*(1-done)*max_{a' legal} Q_target(s', a')``.

        The next-state mask is applied to the target-net Q values BEFORE the max (the
        algorithm-choice.md "subtle but important" detail) so the target never leaks value
        through actions that are illegal at ``s'``.
        """
        with torch.no_grad():
            q_next = self.target(batch.next_obs)  # (B, n_actions)
            q_next = q_next.masked_fill(~batch.next_mask, _MASK_FILL)  # mask BEFORE the max
            q_next_max = q_next.max(dim=1).values  # (B,)
            return batch.reward + self.gamma * (1.0 - batch.done) * q_next_max

    def learn(self, batch: Batch, *, return_diagnostics: bool = False) -> float | dict[str, float]:
        """One gradient step on the MSE TD loss. Returns the scalar loss value.

        ``Q(s,a)`` from the online net, target from the frozen target net, MSE, backward,
        global-norm grad clip, ``optimizer.step()``. Does NOT sync the target (the loop owns that).

        With ``return_diagnostics=True`` returns a dict ``{"loss", "grad_norm", "q_mean",
        "q_max"}`` for the training-loop CSV (T-03-06). ``grad_norm`` is the global gradient
        norm BEFORE clipping - the value ``clip_grad_norm_`` returns, which is the one worth
        logging (the post-clip norm is uninformative, always <= ``grad_clip``); ``q_mean`` /
        ``q_max`` are over the online net's Q for the actions actually taken in the batch.
        """
        target = self._td_target(batch)
        q = self.online(batch.obs).gather(1, batch.action.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        loss_val = float(loss.item())
        if not return_diagnostics:
            return loss_val
        return {
            "loss": loss_val,
            "grad_norm": float(grad_norm),
            "q_mean": float(q.mean().item()),
            "q_max": float(q.max().item()),
        }

    def sync_target(self) -> None:
        """Hard update: copy online weights into the target net (call every TARGET_UPDATE_FREQ)."""
        self.target.load_state_dict(self.online.state_dict())
