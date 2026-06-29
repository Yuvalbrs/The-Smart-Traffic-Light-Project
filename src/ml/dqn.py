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

import math
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

# --- Distributional (IQN) extension: T-03-09 risk-sensitive DQN -----------------------------
# Diagnosed weakness (sess16): gridlock is a sharply-separated lower tail of the return
# distribution (-17k..-72k vs -1.7k..-11k clean), and the plain DQN optimizes the MEAN, which is
# blind to that tail at the decision points that still matter. IQN learns the whole return
# distribution; CVaR_alpha action-selection then optimizes the worst-alpha% of outcomes (the
# gridlock episodes) instead of the average. CVaR is applied at action-SELECTION only (the
# distribution is trained risk-neutral) - the simpler, valid approach (Dabney et al. 2018 IQN;
# RACER) that lets ONE trained model be swept over alpha at eval.
N_COS = 64  # cosine basis size for the quantile (tau) embedding (Dabney et al. 2018)
N_QUANTILES = 8  # tau samples per state in the quantile-Huber loss (current AND target; N=N')
K_POLICY = 32  # tau samples for CVaR action-selection (a fixed midpoint grid -> deterministic)
HUBER_KAPPA = 1.0  # quantile-Huber threshold


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


class IQNQNetwork(nn.Module):
    """Implicit Quantile Network (Dabney et al. 2018): per-action return quantiles.

    Architecture mirrors the plain :class:`QNetwork` trunk so the two are comparable: a state
    embedding ``psi(s)`` (``obs_dim -> 128 -> 128``, ReLU) is multiplied element-wise by a
    quantile embedding ``phi(tau)`` (a ``N_COS``-term cosine basis -> 128, ReLU), and a small head
    maps the product to ``n_actions`` quantile values ``Z_tau(s, a)``. Sampling many ``tau`` in
    ``[0, 1]`` and reading ``Z_tau`` traces out the whole return distribution per action; the mean
    over ``tau`` recovers the ordinary Q-value, and the mean over ``tau in [0, alpha]`` is the
    CVaR_alpha (the worst-alpha% expected return) the policy optimizes.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        *,
        hidden_size: int = HIDDEN_SIZE,
        n_actions: int = N_PHASES,
        n_cos: int = N_COS,
    ) -> None:
        super().__init__()
        self.n_cos = n_cos
        # cos(pi * i * tau) for i = 1..n_cos; registered so it rides device moves / state_dict.
        self.register_buffer(
            "_cos_i", torch.arange(1, n_cos + 1, dtype=torch.float32) * math.pi
        )
        self.psi = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.phi = nn.Linear(n_cos, hidden_size)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, n_actions))

    def forward(self, obs: Tensor, taus: Tensor) -> Tensor:
        """Map ``obs`` ``(B, obs_dim)`` + quantile fractions ``taus`` ``(B, N)`` to ``(B, N, A)``.

        ``Z[b, i, a]`` is the ``taus[b, i]``-quantile of the return for action ``a`` at state ``b``.
        """
        b, n = taus.shape
        psi = self.psi(obs)  # (B, H)
        cos = torch.cos(taus.reshape(b, n, 1) * self._cos_i.reshape(1, 1, -1))  # (B, N, n_cos)
        phi = F.relu(self.phi(cos))  # (B, N, H)
        x = psi.unsqueeze(1) * phi  # (B, N, H) - the IQN multiplicative interaction
        return self.head(x)  # (B, N, A)


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
    distributional : bool, optional
        When ``True`` the agent is an IQN (T-03-09): the Q-net is an :class:`IQNQNetwork`, ``learn``
        uses the quantile-Huber loss, and ``act`` selects by CVaR_``cvar_alpha``. Default ``False``
        (the locked plain scalar DQN; behaviour byte-unchanged).
    cvar_alpha : float, optional
        Risk level for CVaR action-selection (distributional only): the mean over the worst-alpha%
        of the return distribution. ``1.0`` = risk-neutral (ordinary mean-Q). Lower = more
        risk-averse (optimizes the gridlock tail). Settable at eval to sweep alpha on one model.
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
        distributional: bool = False,
        cvar_alpha: float = 1.0,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.distributional = distributional
        self.cvar_alpha = cvar_alpha
        if distributional:
            net_cls: type[nn.Module] = IQNQNetwork
            net_kw = dict(hidden_size=hidden_size, n_actions=n_actions)
            # deterministic midpoint grid over [0, alpha] for CVaR action-selection (no RNG at
            # act() -> greedy eval stays reproducible for the paired Wilcoxon test).
            self._policy_taus = ((torch.arange(K_POLICY, dtype=torch.float32) + 0.5) / K_POLICY)
        else:
            net_cls, net_kw = QNetwork, dict(hidden_size=hidden_size, n_actions=n_actions)
        self.online = net_cls(obs_dim, **net_kw).to(self.device)
        self.target = net_cls(obs_dim, **net_kw).to(self.device)
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
            scores = self._action_scores(obs_t).squeeze(0)  # (n_actions,) mean-Q or CVaR_alpha
            scores = scores.masked_fill(~mask_t, _MASK_FILL)
            return int(scores.argmax().item())

    def _cvar_scores(self, net: nn.Module, obs_t: Tensor) -> Tensor:
        """CVaR_``cvar_alpha`` per action from ``net``: mean of its return quantiles over [0, alpha].

        ``alpha=1`` recovers the ordinary mean-Q. Uses the fixed deterministic ``_policy_taus``
        midpoint grid so the score is reproducible (no RNG) - shared by ``act`` (online net) and the
        Bellman target's next-action choice (target net), keeping behaviour and bootstrap consistent.
        """
        b = obs_t.shape[0]
        taus = (self._policy_taus * self.cvar_alpha).to(self.device).expand(b, K_POLICY)
        return net(obs_t, taus).mean(dim=1)  # (B, A)

    def _mean_q(self, net: nn.Module, obs_t: Tensor) -> Tensor:
        """Risk-NEUTRAL per-action value from ``net``: scalar Q, or the mean over the full quantile
        grid for the IQN (CVaR at alpha=1). Used for behavior cloning so the clone shapes the whole
        distribution's mean, independent of the current ``cvar_alpha``."""
        if not self.distributional:
            return net(obs_t)
        b = obs_t.shape[0]
        taus = self._policy_taus.to(self.device).expand(b, K_POLICY)  # [0,1) midpoint grid = mean
        return net(obs_t, taus).mean(dim=1)

    def _action_scores(self, obs_t: Tensor) -> Tensor:
        """Per-action selection score for ``act``: scalar Q (plain) or CVaR_alpha (distributional)."""
        return self.online(obs_t) if not self.distributional else self._cvar_scores(self.online, obs_t)

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

    def _distributional_loss(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Quantile-Huber loss for the IQN (Dabney et al. 2018); returns ``(loss, q_taken)``.

        The bootstrap action is chosen by CVaR_``cvar_alpha`` of the TARGET net (masked before the
        argmax exactly as the scalar path) - so training is CONSISTENT with action-selection: at
        ``alpha=1`` this is the risk-neutral mean-greedy backup (and a single model supports any eval
        alpha); at ``alpha<1`` the agent is trained risk-averse end to end (behaviour + bootstrap),
        learning the return distribution along the risk-averse trajectory rather than only reading it
        off a risk-neutral model. ``q_taken`` is the per-sample mean-Q of the taken action (mean over
        its quantiles), a drop-in for the scalar ``q_mean``/``q_max`` diagnostics.
        """
        b = batch.obs.shape[0]
        with torch.no_grad():
            taus_next = torch.rand(b, N_QUANTILES, device=self.device)
            z_next = self.target(batch.next_obs, taus_next)  # (B, N', A) bootstrap distribution
            scores = self._cvar_scores(self.target, batch.next_obs).masked_fill(~batch.next_mask, _MASK_FILL)
            a_star = scores.argmax(dim=1)  # CVaR_alpha-greedy next action (mean-greedy when alpha=1)
            z_next_a = z_next.gather(2, a_star.view(b, 1, 1).expand(b, N_QUANTILES, 1)).squeeze(2)
            target = batch.reward.unsqueeze(1) + self.gamma * (1.0 - batch.done).unsqueeze(1) * z_next_a
        taus = torch.rand(b, N_QUANTILES, device=self.device)
        z = self.online(batch.obs, taus)  # (B, N, A)
        z_a = z.gather(2, batch.action.view(b, 1, 1).expand(b, N_QUANTILES, 1)).squeeze(2)  # (B, N)
        delta = target.unsqueeze(1) - z_a.unsqueeze(2)  # (B, N, N') pairwise TD errors
        huber = torch.where(
            delta.abs() <= HUBER_KAPPA,
            0.5 * delta.pow(2),
            HUBER_KAPPA * (delta.abs() - 0.5 * HUBER_KAPPA),
        )
        rho = (taus.unsqueeze(2) - (delta.detach() < 0).float()).abs() * huber / HUBER_KAPPA
        loss = rho.sum(dim=1).mean(dim=1).mean()  # sum over current-tau, mean over target-tau, mean batch
        return loss, z_a.mean(dim=1)

    def learn(self, batch: Batch, *, return_diagnostics: bool = False) -> float | dict[str, float]:
        """One gradient step on the TD loss (MSE; quantile-Huber when distributional).

        ``Q(s,a)`` from the online net, target from the frozen target net, backward, global-norm
        grad clip, ``optimizer.step()``. Does NOT sync the target (the loop owns that).

        With ``return_diagnostics=True`` returns a dict ``{"loss", "grad_norm", "q_mean",
        "q_max"}`` for the training-loop CSV (T-03-06). ``grad_norm`` is the global gradient
        norm BEFORE clipping - the value ``clip_grad_norm_`` returns, which is the one worth
        logging (the post-clip norm is uninformative, always <= ``grad_clip``); ``q_mean`` /
        ``q_max`` are over the online net's Q for the actions actually taken in the batch (the
        per-quantile mean-Q in distributional mode).
        """
        if self.distributional:
            loss, q = self._distributional_loss(batch)
        else:
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

    def warm_start_from_scalar(self, scalar_online_state: dict[str, Tensor]) -> None:
        """Initialize this IQN's state trunk + head from a trained plain ``QNetwork`` (transfer init).

        The IQN trains slower than the scalar DQN (it must fit a whole distribution), so on the same
        300-episode budget the risk-NEUTRAL base was undertrained (sess16: 100% gridlock on feasible
        SCN-04 at alpha=1). Copying the converged plain trunk (``net.0``/``net.2`` -> ``psi.0``/
        ``psi.2``) and head (``net.4`` -> ``head.1``) starts the distribution near a good mean policy;
        only the quantile embedding ``phi`` is learned from scratch. Requires a distributional agent.
        """
        if not self.distributional:
            raise ValueError("warm_start_from_scalar requires a distributional (IQN) agent")
        sd = scalar_online_state
        with torch.no_grad():
            self.online.psi[0].weight.copy_(sd["net.0.weight"])
            self.online.psi[0].bias.copy_(sd["net.0.bias"])
            self.online.psi[2].weight.copy_(sd["net.2.weight"])
            self.online.psi[2].bias.copy_(sd["net.2.bias"])
            self.online.head[1].weight.copy_(sd["net.4.weight"])
            self.online.head[1].bias.copy_(sd["net.4.bias"])
        self.sync_target()

    def pretrain_bc(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        masks: np.ndarray,
        *,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> float:
        """Behavior-clone a guide controller (e.g. Webster) into the policy before online RL.

        Supervised cross-entropy so the agent's masked per-action value prefers the guide's action -
        the agent then STARTS near a robust controller (≈Webster's gridlock rate) and online RL only
        has to IMPROVE on it, instead of escaping plain-DQN's 80-93% gridlock (sess16: warm-starting
        from a gridlock-prone policy backfired; from Webster it should not). Pure supervised, no env;
        shapes the risk-neutral mean (``_mean_q``) so it is independent of ``cvar_alpha``. Uses a
        temporary optimizer (a different objective than the TD loss) and syncs the target after.
        Returns the final epoch's mean loss.
        """
        s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        m = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=self.device)
        n = s.shape[0]
        opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        last = 0.0
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            losses: list[float] = []
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                scores = self._mean_q(self.online, s[idx]).masked_fill(~m[idx], _MASK_FILL)
                loss = F.cross_entropy(scores, a[idx])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
                opt.step()
                losses.append(float(loss.item()))
            last = float(np.mean(losses)) if losses else 0.0
        self.sync_target()
        return last
