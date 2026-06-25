"""T-03-04 - Uniform experience-replay buffer for the DQN.

A simple fixed-capacity ring buffer (``collections.deque(maxlen=...)``) holding
``(s, a, r, s', done, next_mask)`` transitions, with uniform random sampling
(training-infrastructure.md "Replay buffer"; no prioritized replay, that is an
explicit choice in algorithm-choice.md). The replay buffer is what lets DQN learn
from *decorrelated* past experience instead of the latest, highly-correlated step -
the stabilizer (with the target net) that makes off-policy Q-learning work.

Only the NEXT-state mask is stored: the Bellman target needs the legal actions at
``s'`` (applied before the max, in :meth:`DQNAgent._td_target`), while ``Q(s,a)`` is
gathered by the action actually taken, so the mask at ``s`` is never read by the
update (true for plain and Double DQN alike). Storing it would be 50 MB of dead bytes.

``sample`` returns the exact :class:`~src.ml.dqn.Batch` the agent consumes, so
``agent.learn(buffer.sample(64))`` composes directly. Sampling draws from the
buffer's own seeded RNG, so a fixed seed reproduces the training stream end to end
(project-rules: same seed -> same outputs).

Scope (T-03-04): the buffer only. The min-replay-before-learning gate (1000) and the
sample cadence live in the training loop (T-03-06); this module just exposes ``__len__``.
"""

from __future__ import annotations

import collections
import random

import numpy as np
import torch

from src.ml.dqn import N_PHASES, Batch

CAPACITY = 100_000  # fixed (training-infrastructure.md): ~one full run of 300x360 transitions
MIN_REPLAY = 1_000  # the LOOP waits for this many before learning; exposed here for that gate

# One stored transition. States/masks are kept as compact numpy arrays (float32 / bool);
# scalars stay python numbers. ~500 bytes each -> 100k ~ 50 MB (training-infrastructure.md).
_Transition = tuple[np.ndarray, int, float, np.ndarray, float, np.ndarray]


class ReplayBuffer:
    """Fixed-capacity uniform replay buffer of DQN transitions.

    Parameters
    ----------
    capacity : int, optional
        Max transitions retained; older ones are evicted (ring buffer). Default 100k.
    n_actions : int, optional
        Action-space size, for validating the next-state mask shape. Default 8.
    device : str, optional
        Device the sampled :class:`Batch` tensors land on. Default ``"cpu"``.
    seed : int, optional
        Seeds the sampling RNG so the training stream is reproducible.
    """

    def __init__(
        self,
        capacity: int = CAPACITY,
        *,
        n_actions: int = N_PHASES,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.n_actions = n_actions
        self.device = torch.device(device)
        self._buffer: collections.deque[_Transition] = collections.deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        """Append one transition. ``next_mask`` is the legal-action mask at ``next_state``.

        States are copied to float32 and the mask to bool so the buffer never aliases (or is
        mutated by) arrays the env reuses across steps. The mask must be length ``n_actions``.
        """
        mask_arr = np.asarray(next_mask, dtype=bool)
        if mask_arr.shape != (self.n_actions,):
            raise ValueError(
                f"next_mask must have shape ({self.n_actions},), got {mask_arr.shape}"
            )
        self._buffer.append(
            (
                np.array(state, dtype=np.float32),  # copy
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),  # copy
                float(done),
                mask_arr.copy(),
            )
        )

    def sample(self, batch_size: int = 64) -> Batch:
        """Uniformly sample ``batch_size`` transitions as a stacked :class:`Batch`.

        Raises ``ValueError`` if fewer than ``batch_size`` transitions are stored (the loop
        gates on ``len(buffer) >= MIN_REPLAY`` before ever calling this). Indices are drawn
        from ``range(len)`` (a real sequence) to avoid relying on deque being samplable.
        """
        n = len(self._buffer)
        if batch_size > n:
            raise ValueError(f"cannot sample {batch_size} from a buffer of {n}")
        idx = self._rng.sample(range(n), batch_size)
        rows = [self._buffer[i] for i in idx]
        states, actions, rewards, next_states, dones, masks = zip(*rows)
        return Batch(
            obs=torch.from_numpy(np.stack(states)).to(self.device),
            action=torch.tensor(actions, dtype=torch.long, device=self.device),
            reward=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            next_obs=torch.from_numpy(np.stack(next_states)).to(self.device),
            done=torch.tensor(dones, dtype=torch.float32, device=self.device),
            next_mask=torch.from_numpy(np.stack(masks)).to(self.device),
        )

    def __len__(self) -> int:
        """Number of transitions currently stored (caps at ``capacity``)."""
        return len(self._buffer)
