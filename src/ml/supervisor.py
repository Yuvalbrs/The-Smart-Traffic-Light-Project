"""Safety Supervisor - an RL + classical-fallback hybrid controller (project v2 addition).

Targets the one demonstrated weakness of the DQN: it gridlocks ~93% of the time under heavy load
while the classical Webster controller stays at ~20%. The supervisor does NOT modify or retrain
the DQN - it arbitrates, each decision step, between two finished policies:

* the trained **DQN agent** drives under normal load (where it beats the baselines), and
* a robust **fallback** controller (Webster - empirically the most gridlock-robust baseline on
  this intersection) takes over once congestion crosses a threshold, handing back once it clears.

The switch is driven by a saturation indicator - the sum of raw per-movement queues - with
**hysteresis** so it does not chatter at the boundary: enter fallback only after the indicator
stays above ``threshold`` for ``hysteresis`` consecutive steps; leave only after it stays below
``exit_ratio * threshold`` for ``hysteresis`` steps.

Crucially this adds ZERO inputs to the DQN's state, so it is structurally immune to the failure
that sank the forecast-concatenation hybrid (more raw inputs -> worse). The interface mirrors the
baseline controllers (``reset(env)`` + ``select_action(obs, mask)``) so it plugs into the existing
eval harness in the controller slot. ``active_frac`` reports the fraction of steps the fallback
held control (the graceful-degradation measure, reported honestly).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SafetySupervisor:
    """Switches control between a DQN agent and a robust fallback by a queue-saturation rule.

    Parameters
    ----------
    agent : object
        Trained DQN with ``act(obs, mask, epsilon) -> int``; driven greedily (epsilon=0).
    fallback : object
        Robust controller with ``select_action(obs, mask) -> int`` (and optional ``reset(env)``).
    threshold : float
        Saturation indicator (sum of raw per-movement queues) above which fallback engages.
    hysteresis : int, optional
        Consecutive steps the indicator must stay past a bound before the mode flips. Default 5.
    exit_ratio : float, optional
        Fallback releases when the indicator falls below ``exit_ratio * threshold``. Default 0.7.
    """

    def __init__(
        self, agent: Any, fallback: Any, *, threshold: float,
        hysteresis: int = 5, exit_ratio: float = 0.7,
    ) -> None:
        self.agent = agent
        self.fallback = fallback
        self.threshold = float(threshold)
        self.exit_threshold = float(threshold) * float(exit_ratio)
        self.hysteresis = int(hysteresis)
        self._env: Any = None
        self._in_fallback = False
        self._hi = 0
        self._lo = 0
        self.active_steps = 0
        self.total_steps = 0

    def reset(self, env: Any) -> None:
        """Bind the env (for the saturation read) and clear the switching state for a new episode."""
        self._env = env
        if hasattr(self.fallback, "reset"):
            self.fallback.reset(env)
        self._in_fallback = False
        self._hi = self._lo = 0
        self.active_steps = self.total_steps = 0

    def _saturation(self) -> float:
        """Current saturation indicator = sum of raw per-movement queue lengths."""
        queue, _count = self._env.movement_features()
        return float(np.sum(queue))

    def _update_mode(self, sat: float) -> None:
        """Advance the hysteresis state machine for the current saturation reading."""
        if not self._in_fallback:
            self._hi = self._hi + 1 if sat > self.threshold else 0
            if self._hi >= self.hysteresis:
                self._in_fallback, self._hi = True, 0
        else:
            self._lo = self._lo + 1 if sat < self.exit_threshold else 0
            if self._lo >= self.hysteresis:
                self._in_fallback, self._lo = False, 0

    def select_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """Pick the phase from whichever policy currently holds control."""
        self._update_mode(self._saturation())
        self.total_steps += 1
        if self._in_fallback:
            self.active_steps += 1
            return int(self.fallback.select_action(obs, mask))
        return int(self.agent.act(obs, mask, epsilon=0.0))

    @property
    def active_frac(self) -> float:
        """Fraction of this episode's steps the fallback held control (0..1)."""
        return self.active_steps / self.total_steps if self.total_steps else 0.0
