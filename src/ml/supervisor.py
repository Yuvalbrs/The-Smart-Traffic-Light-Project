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


class EpisodeLevelSelector:
    """Decide ONCE per episode (after a short probe) whether the DQN or the fallback drives.

    The reactive :class:`SafetySupervisor` failed because mid-episode switching is too late -
    gridlock is a point of no return, so by the time congestion triggers the switch the fallback
    inherits an unrecoverable jam (sess15). This commits UP FRONT instead: run the SAFE fallback
    (Webster) for a short probe window while measuring demand, then commit for the rest of the
    episode to the DQN (light episodes, where it wins on wait) or stay on the fallback (heavy
    episodes, inheriting its gridlock-robustness). Because the DQN never drives before the
    decision, it cannot sow a jam during the probe - the failure mode that sank the supervisor.

    Same controller interface as the baselines (``reset(env)`` + ``select_action(obs, mask)``), so
    it drops into the eval harness's controller slot. ``active_frac`` is 1.0 on episodes routed to
    the fallback and ``probe_steps/total`` on episodes routed to the DQN (the early safe probe).

    Parameters
    ----------
    agent : object
        Trained DQN with ``act(obs, mask, epsilon) -> int`` (driven greedily).
    fallback : object
        Robust controller with ``select_action(obs, mask) -> int`` (and optional ``reset(env)``).
    threshold : float
        Cumulative vehicles INSERTED over the probe above which the episode is judged heavy and
        stays on the fallback. Insertions are a controller-independent demand signal (queues/pressure
        under the safe probe controller instead reflect how well it coped - and barely separate
        feasible from heavy early; insertions separate cleanly: light <=146 vs heavy >=153 over a
        300 s probe on this net). Default operating point ~150.
    probe_steps : int, optional
        Decision steps to run the fallback while measuring demand before committing. Default 30
        (=300 s at the 10 s decision interval = the KPI warm-up boundary; gives a clean demand gap).
    """

    def __init__(self, agent: Any, fallback: Any, *, threshold: float, probe_steps: int = 30) -> None:
        self.agent = agent
        self.fallback = fallback
        self.threshold = float(threshold)
        self.probe_steps = int(probe_steps)
        self._env: Any = None
        self._step = 0
        self._use_fallback = True  # run the safe fallback during the probe
        self._decided = False
        self.active_steps = 0
        self.total_steps = 0

    def reset(self, env: Any) -> None:
        """Bind the env (for the demand read) and clear the per-episode decision state."""
        self._env = env
        if hasattr(self.fallback, "reset"):
            self.fallback.reset(env)
        self._step = 0
        self._use_fallback = True
        self._decided = False
        self.active_steps = self.total_steps = 0

    def select_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """Run the fallback through the probe, commit once on demand, then run the committed policy."""
        if not self._decided:
            self._step += 1
            if self._step >= self.probe_steps:
                self._use_fallback = self._env.departed_count > self.threshold
                self._decided = True
        self.total_steps += 1
        if self._use_fallback:
            self.active_steps += 1
            return int(self.fallback.select_action(obs, mask))
        return int(self.agent.act(obs, mask, epsilon=0.0))

    @property
    def active_frac(self) -> float:
        """Fraction of this episode's steps the fallback held control (0..1)."""
        return self.active_steps / self.total_steps if self.total_steps else 0.0
