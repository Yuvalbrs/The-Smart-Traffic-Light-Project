"""T-02-06 - SUMO actuated baseline (the deployed-real-world analog).

Gap-based actuated control runs entirely inside SUMO (baselines-implementation.md
Baseline 3): the controller is just a thin shim that flips :class:`SUMOEnv` into
``signal_mode="actuated"`` and then does nothing - SUMO's actuated program (built
by ``scripts/build_actuated.py``, loaded via the additional-file) owns the lights.

The env still computes pressures, reward, and the gridlock-guard counters
identically to every other controller, so the comparison stays apples-to-apples.
"""

from __future__ import annotations

import numpy as np


class SUMOActuatedController:
    """Hands phase control to SUMO. Matches the ``(state, mask) -> int`` interface.

    ``select_action`` is a no-op: the env ignores the action in actuated mode and
    SUMO's gap-out logic decides every switch. Construct the env with
    ``SUMOEnv(route, signal_mode="actuated")`` for this baseline to take effect.
    """

    def reset(self, env=None) -> None:  # noqa: ANN001 - env mode is set at construction
        """No-op: actuated control is configured on the env, not the controller."""

    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:  # noqa: ARG002
        """Return a dummy action; SUMO drives the lights in actuated mode."""
        return 0
