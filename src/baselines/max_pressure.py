"""T-02-05 - Greedy max-pressure controller (the strong rule-based baseline).

Max-pressure (Varaiya, 2013; baselines-implementation.md Baseline 2). At every
decision step pick the legal NEMA phase whose served movements carry the most
total pressure - i.e. the phase that drains the largest standing imbalance. This
is the canonical greedy controller that motivated the pressure-based reward, so
beating it is the central empirical claim of the project.

Stateless: the choice depends only on the current observation + mask, never on
history. It reuses the env's phase->movement map (``movements.yaml`` SSOT, via
:func:`src.env.intersection.load_phase_movements`) so "which movements a phase
serves" matches the env exactly, and the same action mask the RL agent obeys -
an apples-to-apples comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.env.intersection import (
    N_MOVEMENTS,
    N_PHASES,
    MOVEMENTS_SPEC,
    load_phase_movements,
    unsquash_pressures,
)


class MaxPressureController:
    """Greedy max-pressure baseline. Matches the ``(state, mask) -> int`` interface.

    Parameters
    ----------
    action_movements : mapping
        action (0..7) -> the canonical movement indices (0..11) it greens. Build
        from the spec via :meth:`from_spec`; the explicit constructor exists for
        tests and for reuse of an already-loaded map.
    """

    def __init__(self, action_movements: Mapping[int, Sequence[int]]) -> None:
        self._action_movements: dict[int, tuple[int, ...]] = {
            int(a): tuple(int(m) for m in ms) for a, ms in action_movements.items()
        }

    @classmethod
    def from_spec(cls, movements_path: str | Path = MOVEMENTS_SPEC) -> "MaxPressureController":
        """Build the controller from the ``movements.yaml`` SSOT (no live SUMO)."""
        return cls(load_phase_movements(movements_path))

    def reset(self, env=None) -> None:  # noqa: ANN001 - env unused (stateless)
        """No-op: max-pressure keeps no internal state across steps."""

    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        """Return the legal phase with the greatest total served pressure.

        Pressure per phase = sum of the (un-normalized) movement pressures it
        greens. Masked-invalid actions are excluded (``-inf``); ``mask.any()`` is
        an env invariant, so a finite-scored action always exists. Ties resolve to
        the lowest action index (``np.argmax``), as documented in
        baselines-implementation.md.
        """
        # Invert the observation squash to recover true vehicle-count pressures.
        # This used to be `state[:12] * 10.0`, justified as "argmax is scale-invariant,
        # so this is for readability, not correctness". Scale-invariance held; CLIP-
        # invariance did not. The old observation clipped at +/-10, and 26.9-33.3% of
        # dims saturated under congestion, so this baseline's argmax degraded toward
        # np.argmax's low-index tiebreak exactly when the junction was busiest - i.e.
        # max-pressure was crippled in precisely the regime the comparison is about, and
        # "we beat max-pressure" would have meant beating a hobbled baseline. tanh is
        # monotone but NONLINEAR, so summing squashed values across a phase does not
        # preserve the ordering of summed raw pressures; the inverse is required, not
        # optional. See decisions.md 2026-08-30.
        pressures = unsquash_pressures(state[:N_MOVEMENTS])
        scores = np.full(N_PHASES, -np.inf, dtype=np.float64)
        for action, movements in self._action_movements.items():
            if mask[action]:
                scores[action] = float(pressures[list(movements)].sum())
        return int(np.argmax(scores))
