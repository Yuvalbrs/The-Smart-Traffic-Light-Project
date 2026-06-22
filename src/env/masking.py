"""T-02-02 - Action masking (min-green / max-green) for the Discrete(8) action space.

Safety primitives are enforced by the environment, not learned (safety-masking.md):
the env exposes a boolean mask of valid actions and the agent sets masked actions'
Q-values to -inf before argmax. The locked rule has three regimes on the green
timer:

* ``time_in_phase < min_green`` -> CANNOT switch (only the current phase is valid);
* ``time_in_phase >= max_green`` -> MUST switch (every phase but the current is valid);
* otherwise -> free choice (all valid).

The invariant ``mask.any()`` always holds (safety-masking.md interface contract).

NEMA barrier note: the locked design permits *any* phase change (a barrier crossing
just triggers the all-red clearance handled in T-02-03), so the mask does NOT forbid
crossings. :func:`barrier_crossing_mask` exposes which actions cross the barrier so
downstream code is barrier-aware, per the backlog audit, without over-constraining
beyond the masking SSOT.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.env.intersection import N_PHASES

if TYPE_CHECKING:
    from src.env.intersection import Intersection


def compute_mask(
    current_phase: int, time_in_phase: float, *, min_green: int, max_green: int
) -> np.ndarray:
    """Return the length-8 boolean validity mask for the current phase + timer.

    Parameters
    ----------
    current_phase : int
        The currently active green phase (0-7).
    time_in_phase : float
        Seconds the current green phase has been active (post-transition timer).
    min_green, max_green : int
        Locked green-time bounds (10 / 60 s).

    Returns
    -------
    np.ndarray
        ``shape (8,), dtype=bool`` - ``True`` where the action is valid.
    """
    mask = np.zeros(N_PHASES, dtype=bool)
    if time_in_phase >= max_green:
        mask[:] = True
        mask[current_phase] = False  # max-green reached: must switch away
    elif time_in_phase < min_green:
        mask[current_phase] = True  # min-green not met: can only hold
    else:
        mask[:] = True  # free choice
    return mask


def barrier_crossing_mask(intersection: "Intersection", current_phase: int) -> np.ndarray:
    """Return a length-8 bool array: ``True`` where switching crosses the NEMA barrier.

    Advisory (barrier crossings are legal, with all-red) - lets downstream code know
    which transitions incur the 2 s clearance.
    """
    return np.array(
        [intersection.is_barrier_crossing(current_phase, a) for a in range(N_PHASES)],
        dtype=bool,
    )
