"""T-02-04 - Webster's classical fixed-time controller (the floor baseline).

Webster (1958): compute a fixed cycle + green splits offline from the design
demand, then run them open-loop (baselines-implementation.md). Non-adaptive on
purpose - it must not re-fit at runtime.

Cycle / splits::

    C   = (1.5*L + 5) / (1 - Y)          # optimal cycle (s)
    g_i = (C - L) * y_i / Y              # effective green for critical phase i
    L   = lost_time_per_phase * n_phases # total lost time per cycle
    y_i = q_i / s_i                      # flow ratio (demand / saturation flow)
    Y   = sum_i y_i                      # critical flow-ratio sum

Critical phases = the 4 non-overlapping NEMA phases that cover every controlled
movement once: phase 0 (NS-through), 1 (NS-left), 4 (EW-through), 5 (EW-left)
(movements.yaml). The 8-phase wording in baselines-implementation.md is
inconsistent with Webster's critical-movement method; the 4 critical phases are
the correct construction. Right turns are free (uncontrolled) and need no green.

Locked feasibility rule (open-items E4 / decisions.md), keyed on Y:

* ``Y < 0.90``        -> normal Webster;
* ``0.90 <= Y < 1.0`` -> degraded: clamp the cycle to ``C_max=120`` s, ``status="degraded"``;
* ``Y >= 1.0``        -> oversaturated: Webster N/A; substitute a named FixedTime-120
  fallback (equal splits), ``status="na"`` - never blank or silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.scenarios.config import Scenario

# Critical phase -> (axis, turn): which demand drives each phase's green.
# (action index, axis attribute on Scenario, turn key in turn_split)
_CRITICAL_PHASES: tuple[tuple[int, str, str], ...] = (
    (0, "ns", "through"),
    (1, "ns", "left"),
    (4, "ew", "through"),
    (5, "ew", "left"),
)

SAT_FLOW_PER_LANE = 1800.0  # veh/h/lane, standard Webster saturation flow
LOST_TIME_PER_PHASE = 4.0  # s (startup + clearance), baselines-implementation.md
C_MAX = 120.0  # s, degraded-cycle clamp (open-items E4)
MIN_GREEN = 10.0  # s, locked min-green floor (safety-masking.md)
_AMBER_PLUS = 5.0  # the "+5" term in Webster's optimal-cycle formula


@dataclass(frozen=True)
class WebsterPlan:
    """A computed fixed-time plan: ordered (action, green_s) + diagnostics."""

    phases: tuple[tuple[int, float], ...]  # (action, green_s) in cycle order
    cycle_s: float
    flow_ratio_Y: float
    status: str  # "normal" | "degraded" | "na"

    @property
    def is_feasible(self) -> bool:
        """True unless Webster was declared N/A (oversaturated)."""
        return self.status != "na"


def compute_webster_plan(
    ns_rate: float,
    ew_rate: float,
    turn_split: dict[str, float],
    *,
    sat_flow_per_lane: float = SAT_FLOW_PER_LANE,
    lost_time_per_phase: float = LOST_TIME_PER_PHASE,
    c_max: float = C_MAX,
    min_green: float = MIN_GREEN,
) -> WebsterPlan:
    """Compute a Webster plan from per-axis design demand (veh/h).

    Parameters
    ----------
    ns_rate, ew_rate : float
        Design (peak) arrival rate per axis, veh/h.
    turn_split : dict
        ``{"left", "through", "right"}`` fractions (must sum to 1).
    """
    rates = {"ns": ns_rate, "ew": ew_rate}
    # flow ratio y_i = demand_i / saturation; one controlled lane per movement.
    y = [rates[axis] * turn_split[turn] / sat_flow_per_lane for _a, axis, turn in _CRITICAL_PHASES]
    actions = [a for a, _axis, _turn in _CRITICAL_PHASES]
    n = len(actions)
    L = lost_time_per_phase * n
    Y = float(sum(y))

    if Y >= 1.0:  # oversaturated -> Webster N/A, named FixedTime-120 fallback
        green = (c_max - L) / n
        phases = tuple((a, float(green)) for a in actions)
        return WebsterPlan(phases=phases, cycle_s=float(c_max), flow_ratio_Y=Y, status="na")

    c0 = (1.5 * L + _AMBER_PLUS) / (1.0 - Y)
    if Y >= 0.90:  # degraded -> clamp the cycle
        cycle = min(c0, c_max)
        status = "degraded"
    else:
        cycle = c0
        status = "normal"

    greens = [max(min_green, (cycle - L) * yi / Y) for yi in y]
    phases = tuple((a, float(g)) for a, g in zip(actions, greens))
    return WebsterPlan(
        phases=phases, cycle_s=float(L + sum(greens)), flow_ratio_Y=Y, status=status
    )


def _peak_rate(axis, duration_s: int, *, step_s: int = 30) -> float:
    """Peak (design) arrival rate over the episode for one axis."""
    return max(axis.rate_at(t) for t in range(0, duration_s + 1, step_s))


def webster_plan_for_scenario(scenario: "Scenario", **kwargs) -> WebsterPlan:
    """Build a Webster plan from a scenario's peak demand + its turn split."""
    if scenario.is_arterial:
        raise ValueError(
            f"{scenario.id}: webster_plan_for_scenario is single-intersection only; "
            "arterial scenarios need one plan per junction (see coordinated_webster_pair)"
        )
    ns = _peak_rate(scenario.ns, scenario.duration_s)
    ew = _peak_rate(scenario.ew, scenario.duration_s)
    return compute_webster_plan(ns, ew, scenario.turn_split, **kwargs)


class WebsterController:
    """Runs a :class:`WebsterPlan` open-loop on ``SUMOEnv`` (time-based phase cycling).

    ``select_action`` matches the baseline interface ``(state, mask) -> int``.
    The plan is fixed; the controller advances to the next scheduled phase once
    its green time has elapsed, quantized to the env's decision interval. The env
    mask is respected defensively (a scheduled action that is momentarily invalid
    is replaced by a valid one - the timers should keep this from happening).

    Parameters
    ----------
    plan : WebsterPlan
        Pre-computed fixed-time plan.
    decision_interval_s : int
        Env step duration (s); controller advances elapsed by this each call.
    offset_s : float
        Phase offset (s).  At :meth:`reset` the controller fast-forwards to the
        position ``offset_s % sum(greens)`` seconds into the cycle, producing a
        green wave when two controllers share the same :class:`WebsterPlan` cycle
        length and carry different offsets.  ``offset_s=0`` (default) is
        byte-identical to the legacy single-junction behaviour.
    """

    def __init__(
        self,
        plan: WebsterPlan,
        *,
        decision_interval_s: int = 10,
        offset_s: float = 0.0,
    ) -> None:
        self.plan = plan
        self._decision_interval_s = decision_interval_s
        self._offset_s = offset_s
        self._idx = 0
        self._elapsed = 0.0

    def reset(self, env=None) -> None:  # noqa: ANN001 - env unused (open-loop)
        """Reset the cycle position, shifted by *offset_s* (0.0 = legacy behaviour).

        When ``offset_s > 0``, the controller starts as if it has already run for
        ``offset_s`` seconds of green time, enabling green-wave coordination between
        two controllers that share the same cycle length.
        """
        self._idx = 0
        self._elapsed = 0.0
        if self._offset_s > 0.0:
            total_green = sum(g for _a, g in self.plan.phases)
            remaining = self._offset_s % total_green
            for i, (_action, green) in enumerate(self.plan.phases):
                if remaining < green:
                    self._idx = i
                    self._elapsed = remaining
                    return
                remaining -= green
            # offset lands on an exact cycle boundary -> stay at idx=0, elapsed=0.0

    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:  # noqa: ARG002
        """Return the scheduled phase for this decision step (mask-respecting)."""
        action, green = self.plan.phases[self._idx]
        if self._elapsed >= green:  # current phase's green elapsed -> advance
            self._idx = (self._idx + 1) % len(self.plan.phases)
            self._elapsed = 0.0
            action = self.plan.phases[self._idx][0]
        self._elapsed += self._decision_interval_s

        if not mask[action]:  # defensive: never return a masked-invalid action
            action = int(np.flatnonzero(mask)[0])
        return int(action)


# ---------------------------------------------------------------------------
# Coordinated (green-wave) helpers
# ---------------------------------------------------------------------------


def _rescale_plan_to_cycle(
    plan: WebsterPlan,
    new_cycle_s: float,
    *,
    min_green: float = MIN_GREEN,
) -> WebsterPlan:
    """Return *plan* with green splits rescaled to *new_cycle_s*.

    Parameters
    ----------
    plan : WebsterPlan
        Source plan whose green pool will be redistributed.
    new_cycle_s : float
        Target total cycle length (s).  Should be >= ``LOST_TIME_PER_PHASE * n``.
    min_green : float
        Safety floor (s).  If proportional rescaling would produce a green below
        this value it is clamped upward; the resulting ``cycle_s`` then exceeds
        *new_cycle_s* by the sum of those positive corrections.

    Notes
    -----
    **Rescale rule**: let ``L = LOST_TIME_PER_PHASE * n``,
    ``old_pool = plan.cycle_s - L``, ``new_pool = new_cycle_s - L``.
    Each green is rescaled proportionally: ``g_new_i = new_pool * g_old_i / old_pool``.
    The min-green floor is applied afterwards; ``cycle_s`` is set to
    ``L + sum(g_new)``, so callers can detect whether clamping occurred by
    comparing ``result.cycle_s`` against *new_cycle_s*.
    """
    n = len(plan.phases)
    L = LOST_TIME_PER_PHASE * n
    old_pool = plan.cycle_s - L
    new_pool = new_cycle_s - L

    new_phases: list[tuple[int, float]] = []
    for action, g_old in plan.phases:
        g_new = (new_pool * g_old / old_pool) if old_pool > 0.0 else (new_pool / n)
        new_phases.append((action, float(max(min_green, g_new))))

    actual_cycle = L + sum(g for _a, g in new_phases)
    return WebsterPlan(
        phases=tuple(new_phases),
        cycle_s=float(actual_cycle),
        flow_ratio_Y=plan.flow_ratio_Y,
        status=plan.status,
    )


def coordinated_webster_pair(
    plan_c1: WebsterPlan,
    plan_c2: WebsterPlan,
    link_travel_time_s: float,
    *,
    decision_interval_s: int = 10,
    min_green: float = MIN_GREEN,
) -> tuple[WebsterController, WebsterController]:
    """Build a coordinated (green-wave) Webster controller pair for two junctions.

    Parameters
    ----------
    plan_c1 : WebsterPlan
        Webster plan for the upstream junction (C1).  Runs with ``offset_s=0``.
    plan_c2 : WebsterPlan
        Webster plan for the downstream junction (C2).  Receives the phase offset.
    link_travel_time_s : float
        Free-flow travel time from C1 to C2 (s), used as the downstream offset.
        Typical arterial value: 300 m / 13.89 m/s ≈ 21.6 s.
    decision_interval_s : int
        Env decision interval forwarded to both :class:`WebsterController` instances.
    min_green : float
        Min-green floor applied during cycle rescaling (default: ``MIN_GREEN``).

    Returns
    -------
    tuple[WebsterController, WebsterController]
        ``(ctrl_c1, ctrl_c2)`` ready to call :meth:`WebsterController.reset` and
        :meth:`WebsterController.select_action`.

    Notes
    -----
    **Shared-cycle rule**: ``shared_cycle = max(plan_c1.cycle_s, plan_c2.cycle_s)``.
    Both plans are rescaled proportionally to *shared_cycle* via
    :func:`_rescale_plan_to_cycle`.  Because the shared cycle is the *maximum*
    of the two originals, greens can only grow or stay the same; the min-green
    floor is therefore not triggered under normal Webster inputs but is enforced
    defensively for caller-supplied plans.

    **Coordination direction**: the offset advances C2 by *link_travel_time_s*
    seconds into its cycle at :meth:`WebsterController.reset`.  This implements the
    W→E green wave (C1 upstream, C2 downstream).  For the reverse direction (E→W)
    swap the plan arguments and pass the appropriate travel time.
    """
    shared_cycle = max(plan_c1.cycle_s, plan_c2.cycle_s)
    scaled_c1 = _rescale_plan_to_cycle(plan_c1, shared_cycle, min_green=min_green)
    scaled_c2 = _rescale_plan_to_cycle(plan_c2, shared_cycle, min_green=min_green)

    ctrl_c1 = WebsterController(
        scaled_c1, decision_interval_s=decision_interval_s, offset_s=0.0
    )
    ctrl_c2 = WebsterController(
        scaled_c2, decision_interval_s=decision_interval_s, offset_s=link_travel_time_s
    )
    return ctrl_c1, ctrl_c2
