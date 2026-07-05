"""Tests for coordinated (green-wave) Webster support.

Covers:
- offset_s=0 is byte-identical to the legacy single-junction behaviour (DoD)
- offset_s shifts the phase schedule by exactly offset_s seconds of green time
- coordinated_webster_pair: shared cycle length, correct offsets, min-green floor
- _rescale_plan_to_cycle: proportional rescale and min-green clamping
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.webster import (
    MIN_GREEN,
    WebsterController,
    WebsterPlan,
    _rescale_plan_to_cycle,
    compute_webster_plan,
    coordinated_webster_pair,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SPLIT = {"left": 0.2, "through": 0.6, "right": 0.2}
_LINK_TT = 300.0 / 13.89  # ≈ 21.6 s, arterial free-flow travel time


def _uniform_plan(green_s: float = 10.0) -> WebsterPlan:
    """Construct a 4-phase plan with identical greens (bypasses compute_webster_plan)."""
    phases = ((0, green_s), (1, green_s), (4, green_s), (5, green_s))
    cycle = 4 * 4.0 + 4 * green_s  # L=16s + greens
    return WebsterPlan(phases=phases, cycle_s=cycle, flow_ratio_Y=0.3, status="normal")


def _actions(ctrl: WebsterController, n: int, *, mask_size: int = 8) -> list[int]:
    """Collect *n* actions from *ctrl* (all-valid mask, zero state)."""
    full = np.ones(mask_size, dtype=bool)
    state = np.zeros(20, dtype=np.float32)
    return [ctrl.select_action(state, full) for _ in range(n)]


# ---------------------------------------------------------------------------
# offset_s=0 is byte-identical to legacy behaviour
# ---------------------------------------------------------------------------


def test_offset_zero_identical_to_legacy_sequence() -> None:
    """WebsterController with offset_s=0 must produce the same actions as no offset."""
    plan = compute_webster_plan(200, 200, _SPLIT)

    legacy = WebsterController(plan, decision_interval_s=10)
    with_zero = WebsterController(plan, decision_interval_s=10, offset_s=0.0)

    legacy.reset()
    with_zero.reset()

    assert _actions(legacy, 10) == _actions(with_zero, 10)


def test_offset_zero_state_identical_after_reset() -> None:
    """Internal state after reset must be identical for offset=0 vs. no offset."""
    plan = compute_webster_plan(200, 200, _SPLIT)

    legacy = WebsterController(plan)
    legacy.reset()

    with_zero = WebsterController(plan, offset_s=0.0)
    with_zero.reset()

    assert legacy._idx == with_zero._idx == 0
    assert legacy._elapsed == with_zero._elapsed == 0.0


# ---------------------------------------------------------------------------
# offset_s shifts the phase schedule by exactly offset_s seconds
# ---------------------------------------------------------------------------


def test_offset_state_after_reset() -> None:
    """After reset, _idx and _elapsed correctly reflect the offset position."""
    plan = _uniform_plan(green_s=10.0)  # each phase = 10 s, total_green = 40 s
    ctrl = WebsterController(plan, decision_interval_s=10, offset_s=_LINK_TT)
    ctrl.reset()

    # 21.6 s into 40 s of total green:
    # phase 0 (10 s): 21.6 - 10 = 11.6 remaining
    # phase 1 (10 s): 11.6 - 10 =  1.6 remaining  →  lands in phase 2
    assert ctrl._idx == 2
    assert ctrl._elapsed == pytest.approx(_LINK_TT % 10.0, abs=1e-9)  # 1.6 s


def test_offset_10_shifts_action_sequence_by_one_phase() -> None:
    """offset_s=10 with 10-s greens shifts the sequence by exactly one phase."""
    plan = _uniform_plan(green_s=10.0)

    ctrl_0 = WebsterController(plan, decision_interval_s=10, offset_s=0.0)
    ctrl_10 = WebsterController(plan, decision_interval_s=10, offset_s=10.0)

    ctrl_0.reset()
    ctrl_10.reset()

    seq_0 = _actions(ctrl_0, 8)   # [0, 1, 4, 5, 0, 1, 4, 5]
    seq_10 = _actions(ctrl_10, 8)  # [1, 4, 5, 0, 1, 4, 5, 0]

    # offset=10 sequence is a 1-step shift of offset=0 sequence
    assert seq_10 == seq_0[1:] + seq_0[:1]


def test_offset_21_6_first_action_is_ew_through() -> None:
    """offset_s≈21.6 s (2 full 10-s phases + 1.6 s) starts in EW-through (action 4)."""
    plan = _uniform_plan(green_s=10.0)
    ctrl = WebsterController(plan, decision_interval_s=10, offset_s=_LINK_TT)
    ctrl.reset()

    full = np.ones(8, dtype=bool)
    first = ctrl.select_action(np.zeros(20, dtype=np.float32), full)
    assert first == 4  # action 4 = EW-through (phase index 2)


def test_offset_exact_cycle_boundary_resets_to_start() -> None:
    """An offset equal to the full green cycle wraps back to idx=0, elapsed=0."""
    plan = _uniform_plan(green_s=10.0)
    total_green = sum(g for _, g in plan.phases)  # 40 s

    ctrl = WebsterController(plan, decision_interval_s=10, offset_s=total_green)
    ctrl.reset()

    assert ctrl._idx == 0
    assert ctrl._elapsed == pytest.approx(0.0)


def test_offset_beyond_one_cycle_wraps_correctly() -> None:
    """offset_s > total_green is reduced modulo total_green."""
    plan = _uniform_plan(green_s=10.0)
    total_green = 40.0

    ctrl_base = WebsterController(plan, decision_interval_s=10, offset_s=10.0)
    ctrl_wrap = WebsterController(plan, decision_interval_s=10, offset_s=10.0 + total_green)

    ctrl_base.reset()
    ctrl_wrap.reset()

    assert ctrl_base._idx == ctrl_wrap._idx
    assert ctrl_base._elapsed == pytest.approx(ctrl_wrap._elapsed)


# ---------------------------------------------------------------------------
# _rescale_plan_to_cycle
# ---------------------------------------------------------------------------


def test_rescale_proportional_no_clamp() -> None:
    """Pool-proportional rescaling with no clamping: green POOL scales, cycle exact.

    Lost time L is fixed per cycle, so only the green pool (cycle - L) rescales:
    g_new_i = new_pool * g_old_i / old_pool. Greens do NOT scale by cycle ratio.
    """
    plan = compute_webster_plan(200, 200, _SPLIT)
    new_cycle = plan.cycle_s * 1.5
    scaled = _rescale_plan_to_cycle(plan, new_cycle)

    L = 4.0 * len(plan.phases)
    old_pool = plan.cycle_s - L
    new_pool = new_cycle - L
    orig_greens = [g for _, g in plan.phases]
    new_greens = [g for _, g in scaled.phases]
    for g_orig, g_new in zip(orig_greens, new_greens):
        assert g_new == pytest.approx(new_pool * g_orig / old_pool, rel=1e-6)

    assert scaled.cycle_s == pytest.approx(new_cycle, rel=1e-6)


def test_rescale_preserves_phase_order() -> None:
    """Rescaling does not change the action ordering in phases."""
    plan = compute_webster_plan(300, 500, _SPLIT)
    scaled = _rescale_plan_to_cycle(plan, plan.cycle_s * 1.2)
    assert [a for a, _ in scaled.phases] == [a for a, _ in plan.phases]


def test_rescale_min_green_clamp() -> None:
    """When proportional rescaling drops below MIN_GREEN, clamp is applied."""
    # Build a plan with large greens manually so rescaling down is forced
    big_greens = 40.0
    big_plan = WebsterPlan(
        phases=((0, big_greens), (1, big_greens), (4, big_greens), (5, big_greens)),
        cycle_s=4 * 4.0 + 4 * big_greens,  # 16 + 160 = 176 s
        flow_ratio_Y=0.5,
        status="normal",
    )
    # Target cycle so small that proportional rescaling gives < MIN_GREEN per phase
    # new_pool = 30 - 16 = 14 s; proportional g_new = 14 * 40/160 = 3.5 s < 10 s
    tiny_cycle = 30.0
    scaled = _rescale_plan_to_cycle(big_plan, tiny_cycle)

    for _a, g in scaled.phases:
        assert g >= MIN_GREEN

    # Actual cycle must exceed tiny_cycle due to clamping
    assert scaled.cycle_s > tiny_cycle


def test_rescale_to_same_cycle_is_identity() -> None:
    """Rescaling a plan to its own cycle returns the same greens."""
    plan = compute_webster_plan(400, 300, _SPLIT)
    scaled = _rescale_plan_to_cycle(plan, plan.cycle_s)

    orig_greens = [g for _, g in plan.phases]
    new_greens = [g for _, g in scaled.phases]
    for g_orig, g_new in zip(orig_greens, new_greens):
        assert g_new == pytest.approx(g_orig, rel=1e-9)


# ---------------------------------------------------------------------------
# coordinated_webster_pair
# ---------------------------------------------------------------------------


def test_coordinated_pair_same_cycle_length() -> None:
    """Both controllers from coordinated_webster_pair share the same cycle length."""
    plan_c1 = compute_webster_plan(200, 200, _SPLIT)
    plan_c2 = compute_webster_plan(400, 300, _SPLIT)

    ctrl_c1, ctrl_c2 = coordinated_webster_pair(plan_c1, plan_c2, _LINK_TT)

    assert ctrl_c1.plan.cycle_s == pytest.approx(ctrl_c2.plan.cycle_s, rel=1e-9)


def test_coordinated_pair_cycle_is_max_of_inputs() -> None:
    """Shared cycle equals max(plan_c1.cycle_s, plan_c2.cycle_s)."""
    plan_c1 = compute_webster_plan(200, 200, _SPLIT)
    plan_c2 = compute_webster_plan(400, 300, _SPLIT)

    shared_expected = max(plan_c1.cycle_s, plan_c2.cycle_s)
    ctrl_c1, ctrl_c2 = coordinated_webster_pair(plan_c1, plan_c2, _LINK_TT)

    assert ctrl_c1.plan.cycle_s == pytest.approx(shared_expected, rel=1e-6)


def test_coordinated_pair_upstream_has_no_offset() -> None:
    """Upstream controller (C1) always has offset_s=0."""
    plan_c1 = compute_webster_plan(200, 200, _SPLIT)
    plan_c2 = compute_webster_plan(300, 300, _SPLIT)

    ctrl_c1, _ = coordinated_webster_pair(plan_c1, plan_c2, _LINK_TT)

    assert ctrl_c1._offset_s == 0.0


def test_coordinated_pair_downstream_offset_equals_link_tt() -> None:
    """Downstream controller (C2) has offset_s = link_travel_time_s."""
    plan_c1 = compute_webster_plan(200, 200, _SPLIT)
    plan_c2 = compute_webster_plan(300, 300, _SPLIT)

    _, ctrl_c2 = coordinated_webster_pair(plan_c1, plan_c2, _LINK_TT)

    assert ctrl_c2._offset_s == pytest.approx(_LINK_TT)


def test_coordinated_pair_min_green_preserved() -> None:
    """After coordinated_webster_pair, all greens in both plans meet MIN_GREEN."""
    plan_c1 = compute_webster_plan(200, 200, _SPLIT)
    plan_c2 = compute_webster_pair = compute_webster_plan(600, 400, _SPLIT)

    ctrl_c1, ctrl_c2 = coordinated_webster_pair(plan_c1, plan_c2, _LINK_TT)

    for _a, g in ctrl_c1.plan.phases:
        assert g >= MIN_GREEN
    for _a, g in ctrl_c2.plan.phases:
        assert g >= MIN_GREEN


def test_coordinated_pair_c2_starts_at_ew_through_after_reset() -> None:
    """With equal-green plans and link_tt ≈ 21.6 s, C2 starts in EW-through phase."""
    plan = _uniform_plan(green_s=10.0)
    ctrl_c1, ctrl_c2 = coordinated_webster_pair(plan, plan, _LINK_TT)

    ctrl_c1.reset()
    ctrl_c2.reset()

    full = np.ones(8, dtype=bool)
    state = np.zeros(20, dtype=np.float32)

    first_c1 = ctrl_c1.select_action(state, full)  # offset=0 → phase 0 (NS-through)
    first_c2 = ctrl_c2.select_action(state, full)  # offset≈21.6 → phase 2 (EW-through)

    assert first_c1 == 0  # NS-through
    assert first_c2 == 4  # EW-through


def test_coordinated_pair_same_plan_identity() -> None:
    """Passing identical plans produces controllers whose greens are identical."""
    plan = compute_webster_plan(300, 300, _SPLIT)
    ctrl_c1, ctrl_c2 = coordinated_webster_pair(plan, plan, _LINK_TT)

    assert ctrl_c1.plan.phases == ctrl_c2.plan.phases
    assert ctrl_c1.plan.cycle_s == pytest.approx(ctrl_c2.plan.cycle_s)
