"""Guards for the Phase-0 diagnostic's PRE-REGISTERED verdict logic.

The thresholds in ``scripts.diag_exit_occupancy`` decide a locked project decision (whether the
exit-occupancy masking candidate lives or dies), so the rule that reads them must not drift.
These tests exercise the pure functions only - no SUMO, no checkpoints.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.diag_exit_occupancy import (
    ATTEMPT_AT,
    DEAD_BELOW,
    ONSET_PENDING,
    ONSET_WINDOW,
    _occupancy,
    _rolling_mean,
    episode_summary,
    onset_step,
    verdict,
)


class _FakeEdge:
    def __init__(self, values: dict[str, float]) -> None:
        self._v = values

    def getLastStepOccupancy(self, edge_id: str) -> float:
        return self._v[edge_id]


class _FakeConn:
    def __init__(self, values: dict[str, float]) -> None:
        self.edge = _FakeEdge(values)


def test_occupancy_passes_through_fractions() -> None:
    conn = _FakeConn({"a": 0.25, "b": 0.5})
    assert _occupancy(conn, ("a", "b")) == pytest.approx([0.25, 0.5])


def test_occupancy_normalizes_percentages() -> None:
    """SUMO has reported this field as a percentage in some versions; 65.0 must not mean 65x full."""
    conn = _FakeConn({"a": 65.0, "b": 2.0})
    assert _occupancy(conn, ("a", "b")) == pytest.approx([0.65, 0.02])


def test_occupancy_is_clipped_to_unit_interval() -> None:
    conn = _FakeConn({"a": -0.0, "b": 120.0})
    out = _occupancy(conn, ("a", "b"))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_rolling_mean_is_trailing_and_length_preserving() -> None:
    out = _rolling_mean(np.array([1.0, 3.0, 5.0]), 2)
    assert out == pytest.approx([1.0, 2.0, 4.0])


def test_onset_requires_a_sustained_backlog_not_a_spike() -> None:
    """A single spike above the threshold is not gridlock onset; a sustained one is."""
    spike = np.zeros(30)
    spike[5] = 100 * ONSET_PENDING
    assert onset_step(spike) is None

    sustained = np.zeros(30)
    sustained[10:] = ONSET_PENDING
    assert onset_step(sustained) == 10 + ONSET_WINDOW - 1


def test_onset_ignores_a_backlog_that_drains_before_the_window_closes() -> None:
    """Backlog must persist for the whole window, not merely average high across it."""
    blip = np.zeros(30)
    blip[10 : 10 + ONSET_WINDOW - 1] = ONSET_PENDING * 10  # high, but one step short
    assert onset_step(blip) is None


def test_onset_is_none_when_nothing_ever_backs_up() -> None:
    assert onset_step(np.zeros(50)) is None


def _rows(exit_occ: float, appr_occ: float, pending: list[int]) -> list[dict]:
    return [
        {
            "exit_occ_mean": exit_occ,
            "exit_occ_max": exit_occ,
            "appr_occ_mean": appr_occ,
            "pending": p,
            "sim_time": i * 10.0,
        }
        for i, p in enumerate(pending)
    ]


def _summary(exit_occ: float, gridlocked: bool = True) -> dict:
    pending = [0] * 10 + [ONSET_PENDING * 4] * 30
    s = episode_summary(_rows(exit_occ, 0.6, pending))
    s.update({"train_seed": 42, "eval_seed": 7000, "gridlocked": gridlocked, "backlog_frac": 0.7})
    return s


def test_verdict_dead_when_exits_stay_empty_while_backlog_grows() -> None:
    v, why = verdict([_summary(0.02) for _ in range(5)])
    assert v == "DEAD"
    assert "never reach" in why


def test_verdict_attempt_when_exits_are_full_before_onset() -> None:
    v, why = verdict([_summary(ATTEMPT_AT + 0.05) for _ in range(5)])
    assert v == "ATTEMPT-WARRANTED"
    assert "spillback is real" in why


def test_verdict_inconclusive_between_the_thresholds() -> None:
    mid = (DEAD_BELOW + ATTEMPT_AT) / 2
    v, _ = verdict([_summary(mid) for _ in range(5)])
    assert v == "INCONCLUSIVE"


def test_verdict_refuses_to_conclude_without_gridlock() -> None:
    """A run where nothing gridlocked is not evidence for either branch."""
    v, why = verdict([_summary(0.02, gridlocked=False) for _ in range(3)])
    assert v == "NO-GRIDLOCK"
    assert "cannot be answered" in why


def test_thresholds_match_the_preregistered_values() -> None:
    """finish-plan.md Phase 0 fixed these before the measurement; they are not tunable after."""
    assert (DEAD_BELOW, ATTEMPT_AT) == (0.70, 0.85)
