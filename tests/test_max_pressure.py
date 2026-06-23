"""T-02-05 - Tests for the greedy max-pressure baseline.

DoD (the measurable parts): max-pressure runs on all 5 scenarios via SUMOEnv and
never returns a masked action; the controller picks the legal phase with the
greatest total served pressure. The "outperforms Webster where feasible" sanity
needs KPIs (T-02-08) + the eval harness (T-04-05) and is asserted there, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.baselines.max_pressure import MaxPressureController
from src.env.intersection import load_phase_movements
from src.env.sumo_env import SUMOEnv
from src.scenarios.config import load_all


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    build_net()


def _state(pressures: dict[int, float]) -> np.ndarray:
    """A 20-dim obs with the given normalized movement pressures (rest 0)."""
    s = np.zeros(20, dtype=np.float32)
    for idx, val in pressures.items():
        s[idx] = val
    return s


# --- the phase->movement map (movements.yaml SSOT) ---

def test_phase_movement_map_matches_spec() -> None:
    amap = load_phase_movements()
    assert amap == {
        0: (1, 7),    # NS through  -> M1, M7
        1: (0, 6),    # NS left     -> M0, M6
        2: (0, 1),    # N approach  -> M0, M1
        3: (6, 7),    # S approach  -> M6, M7
        4: (4, 10),   # EW through  -> M4, M10
        5: (3, 9),    # EW left     -> M3, M9
        6: (3, 4),    # E approach  -> M3, M4
        7: (9, 10),   # W approach  -> M9, M10
    }


# --- action selection, pure ---

def test_picks_phase_with_max_total_pressure() -> None:
    ctrl = MaxPressureController.from_spec()
    full = np.ones(8, dtype=bool)
    # heavy on M4 and M10 -> phase 4 (EW through) sums 1.8, no other phase exceeds.
    state = _state({4: 0.9, 10: 0.9})
    assert ctrl.select_action(state, full) == 4


def test_respects_mask_and_breaks_ties_low_index() -> None:
    ctrl = MaxPressureController.from_spec()
    state = _state({4: 0.9, 10: 0.9})  # phase 4 best; 6 and 10's phases tie at 0.9
    mask = np.ones(8, dtype=bool)
    mask[4] = False  # forbid the winner
    # phase 6 (M3,M4)=0.9 and phase 7 (M9,M10)=0.9 tie -> lowest index wins.
    assert ctrl.select_action(state, mask) == 6


def test_returns_current_when_only_current_is_legal() -> None:
    ctrl = MaxPressureController.from_spec()
    mask = np.zeros(8, dtype=bool)
    mask[3] = True  # pre-min-green: only the current phase is valid
    assert ctrl.select_action(_state({}), mask) == 3  # picked despite zero pressure


def test_picks_least_negative_when_all_pressures_negative() -> None:
    ctrl = MaxPressureController.from_spec()
    full = np.ones(8, dtype=bool)
    # every served movement negative; phase 0 (M1,M7) is the least-bad.
    state = _state({i: -0.9 for i in range(12)})
    state[1] = state[7] = -0.1
    assert ctrl.select_action(state, full) == 0


def test_stateless_and_deterministic() -> None:
    ctrl = MaxPressureController.from_spec()
    full = np.ones(8, dtype=bool)
    state = _state({3: 0.5, 9: 0.5})
    ctrl.reset()  # no-op
    a1 = ctrl.select_action(state, full)
    a2 = ctrl.select_action(state, full)
    assert a1 == a2 == 5  # EW left (M3, M9)


# --- DoD: runs on every scenario via the real env, never returns a masked action ---

def test_max_pressure_drives_env_on_all_scenarios(tmp_path) -> None:
    for scn in load_all():
        route = write_routes(scn, 0, out_dir=tmp_path)
        ctrl = MaxPressureController.from_spec()
        env = SUMOEnv(route, episode_length_s=120)
        try:
            obs, info = env.reset()
            ctrl.reset(env)
            for _ in range(12):
                action = ctrl.select_action(obs, info["mask"])
                assert info["mask"][action], f"{scn.id}: returned a masked action"
                obs, reward, terminated, truncated, info = env.step(action)
                assert reward <= 0.0
                if terminated or truncated:
                    break
        finally:
            env.close()
