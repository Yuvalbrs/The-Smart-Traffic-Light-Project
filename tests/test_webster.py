"""T-02-04 - Tests for the Webster fixed-time baseline.

DoD: Webster runs on all 5 scenarios; the locked feasibility rule (E4) is
implemented - Y<0.90 normal, 0.90<=Y<1.0 degraded (clamp C_max=120), Y>=1.0
Webster N/A with a named FixedTime-120 fallback (never blank/dropped).
"""

from __future__ import annotations

import numpy as np
import pytest

import traci
from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.baselines.webster import (
    C_MAX,
    MIN_GREEN,
    WebsterController,
    compute_webster_plan,
    webster_plan_for_scenario,
)
from src.env.sumo_env import SUMOEnv
from src.scenarios.config import load_all

_SPLIT = {"left": 0.2, "through": 0.6, "right": 0.2}


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    build_net()


# --- the feasibility rule (E4), pure ---

def test_normal_regime() -> None:
    plan = compute_webster_plan(200, 200, _SPLIT)  # light -> Y small
    assert plan.status == "normal"
    assert plan.flow_ratio_Y < 0.90
    assert tuple(a for a, _g in plan.phases) == (0, 1, 4, 5)  # critical NEMA phases
    assert all(g >= MIN_GREEN for _a, g in plan.phases)  # min-green floor
    assert plan.cycle_s == pytest.approx(sum(g for _a, g in plan.phases) + 16.0)
    assert plan.is_feasible


def test_degraded_regime_clamps_cycle() -> None:
    plan = compute_webster_plan(1050, 1050, _SPLIT)  # 0.90 <= Y < 1.0
    assert plan.status == "degraded"
    assert 0.90 <= plan.flow_ratio_Y < 1.0
    assert plan.cycle_s <= C_MAX + 1e-6
    assert plan.is_feasible


def test_oversaturated_is_na_with_fixedtime_fallback() -> None:
    plan = compute_webster_plan(1200, 1200, _SPLIT)  # Y >= 1.0
    assert plan.status == "na"
    assert plan.flow_ratio_Y >= 1.0
    assert not plan.is_feasible
    assert plan.cycle_s == pytest.approx(C_MAX)  # FixedTime-120
    greens = [g for _a, g in plan.phases]
    assert greens == pytest.approx([greens[0]] * 4)  # equal splits


def test_plan_is_deterministic() -> None:
    assert compute_webster_plan(300, 500, _SPLIT) == compute_webster_plan(300, 500, _SPLIT)


# --- DoD: runs on all 5 scenarios ---

def test_plan_for_all_five_scenarios() -> None:
    for scn in load_all():
        plan = webster_plan_for_scenario(scn)
        assert plan.status in {"normal", "degraded", "na"}
        assert len(plan.phases) == 4
        assert plan.cycle_s > 0


# --- the controller ---

def test_controller_cycles_through_critical_phases() -> None:
    plan = compute_webster_plan(200, 200, _SPLIT)  # all greens floored to 10s
    ctrl = WebsterController(plan, decision_interval_s=10)
    ctrl.reset()
    full = np.ones(8, dtype=bool)
    actions = [ctrl.select_action(np.zeros(20, dtype=np.float32), full) for _ in range(5)]
    assert actions == [0, 1, 4, 5, 0]  # one 10s decision per phase, then wraps


def test_controller_respects_mask() -> None:
    plan = compute_webster_plan(200, 200, _SPLIT)
    ctrl = WebsterController(plan)
    only_three = np.zeros(8, dtype=bool)
    only_three[3] = True
    a = ctrl.select_action(np.zeros(20, dtype=np.float32), only_three)
    assert a == 3  # scheduled action is masked out -> falls back to the valid one


def test_webster_drives_env(tmp_path) -> None:
    scn = next(s for s in load_all() if s.id == "SCN-01")
    route = write_routes(scn, 0, out_dir=tmp_path)
    plan = webster_plan_for_scenario(scn)
    ctrl = WebsterController(plan)
    env = SUMOEnv(route, episode_length_s=120)
    try:
        obs, info = env.reset()
        ctrl.reset(env)
        for _ in range(12):
            action = ctrl.select_action(obs, info["mask"])
            assert info["mask"][action]  # controller never returns a masked action
            obs, reward, terminated, truncated, info = env.step(action)
            assert reward <= 0.0
            if terminated or truncated:
                break
    finally:
        env.close()
