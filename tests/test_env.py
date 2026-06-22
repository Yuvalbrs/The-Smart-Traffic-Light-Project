"""T-02-01 - Tests for SUMOEnv + the Intersection model.

DoD: a mocked agent runs a full episode; the API matches Gym conventions; and the
two B3 guards are tested - (1) reset() flushes the insertion buffer via
traci.load (no cross-episode leak), (2) getMinExpectedNumber()==0 terminates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import traci
from scripts.build_network import build_net
from src.env.intersection import N_MOVEMENTS, Intersection
from src.env.sumo_env import SUMOEnv


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    build_net()


def _write_route(path: Path, vehicles: list[tuple[str, float, int, str]]) -> Path:
    """Write a minimal .rou.xml. Each vehicle = (id, depart, lane, "in out")."""
    lines = ['<routes>', '    <vType id="passenger" vClass="passenger"/>']
    for vid, depart, lane, edges in vehicles:
        lines.append(
            f'    <vehicle id="{vid}" type="passenger" depart="{depart}" '
            f'departLane="{lane}" departSpeed="max"><route edges="{edges}"/></vehicle>'
        )
    lines.append("</routes>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- Intersection model (SUMO net only) ---

def test_intersection_pressure_and_green_state() -> None:
    traci.start(["sumo", "-n", "config/network/intersection.net.xml", "--no-step-log", "true"])
    try:
        ix = Intersection.from_traci(traci, "C")
        p = ix.pressures(traci)
        assert p.shape == (N_MOVEMENTS,)
        assert np.all(p == 0)  # empty net -> zero pressure
        g = ix.green_state(0)  # NS through (M1, M7) + free rights
        assert len(g) == 16 and set(g) <= {"G", "r"} and "G" in g
        # different actions produce different green strings
        assert ix.green_state(0) != ix.green_state(4)
    finally:
        traci.close()


# --- T-02-03: yellow + all-red transitions ---

def test_transition_states_and_barrier_logic() -> None:
    traci.start(["sumo", "-n", "config/network/intersection.net.xml", "--no-step-log", "true"])
    try:
        ix = Intersection.from_traci(traci, "C")
        # barrier: NS group = 0..3, EW group = 4..7
        assert ix.is_barrier_crossing(0, 4) is True
        assert ix.is_barrier_crossing(3, 5) is True
        assert ix.is_barrier_crossing(0, 1) is False
        assert ix.is_barrier_crossing(4, 7) is False

        # yellow on a within-group switch: ending greens show 'y'; SUMO accepts it
        yellow = ix.yellow_state(0, 1)
        assert "y" in yellow and len(yellow) == 16
        traci.trafficlight.setRedYellowGreenState("C", yellow)
        traci.simulationStep()
        assert "y" in traci.trafficlight.getRedYellowGreenState("C")

        # all-red: no yellow, controlled links red, free rights still green
        all_red = ix.all_red_state()
        assert "y" not in all_red and "G" in all_red  # only free rights are G
        traci.trafficlight.setRedYellowGreenState("C", all_red)
        traci.simulationStep()
        live = traci.trafficlight.getRedYellowGreenState("C")
        assert "y" not in live and "G" in live
    finally:
        traci.close()


def test_step_conserves_decision_window_across_switches(tmp_path) -> None:
    """Yellow/all-red are spent INSIDE the 10 s window, never added to it."""
    from scripts.build_routes import write_routes
    from src.scenarios.config import load_all

    scn = next(s for s in load_all() if s.id == "SCN-02")  # heavy -> never empties in 60 s
    route = write_routes(scn, 0, out_dir=tmp_path)
    env = SUMOEnv(route, episode_length_s=600)
    try:
        env.reset()
        for k in range(6):
            action = 4 if k % 2 == 0 else 0  # alternate -> barrier crossing every step
            _, _, terminated, truncated, info = env.step(action)
            assert not (terminated or truncated)
            assert info["sim_time"] == pytest.approx((k + 1) * 10)  # exactly 10 s/step
    finally:
        env.close()


# --- Gym API conformance ---

def test_gym_api_and_observation_contract(tmp_path) -> None:
    route = _write_route(tmp_path / "r.rou.xml", [("v0", 50.0, 1, "n_t t_s")])
    env = SUMOEnv(route, episode_length_s=60)
    try:
        assert env.action_space.n == 8
        assert env.observation_space.shape == (20,)
        obs, info = env.reset()
        assert obs.shape == (20,) and obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        # pressures normalized into [-1, 1]; phase one-hot sums to exactly 1
        assert -1.0 <= obs[:12].min() and obs[:12].max() <= 1.0
        assert obs[12:].sum() == 1.0
        assert set(np.unique(obs[12:])) <= {0.0, 1.0}

        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (20,) and isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert "sim_time" in info
    finally:
        env.close()


def test_reward_isolates_switch_penalty_on_empty_network(tmp_path) -> None:
    """With no vehicles (pressure 0), reward == 0 for a hold and -lambda for a switch."""
    route = _write_route(tmp_path / "r.rou.xml", [("v0", 500.0, 1, "n_t t_s")])
    env = SUMOEnv(route, episode_length_s=60, switch_penalty=0.1)
    try:
        env.reset()  # applies phase 0; last_action = 0
        _, r_hold, *_ = env.step(0)  # same action, empty -> exactly 0.0
        assert r_hold == pytest.approx(0.0)
        _, r_switch, *_ = env.step(1)  # switched, still empty -> exactly -0.1
        assert r_switch == pytest.approx(-0.1)
    finally:
        env.close()


# --- the DoD: a mocked agent runs a full episode ---

def test_mocked_agent_runs_full_episode(tmp_path) -> None:
    from scripts.build_routes import write_routes
    from src.scenarios.config import load_all

    scn = next(s for s in load_all() if s.id == "SCN-02")  # heavy
    route = write_routes(scn, 0, out_dir=tmp_path)
    env = SUMOEnv(route, episode_length_s=120)  # 12 decisions, fast
    rng = np.random.default_rng(0)
    try:
        env.reset()
        done = False
        steps = 0
        info: dict = {}
        while not done and steps < 200:
            _, reward, terminated, truncated, info = env.step(int(rng.integers(8)))
            assert reward <= 0.0  # pressure term <= 0, switch penalty <= 0
            done = terminated or truncated
            steps += 1
        assert done
        ep = info["episode"]
        assert ep["departed_count"] > 0
        assert {"loaded_count", "departed_count", "arrived_count",
                "insertion_backlog_fraction"} <= set(ep)
    finally:
        env.close()


# --- B3 guard #2: natural termination when the network empties ---

def test_b3_natural_termination(tmp_path) -> None:
    # two NS-through vehicles served by action 0; they arrive well before horizon.
    route = _write_route(
        tmp_path / "r.rou.xml",
        [("v0", 1.0, 1, "n_t t_s"), ("v1", 2.0, 1, "s_t t_n")],
    )
    env = SUMOEnv(route, episode_length_s=3600)
    try:
        env.reset()
        terminated = False
        for _ in range(120):  # up to 1200 s
            _, _, terminated, truncated, info = env.step(0)  # serve NS through
            if terminated or truncated:
                break
        assert terminated  # getMinExpectedNumber()==0 fired before the 3600 s horizon
        assert info["sim_time"] < 3600
        assert info["episode"]["arrived_count"] == 2
    finally:
        env.close()


# --- B3 guard #1: reset flushes the insertion buffer (no cross-episode leak) ---

def test_b3_reset_flushes_between_episodes(tmp_path) -> None:
    route = _write_route(
        tmp_path / "r.rou.xml",
        [("v0", 1.0, 1, "n_t t_s"), ("v1", 2.0, 1, "s_t t_n")],
    )
    env = SUMOEnv(route, episode_length_s=60)
    try:
        env.reset()
        for _ in range(6):
            env.step(0)
        assert env._started  # process is up

        # second episode in the SAME process must reuse traci.load
        env.reset()
        assert env._started  # never closed -> load path, not start
        assert traci.simulation.getTime() == 0.0  # clock reset
        assert traci.vehicle.getIDCount() == 0  # NO leaked vehicles from episode 1
        assert traci.simulation.getMinExpectedNumber() > 0  # routes reloaded fresh
    finally:
        env.close()
