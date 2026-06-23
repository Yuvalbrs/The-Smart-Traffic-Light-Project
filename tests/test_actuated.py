"""T-02-06 - Tests for the SUMO actuated baseline.

DoD (measurable parts): the actuated program + detectors are generated
deterministically; actuated runs cleanly on all 5 scenarios via SUMOEnv with SUMO
(not Python) driving the lights. The quality comparison vs the other controllers
needs KPIs (T-02-08) and is asserted in the eval harness (T-04-05), not here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import traci
from scripts.build_actuated import CYCLE, build_actuated_add
from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.baselines.actuated import SUMOActuatedController
from src.env.intersection import Intersection
from src.env.sumo_env import SUMOEnv
from src.scenarios.config import load_all


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    build_net()


# --- the generated additional-file ---

def test_build_actuated_add_structure(tmp_path) -> None:
    out = build_actuated_add(out_path=tmp_path / "actuated.add.xml")
    root = ET.parse(out).getroot()

    detectors = root.findall("inductionLoop")
    assert len(detectors) == 8  # one per controlled incoming lane (free rights excluded)

    tl = root.find("tlLogic")
    assert tl.get("type") == "actuated"
    assert any(p.get("key") == "max-gap" and p.get("value") == "3.0" for p in tl.findall("param"))
    # every detector is bound to its lane (param key = lane id)
    lane_params = {p.get("key"): p.get("value") for p in tl.findall("param") if p.get("key") != "max-gap"}
    assert len(lane_params) == 8

    phases = tl.findall("phase")
    greens = [p for p in phases if p.get("minDur") == "10"]
    assert len(greens) == 8  # the 8 NEMA phases, minDur=10 / maxDur=60
    assert all(p.get("maxDur") == "60" for p in greens)
    all_reds = [p for p in phases if p.get("duration") == "2"]
    assert len(all_reds) == 2  # exactly the two barrier crossings (3->4, 7->0)


def test_generated_greens_match_intersection_phases(tmp_path) -> None:
    out = build_actuated_add(out_path=tmp_path / "actuated.add.xml")
    tl = ET.parse(out).getroot().find("tlLogic")
    greens = [p.get("state") for p in tl.findall("phase") if p.get("minDur") == "10"]

    traci.start(["sumo", "-n", "config/network/intersection.net.xml", "--no-step-log", "true"])
    try:
        ix = Intersection.from_traci(traci, "C")
        assert greens == [ix.green_state(a) for a in CYCLE]  # same phase set as the env
    finally:
        traci.close()


# --- the controller shim ---

def test_controller_is_a_noop() -> None:
    ctrl = SUMOActuatedController()
    ctrl.reset()
    assert ctrl.select_action(np.zeros(20, dtype=np.float32), np.ones(8, dtype=bool)) == 0


# --- DoD: SUMO drives the lights, runs on all 5 scenarios ---

def test_actuated_drives_env_on_all_scenarios(tmp_path) -> None:
    build_actuated_add()  # ensure the committed add-file matches the current net
    for scn in load_all():
        route = write_routes(scn, 0, out_dir=tmp_path)
        ctrl = SUMOActuatedController()
        env = SUMOEnv(route, episode_length_s=150, signal_mode="actuated")
        try:
            obs, info = env.reset()
            ctrl.reset(env)
            phases_seen = set()
            done = False
            steps = 0
            while not done and steps < 15:
                action = ctrl.select_action(obs, info["mask"])
                obs, reward, terminated, truncated, info = env.step(action)
                assert reward <= 0.0
                phases_seen.add(info["phase"])
                done = terminated or truncated
                steps += 1
            # SUMO advanced through >=2 green phases on its own -> it is driving.
            assert len(phases_seen) >= 2, f"{scn.id}: actuated program did not switch phases"
            assert info["episode"]["departed_count"] > 0
        finally:
            env.close()


def test_actuated_is_deterministic(tmp_path) -> None:
    build_actuated_add()
    scn = next(s for s in load_all() if s.id == "SCN-01")
    route = write_routes(scn, 0, out_dir=tmp_path)

    def run() -> tuple:
        env = SUMOEnv(route, episode_length_s=120, signal_mode="actuated", sumo_seed=7)
        try:
            env.reset()
            rewards = []
            for _ in range(12):
                _, r, term, trunc, info = env.step(0)
                rewards.append(r)
                if term or trunc:
                    break
            return tuple(rewards), info["episode"]["arrived_count"]
        finally:
            env.close()

    assert run() == run()  # same seed -> identical reward trace + throughput
