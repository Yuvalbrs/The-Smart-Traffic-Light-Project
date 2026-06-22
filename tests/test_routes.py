"""Tests for the T-01-08 route generator.

DoD: a ``.rou.xml`` per ``(scenario, seed)``; each loads in SUMO and inserts
vehicles; same seed -> byte-identical routes. The byte-identity test is the
load-bearing one - it is the foundation of the whole reproducibility contract.
"""

from __future__ import annotations

import pytest
from sumolib import checkBinary

import traci
from scripts.build_network import _NET_FILE, IN_EDGE, OUT_EDGE, build_net
from scripts.build_routes import _route_xml, generate_trips, write_routes
from src.scenarios.config import load_all


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    """Compile the network once (routes load against it)."""
    build_net()


def _scn(scenario_id: str):
    return next(s for s in load_all() if s.id == scenario_id)


# --- determinism (the reproducibility contract) ---

def test_same_seed_byte_identical() -> None:
    scn = _scn("SCN-03")  # the time-varying one - hardest case
    a = _route_xml(scn, 0, generate_trips(scn, 0))
    b = _route_xml(scn, 0, generate_trips(scn, 0))
    assert a == b


def test_different_seeds_differ() -> None:
    scn = _scn("SCN-02")
    assert _route_xml(scn, 0, generate_trips(scn, 0)) != _route_xml(scn, 1, generate_trips(scn, 1))


# --- structure ---

def test_trips_sorted_by_depart() -> None:
    scn = _scn("SCN-05")
    departs = [t.depart for t in generate_trips(scn, 7)]
    assert departs == sorted(departs)


def test_routes_and_lanes_well_formed() -> None:
    scn = _scn("SCN-01")
    in_edges = set(IN_EDGE.values())
    out_edges = set(OUT_EDGE.values())
    lane_for = {"left": 2, "through": 1, "right": 0}
    for t in generate_trips(scn, 0):
        a, b = t.route_edges.split()
        assert a in in_edges and b in out_edges
        assert t.depart_lane == lane_for[t.turn]


def test_heavier_scenario_makes_more_vehicles() -> None:
    light = len(generate_trips(_scn("SCN-01"), 0))   # 200 vph/axis
    heavy = len(generate_trips(_scn("SCN-02"), 0))   # 600 vph/axis
    assert heavy > light


# --- the DoD: loads in SUMO and inserts vehicles ---

def test_route_file_inserts_vehicles_in_sumo(tmp_path) -> None:
    scn = _scn("SCN-02")  # heavy -> vehicles appear quickly
    route_path = write_routes(scn, 0, out_dir=tmp_path)
    traci.start([
        checkBinary("sumo"), "-n", str(_NET_FILE),
        "-r", str(route_path), "--no-step-log", "true", "--time-to-teleport", "-1",
    ])
    try:
        for _ in range(300):  # 300 sim-seconds
            traci.simulationStep()
        # at 600 vph across 4 approaches, many vehicles should have entered in 300 s
        assert traci.vehicle.getIDCount() + traci.simulation.getArrivedNumber() > 0
    finally:
        traci.close()
