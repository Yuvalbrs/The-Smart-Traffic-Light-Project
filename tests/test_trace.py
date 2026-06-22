"""T-01-04 - Tests for the JSONL tracer (writer + sim_frame builder).

DoD: a 60-second episode produces a JSONL file with 60 well-formed lines, each
validating against schema v1.1.0 (with ``movement_id`` per vehicle).
"""

from __future__ import annotations

import json

import pytest
from sumolib import checkBinary

import traci
from scripts.build_network import _NET_FILE, build_net
from scripts.build_routes import write_routes
from src.scenarios.config import load_all
from src.schema.validate import SCHEMA_VERSION, SchemaError, validate_envelope
from src.trace.sim_frame import MovementResolver, build_sim_frame
from src.trace.writer import JsonlWriter

TLS_ID = "C"


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    """Compile the network once (resolver + episode load against it)."""
    build_net()


def _scn(scenario_id: str):
    return next(s for s in load_all() if s.id == scenario_id)


def _valid_envelope() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "sim_frame",
        "sim_time": 1.0,
        "seq": 1,
        "transition": False,
        "episode_id": 0,
        "payload": {
            "vehicles": [
                {
                    "id": "v0", "x": 1.0, "y": 2.0, "angle": 90.0, "speed": 3.0,
                    "lane": "n_t_1", "type": "passenger", "movement_id": "M1",
                }
            ],
            "signal": {
                "phase_index": 0,
                "signal_colors": {"M0": "red", "M1": "green"},
                "sumo_state": "rrrr",
                "phase_remaining_s": 5.0,
            },
        },
    }


# --- writer (pure, no SUMO) ---

def test_writer_writes_one_line_per_frame(tmp_path) -> None:
    out = tmp_path / "trace.jsonl"
    with JsonlWriter(out) as w:
        w.write(_valid_envelope())
        w.write(_valid_envelope())
        assert w.count == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "sim_frame"


def test_writer_rejects_invalid_envelope(tmp_path) -> None:
    bad = _valid_envelope()
    del bad["payload"]["vehicles"][0]["movement_id"]  # the whole point of 1.1.0
    with JsonlWriter(tmp_path / "t.jsonl") as w:
        with pytest.raises(SchemaError):
            w.write(bad)


def test_writer_validate_false_bypasses(tmp_path) -> None:
    bad = _valid_envelope()
    del bad["payload"]["vehicles"][0]["movement_id"]
    with JsonlWriter(tmp_path / "t.jsonl", validate=False) as w:
        w.write(bad)  # must not raise
        assert w.count == 1


def test_writer_write_after_close_raises(tmp_path) -> None:
    w = JsonlWriter(tmp_path / "t.jsonl")
    with pytest.raises(RuntimeError):
        w.write(_valid_envelope())


# --- movement resolver (SUMO net only, no routes) ---

def test_resolver_maps_lanes_to_movements() -> None:
    traci.start([checkBinary("sumo"), "-n", str(_NET_FILE), "--no-step-log", "true"])
    try:
        r = MovementResolver.from_traci(traci, TLS_ID)
    finally:
        traci.close()
    # leftmost lane = left movement, middle = through, rightmost = free through+right
    assert r.for_lane("n_t_2") == "M0"  # N left
    assert r.for_lane("n_t_1") == "M1"  # N through
    assert r.for_lane("n_t_0") == "M2"  # N free (right lane)
    assert r.for_lane("e_t_2") == "M3"  # E left
    assert r.for_lane("w_t_2") == "M9"  # W left
    # off-approach lanes have no movement
    assert r.for_lane(":C_0_0") is None
    assert r.for_lane("t_s_1") is None


def test_resolver_signal_colors_cover_all_movements() -> None:
    traci.start([checkBinary("sumo"), "-n", str(_NET_FILE), "--no-step-log", "true"])
    try:
        r = MovementResolver.from_traci(traci, TLS_ID)
        ryg = traci.trafficlight.getRedYellowGreenState(TLS_ID)
    finally:
        traci.close()
    colors = r.signal_colors(ryg)
    assert set(colors) == {f"M{i}" for i in range(12)}
    assert set(colors.values()) <= {"red", "yellow", "green"}


# --- the DoD: a 60-second episode -> 60 valid JSONL lines ---

def test_60s_episode_writes_60_valid_frames(tmp_path) -> None:
    scn = _scn("SCN-02")  # heavy -> vehicles present quickly
    route_path = write_routes(scn, 0, out_dir=tmp_path)
    out = tmp_path / "episode.jsonl"

    traci.start([
        checkBinary("sumo"), "-n", str(_NET_FILE), "-r", str(route_path),
        "--no-step-log", "true", "--time-to-teleport", "-1",
    ])
    saw_movement = False
    action = 0
    try:
        resolver = MovementResolver.from_traci(traci, TLS_ID)
        with JsonlWriter(out) as w:  # validates every frame on write
            for step in range(60):
                if step % 10 == 0:  # a "decision" every 10 s, like the real loop
                    action = step // 10 % 8
                    traci.trafficlight.setPhase(TLS_ID, action)
                traci.simulationStep()
                frame = build_sim_frame(
                    traci, TLS_ID, seq=step, episode_id=0,
                    phase_index=action, resolver=resolver,
                )
                if any(v["movement_id"] is not None for v in frame["payload"]["vehicles"]):
                    saw_movement = True
                w.write(frame)
    finally:
        traci.close()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 60
    for line in lines:  # re-validate independently of the writer
        validate_envelope(json.loads(line))
    assert saw_movement, "no vehicle was ever assigned a movement_id"
