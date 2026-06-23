"""T-01-10 enabler - SUMOEnv optional JSONL tracing.

Verifies the tracer wired into the env writes one valid sim_frame per simulated
second and is byte-deterministic (the basis of the repro smoke test).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.build_network import build_net
from scripts.build_routes import write_routes
from src.env.sumo_env import SUMOEnv
from src.scenarios.config import load_all


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    build_net()


def _run(route, path, *, length_s=30, seed=0) -> None:
    env = SUMOEnv(route, episode_length_s=length_s, sumo_seed=seed, trace_path=path)
    try:
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(0)  # hold phase 0
            done = terminated or truncated
    finally:
        env.close()


def test_trace_one_valid_frame_per_second(tmp_path) -> None:
    scn = next(s for s in load_all() if s.id == "SCN-02")  # heavy -> runs the full 30 s
    route = write_routes(scn, 0)
    path = tmp_path / "trace.jsonl"
    _run(route, path, length_s=30)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30  # one frame per simulated second
    frames = [json.loads(ln) for ln in lines]
    assert all(f["type"] == "sim_frame" for f in frames)
    assert [f["sim_time"] for f in frames] == list(range(1, 31))  # 1..30 s
    # vehicles on an approach carry their movement (schema v1.1.0)
    assert any(v.get("movement_id") for f in frames for v in f["payload"]["vehicles"])


def test_trace_is_byte_deterministic(tmp_path) -> None:
    scn = next(s for s in load_all() if s.id == "SCN-02")
    route = write_routes(scn, 0)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _run(route, a, length_s=50, seed=0)
    _run(route, b, length_s=50, seed=0)
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def test_tracing_off_by_default(tmp_path) -> None:
    scn = next(s for s in load_all() if s.id == "SCN-02")
    route = write_routes(scn, 0)
    env = SUMOEnv(route, episode_length_s=20)  # no trace_path
    try:
        env.reset()
        env.step(0)
        assert env._tracer is None  # no tracing overhead unless asked
    finally:
        env.close()
