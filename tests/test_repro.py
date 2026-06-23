"""T-01-10 / T-01-07 - the golden hash + reproducibility smoke test.

The committed golden (``golden_hashes.json``) must still be reproduced by a fresh
run, and the reference episode must be deterministic. This doubles as a CI gate:
if SUMO / the net / the route generator / the env / the tracer ever drifts, this
test fails the same way the weekly ``scripts.repro_smoke`` would.
"""

from __future__ import annotations

import json

import pytest

from src.repro.reference import GOLDEN_FILE, compute_reference_hash


@pytest.fixture(scope="module")
def reference() -> dict:
    return compute_reference_hash()


def test_golden_file_committed_and_well_formed() -> None:
    assert GOLDEN_FILE.exists(), "run `python -m scripts.golden_hash` and commit golden_hashes.json"
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    assert {"sha256", "n_frames", "scenario", "seed", "episode_length_s"} <= set(golden)
    assert len(golden["sha256"]) == 64


def test_current_run_reproduces_golden(reference) -> None:
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    assert reference["sha256"] == golden["sha256"]  # the repro smoke test, as CI
    assert reference["n_frames"] == golden["n_frames"]


def test_reference_episode_is_deterministic(reference) -> None:
    assert compute_reference_hash()["sha256"] == reference["sha256"]
