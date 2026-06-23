"""T-01-05 - Tests for the LSTM data-generation pipeline.

DoD (measurable parts): generates a CSV per (scenario, seed) in the LSTM input
format (24 per-movement features), deterministic per seed, queue (halting) and
count are distinct quantities. The full 5x10 sweep is a CLI run, not a unit test;
here we use short episodes to keep the suite fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.build_network import build_net
from scripts.generate_lstm_data import _HEADER, generate_one
from src.env.intersection import N_MOVEMENTS
from src.scenarios.config import load_all


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    build_net()


def _read(path) -> list[list[str]]:
    return [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()]


def test_csv_shape_and_header(tmp_path) -> None:
    scn = next(s for s in load_all() if s.id == "SCN-02")  # heavy -> non-empty queues
    path, n_rows = generate_one(scn, 0, out_dir=tmp_path, episode_length_s=120)
    table = _read(path)
    assert table[0] == _HEADER
    assert len(_HEADER) == 2 + 2 * N_MOVEMENTS  # step, sim_time, 12 queue, 12 count
    assert n_rows == len(table) - 1
    assert n_rows == 12  # 120 s / 10 s decision interval
    # sim_time advances in 10 s steps
    assert [row[1] for row in table[1:]] == [f"{(i + 1) * 10}" for i in range(n_rows)]


def test_deterministic_same_seed(tmp_path) -> None:
    scn = next(s for s in load_all() if s.id == "SCN-02")
    a, _ = generate_one(scn, 3, out_dir=tmp_path / "a", episode_length_s=120)
    b, _ = generate_one(scn, 3, out_dir=tmp_path / "b", episode_length_s=120)
    assert a.read_bytes() == b.read_bytes()  # same seed -> byte-identical CSV


def test_queue_is_subset_of_count(tmp_path) -> None:
    """Halting (queue) <= vehicle count per movement, and they are not identical."""
    scn = next(s for s in load_all() if s.id == "SCN-02")
    path, _ = generate_one(scn, 1, out_dir=tmp_path, episode_length_s=300)
    data = np.array([[int(c) for c in row[2:]] for row in _read(path)[1:]])
    queue, count = data[:, :N_MOVEMENTS], data[:, N_MOVEMENTS:]
    assert np.all(queue <= count)  # halting vehicles are a subset of vehicles on lane
    assert np.any(count > queue)  # ... and strictly more at some point (moving traffic)
    assert count.sum() > 0  # the heavy scenario actually produced traffic


def test_counts_are_nonnegative_integers(tmp_path) -> None:
    scn = next(s for s in load_all() if s.id == "SCN-01")
    path, _ = generate_one(scn, 0, out_dir=tmp_path, episode_length_s=120)
    for row in _read(path)[1:]:
        for cell in row[2:]:
            assert cell.lstrip("-").isdigit() and int(cell) >= 0
