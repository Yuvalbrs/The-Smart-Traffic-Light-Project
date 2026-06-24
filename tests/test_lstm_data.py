"""T-03-01 - Tests for the LSTM data loader.

DoD: correctly-shaped (12,24)->(3,12) tensors; train/val/test sizes documented;
and a leakage test - no window spans two files / crosses a scenario split.

Uses tiny SYNTHETIC CSVs (no SUMO): each cell encodes its (row, kind) so the
window slicing can be hand-verified exactly, like the KPI-extractor tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.lstm_data import (
    DEFAULT_TARGET_OFFSETS,
    HORIZON,
    INPUT_LEN,
    N_FEATURES,
    N_MOVEMENTS,
    SPLITS,
    LSTMDataset,
    files_for_split,
    load_split,
    make_dataloaders,
    split_sizes,
)

_SPAN = INPUT_LEN + max(DEFAULT_TARGET_OFFSETS)  # rows consumed by one window
_N_ROWS = 40                                     # per synthetic file -> _N_ROWS-_SPAN+1 windows
_WINDOWS_PER_FILE = _N_ROWS - _SPAN + 1

_HEADER = (
    ["step", "sim_time"]
    + [f"q_M{i}" for i in range(N_MOVEMENTS)]
    + [f"c_M{i}" for i in range(N_MOVEMENTS)]
)


def _write_csv(path, n_rows: int) -> None:
    """Row r (0-based): every queue = r, every count = 100 + r. Easy to hand-check."""
    lines = [",".join(_HEADER)]
    for r in range(n_rows):
        cells = [str(r + 1), str((r + 1) * 10)]
        cells += [str(r)] * N_MOVEMENTS           # q_* = r
        cells += [str(100 + r)] * N_MOVEMENTS     # c_* = 100 + r
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def data_dir(tmp_path):
    """One CSV per split scenario; _N_ROWS rows -> _WINDOWS_PER_FILE windows each."""
    for scenario_ids in SPLITS.values():
        for scenario_id in scenario_ids:
            prefix = f"scn_{scenario_id.split('-')[1]}"
            _write_csv(tmp_path / f"{prefix}_seed_00.csv", n_rows=_N_ROWS)
    return tmp_path


def test_shapes_and_dtypes(data_dir) -> None:
    ds = load_split("train", data_dir)
    x, y = ds[0]
    assert tuple(x.shape) == (INPUT_LEN, N_FEATURES)  # (12, 24)
    assert tuple(y.shape) == (HORIZON, N_MOVEMENTS)    # (3, 12)
    assert x.dtype.is_floating_point and y.dtype.is_floating_point


def test_window_values_are_correctly_sliced(data_dir) -> None:
    """First window: input rows 0..11; targets at last-history-step + each offset."""
    ds = LSTMDataset([data_dir / "scn_01_seed_00.csv"])
    x, y = ds[0]
    # input: queue feature (col 0) equals the row index; count (col 12) = 100 + row
    assert x[0, 0].item() == 0 and x[INPUT_LEN - 1, 0].item() == INPUT_LEN - 1
    assert x[0, N_MOVEMENTS].item() == 100.0
    # target rows = (last history row = INPUT_LEN-1) + offset; queue value == row index
    expected = [float(INPUT_LEN - 1 + o) for o in DEFAULT_TARGET_OFFSETS]
    assert [y[h, 0].item() for h in range(HORIZON)] == expected


def test_windows_per_file_count(data_dir) -> None:
    """One file -> _WINDOWS_PER_FILE windows; train has 3 scenarios x 1 file."""
    assert len(LSTMDataset([data_dir / "scn_01_seed_00.csv"])) == _WINDOWS_PER_FILE
    assert len(load_split("train", data_dir)) == 3 * _WINDOWS_PER_FILE


def test_no_leakage_across_files_or_splits(data_dir) -> None:
    """Every window traces to ONE file, and the three splits are disjoint."""
    sources = {split: set(load_split(split, data_dir).window_sources) for split in SPLITS}
    # each split's windows only come from that split's scenario files
    for split, scenario_ids in SPLITS.items():
        prefixes = {f"scn_{s.split('-')[1]}" for s in scenario_ids}
        assert all(src.split("_seed_")[0] in prefixes for src in sources[split])
    # and no file appears in two splits
    assert sources["train"].isdisjoint(sources["val"])
    assert sources["train"].isdisjoint(sources["test"])
    assert sources["val"].isdisjoint(sources["test"])


def test_no_window_spans_two_files(data_dir) -> None:
    """Concatenating per-file windows must not invent cross-file windows."""
    files = files_for_split("train", data_dir)
    per_file = sum(len(LSTMDataset([f])) for f in files)
    assert len(load_split("train", data_dir)) == per_file  # no extra cross-file windows


def test_dataloaders_batch(data_dir) -> None:
    loaders = make_dataloaders(data_dir, batch_size=4)
    x, y = next(iter(loaders["train"]))
    assert tuple(x.shape) == (4, INPUT_LEN, N_FEATURES)
    assert tuple(y.shape) == (4, HORIZON, N_MOVEMENTS)


def test_split_sizes_documented(data_dir) -> None:
    sizes = split_sizes(data_dir)
    assert sizes == {
        "train": 3 * _WINDOWS_PER_FILE, "val": _WINDOWS_PER_FILE, "test": _WINDOWS_PER_FILE,
    }


def test_real_data_if_present() -> None:
    """If the real 50 CSVs exist, the split sizes are sane and disjoint-nonzero."""
    from src.ml.lstm_data import _DATA_DIR

    if not list(_DATA_DIR.glob("scn_*_seed_*.csv")):
        pytest.skip("real LSTM CSVs not generated")
    sizes = split_sizes()
    assert sizes["train"] > sizes["val"] > 0 and sizes["test"] > 0
