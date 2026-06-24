"""T-03-01 - LSTM training-data loader: window the 50 CSVs into (12,24)->(3,12).

Turns the per-decision-step CSVs from ``scripts/generate_lstm_data`` (T-01-05) into
the supervised sequences the forecaster trains on (``lstm-forecasting.md``):

* **input**  ``(12, 24)`` - 12 history steps (120 sim-s) x 24 features per step
  (12 per-movement queue lengths + 12 per-movement vehicle counts);
* **target** ``(3, 12)``  - the next 3 steps (30 sim-s) of **queue** per movement
  (counts are an input feature only, never forecast - lstm-forecasting.md).

CSV columns (the T-01-05 header): ``step, sim_time, q_M0..q_M11, c_M0..c_M11`` ->
the 24 features are columns ``[2:26]`` in order ``[q0..q11, c0..c11]``; the queue
target is the first 12 of those.

PINNED (DoD: no cross-reference to "Chat 2"):

* ``INPUT_LEN = 12``  history steps;
* ``HORIZON   = 3``   forecast steps;
* ``STRIDE    = 1``   one window per start row (=> ~346 windows/360-row file,
  ~10k train sequences, matching lstm-forecasting.md's data-volume note).

Split is **scenario-level, not random-window** (lstm-forecasting.md "Why
scenario-level split"): train SCN-01/02/03, val SCN-04, test SCN-05. Windows are
built **per file**, so a single window never spans two episodes and never crosses a
train/val/test boundary - the leakage guard the DoD requires. A random window split
would leak: windows from one scenario share dynamics, so validating on other windows
of a *seen* scenario lets the model see the val distribution.

Normalization: NOT applied here - the loader yields raw queue/count values and the
training loop owns any scaling (lstm-forecasting.md trains on raw MSE; queue and
count share the same vehicle-count scale). Surfaced, not silently decided.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data" / "lstm"

INPUT_LEN = 12
HORIZON = 3
N_MOVEMENTS = 12
N_FEATURES = 24  # 12 queue + 12 count
STRIDE = 1
_WINDOW_SPAN = INPUT_LEN + HORIZON  # rows consumed by one (input, target) pair

# Scenario-level split (lstm-forecasting.md "Training data" table). Disjoint by
# construction - asserted in the leakage test.
SPLITS: dict[str, tuple[str, ...]] = {
    "train": ("SCN-01", "SCN-02", "SCN-03"),
    "val": ("SCN-04",),
    "test": ("SCN-05",),
}


def _scenario_glob(scenario_id: str) -> str:
    """``"SCN-01"`` -> ``"scn_01_seed_*.csv"`` (the T-01-05 file naming)."""
    return f"scn_{scenario_id.split('-')[1]}_seed_*.csv"


def files_for_split(split: str, data_dir: Path = _DATA_DIR) -> list[Path]:
    """All CSVs belonging to ``split``, sorted (deterministic ordering)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    files: list[Path] = []
    for scenario_id in SPLITS[split]:
        files += sorted(Path(data_dir).glob(_scenario_glob(scenario_id)))
    return files


def _load_features(path: Path) -> np.ndarray:
    """Load one CSV's feature matrix ``(n_rows, 24)`` = ``[q0..q11, c0..c11]``."""
    rows = path.read_text(encoding="utf-8").splitlines()[1:]  # drop header
    if not rows:
        return np.empty((0, N_FEATURES), dtype=np.float32)
    data = np.array([[float(c) for c in line.split(",")] for line in rows], dtype=np.float32)
    return data[:, 2:]  # drop step + sim_time -> the 24 features


def _window_file(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over ONE file's features -> ``(X, Y)``.

    ``X`` is ``(w, 12, 24)``; ``Y`` is ``(w, 3, 12)`` (queue = first 12 features of
    the 3 future rows). Windows never reach past ``len(feats)``, so they never span
    into another file when callers concatenate per-file results.
    """
    n = len(feats)
    starts = range(0, n - _WINDOW_SPAN + 1, STRIDE)
    xs = [feats[i : i + INPUT_LEN] for i in starts]
    ys = [feats[i + INPUT_LEN : i + _WINDOW_SPAN, :N_MOVEMENTS] for i in starts]
    if not xs:
        return (
            np.empty((0, INPUT_LEN, N_FEATURES), dtype=np.float32),
            np.empty((0, HORIZON, N_MOVEMENTS), dtype=np.float32),
        )
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


class LSTMDataset(Dataset):
    """Windowed (input, target) pairs for one split (``train`` / ``val`` / ``test``).

    Each item is ``(x, y)`` with ``x: (12, 24)`` and ``y: (3, 12)`` float tensors.
    ``window_sources[i]`` is the file-stem each window came from - used by the
    leakage test to prove no window mixes scenarios.
    """

    def __init__(self, files: list[Path]) -> None:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        self.window_sources: list[str] = []
        for path in files:
            x, y = _window_file(_load_features(path))
            if len(x) == 0:
                continue
            xs.append(x)
            ys.append(y)
            self.window_sources += [path.stem] * len(x)
        self._x = (
            torch.from_numpy(np.concatenate(xs)) if xs
            else torch.empty((0, INPUT_LEN, N_FEATURES))
        )
        self._y = (
            torch.from_numpy(np.concatenate(ys)) if ys
            else torch.empty((0, HORIZON, N_MOVEMENTS))
        )

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx]


def load_split(split: str, data_dir: Path = _DATA_DIR) -> LSTMDataset:
    """Build the ``LSTMDataset`` for one split from its scenario CSVs."""
    return LSTMDataset(files_for_split(split, data_dir))


def make_dataloaders(
    data_dir: Path = _DATA_DIR, *, batch_size: int = 64
) -> dict[str, DataLoader]:
    """Train/val/test ``DataLoader``s (train shuffled; val/test in order).

    Batch size 64 per lstm-forecasting.md. Windows themselves carry no
    cross-scenario leakage (scenario-level split + per-file windowing), so
    shuffling the train windows is safe.
    """
    return {
        split: DataLoader(
            load_split(split, data_dir),
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
        for split in SPLITS
    }


def split_sizes(data_dir: Path = _DATA_DIR) -> dict[str, int]:
    """``{split: n_windows}`` - for documenting the train/val/test sizes (DoD)."""
    return {split: len(load_split(split, data_dir)) for split in SPLITS}
