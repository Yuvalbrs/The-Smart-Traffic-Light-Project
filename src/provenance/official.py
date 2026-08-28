"""The ONE place that names the deployed forecaster.

The checkpoint was pinned by hand in five files (``eval_runner``, ``train_matrix``,
``build_analysis``, ``bootstrap_forecaster``, ``collect_dqn_traces``). When the
forecaster was retrained after the gridlock-artefact fixes, two were updated and three
were not - so the analysis layer reported the provenance of a forecaster the experiment
had not used, and the final deliverable printed a stale version string next to correct
results. Sync enforced by a comment is not sync (decisions.md 2026-08-28).

Anything that loads, names, or reports the deployed forecaster imports from here.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = _REPO_ROOT / "checkpoints" / "lstm"

# Deployed forecaster: retrained on the post-gridlock-fix corpus.
# Freeze-gate verdict SHIP_WITH_CAVEAT, skill score @60s = 0.081.
OFFICIAL_LSTM_FILENAME = "lstm__data-aa6ef4458cda__lstm-f70dca8c6ff1.pt"
OFFICIAL_LSTM = CKPT_DIR / OFFICIAL_LSTM_FILENAME


def official_lstm_version() -> str:
    """The ``lstm-<hash>`` id of the deployed forecaster, parsed from its filename.

    Derived rather than restated so the id can never drift from the file actually loaded.
    """
    stem = Path(OFFICIAL_LSTM_FILENAME).stem
    for part in stem.split("__"):
        if part.startswith("lstm-"):
            return part
    raise ValueError(f"cannot parse an lstm version out of {OFFICIAL_LSTM_FILENAME!r}")


def official_data_version() -> str:
    """The ``data-<hash>`` id of the corpus the deployed forecaster was trained on."""
    stem = Path(OFFICIAL_LSTM_FILENAME).stem
    for part in stem.split("__"):
        if part.startswith("data-"):
            return part
    raise ValueError(f"cannot parse a data version out of {OFFICIAL_LSTM_FILENAME!r}")
