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


def assert_official_matches_corpus(data_dir: Path | None = None) -> None:
    """Raise unless the pinned forecaster was trained on the corpus currently on disk.

    The single-pin design (this module) fixed forecaster identity drifting across five files. It
    does NOT catch the other half of the problem: the corpus underneath the pin being regenerated
    while the pin stays put. That is a silent failure - every run keeps loading a real checkpoint
    that simply no longer corresponds to ``data/lstm/``, and every results row records a
    data_version that was true yesterday.

    Called by the campaign entry points so a stale pin stops the run instead of quietly poisoning
    a provenance chain. Cheap: it reads one manifest.
    """
    from scripts.train_lstm import _dataset_data_version  # local: avoids a package import cycle

    root = data_dir if data_dir is not None else _REPO_ROOT / "data" / "lstm"
    on_disk = _dataset_data_version(root)
    pinned = official_data_version()
    if on_disk != pinned:
        raise RuntimeError(
            f"OFFICIAL forecaster is stale.\n"
            f"  pinned  (src/provenance/official.py): {pinned}\n"
            f"  on disk ({root}):                     {on_disk}\n"
            f"The pinned checkpoint {OFFICIAL_LSTM_FILENAME!r} was trained on a corpus that is no "
            f"longer the one on disk. Retrain the forecaster and re-pin "
            f"OFFICIAL_LSTM_FILENAME, or restore the old corpus. Refusing to run rather than "
            f"record a data_version that is not true."
        )


def official_data_version() -> str:
    """The ``data-<hash>`` id of the corpus the deployed forecaster was trained on."""
    stem = Path(OFFICIAL_LSTM_FILENAME).stem
    for part in stem.split("__"):
        if part.startswith("data-"):
            return part
    raise ValueError(f"cannot parse a data version out of {OFFICIAL_LSTM_FILENAME!r}")
