"""T-01-06 - Provenance chain (hard-rule #7).

Public surface:

- version strings + filename: :func:`src.provenance.versions.data_version`,
  :func:`~src.provenance.versions.lstm_version`,
  :func:`~src.provenance.versions.new_run_id`,
  :func:`~src.provenance.versions.checkpoint_filename`, plus the best-effort
  collectors ``git_sha`` / ``sumo_version`` / ``torch_versions``.
- SQLite writers: :func:`src.provenance.records.record_experiment_run`,
  :func:`~src.provenance.records.record_model_artifact`.
"""

from __future__ import annotations

from src.provenance.records import record_experiment_run, record_model_artifact
from src.provenance.versions import (
    checkpoint_filename,
    config_hash,
    data_version,
    git_sha,
    hash_files,
    lstm_version,
    new_run_id,
    sumo_version,
    torch_versions,
)

__all__ = [
    "checkpoint_filename",
    "config_hash",
    "data_version",
    "git_sha",
    "hash_files",
    "lstm_version",
    "new_run_id",
    "record_experiment_run",
    "record_model_artifact",
    "sumo_version",
    "torch_versions",
]
