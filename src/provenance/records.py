"""T-01-06 - SQLite writers that stamp the provenance chain onto result rows.

Thin helpers over the T-01-03 ORM models: they create an ``experiment_run`` with
its full version tuple, and ``model_artifact`` rows whose ``path`` embeds the same
chain (hard-rule #7: every checkpoint filename embeds the versions, every results
row records ``(data_version, lstm_version, run_id, git_sha)``).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.db.models import ExperimentRun, ModelArtifact


def record_experiment_run(
    session: Session,
    *,
    name: str,
    mode: str,
    controller: str,
    config: dict[str, Any],
    run_id: str,
    data_version: str | None = None,
    lstm_version: str | None = None,
    git_sha: str | None = None,
    sumo_version: str | None = None,
) -> ExperimentRun:
    """Create and persist an ``ExperimentRun`` carrying the full provenance tuple.

    The row is added and flushed (so ``run.id`` is populated) but **not**
    committed - the caller owns the transaction boundary.
    """
    run = ExperimentRun(
        name=name,
        mode=mode,
        controller=controller,
        config=config,
        run_id=run_id,
        data_version=data_version,
        lstm_version=lstm_version,
        git_sha=git_sha,
        sumo_version=sumo_version,
    )
    session.add(run)
    session.flush()
    return run


def record_model_artifact(
    session: Session,
    *,
    run: ExperimentRun,
    kind: str,
    path: str,
    step: int,
    metrics: dict[str, Any] | None = None,
) -> ModelArtifact:
    """Create and persist a ``ModelArtifact`` linked to ``run`` (added + flushed)."""
    artifact = ModelArtifact(
        run=run,
        kind=kind,
        path=path,
        step=step,
        metrics=metrics,
    )
    session.add(artifact)
    session.flush()
    return artifact
