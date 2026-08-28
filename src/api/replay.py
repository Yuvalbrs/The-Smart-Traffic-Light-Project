"""T-05-04 - REST replay endpoints.

The five routes are fixed by ``system-architecture-overview.md`` API contracts (as amended by
F10, which dropped ``POST /replay/start`` in favour of REST streaming)::

    GET /runs                     list completed runs (paginated)
    GET /runs/{run_id}/metadata   scenario, seed, algorithm, final KPIs, version chain
    GET /runs/{run_id}/timeline   full vehicle-snapshot history (JSONL stream)
    GET /runs/{run_id}/kpis       pre-computed KPI table
    GET /runs/{run_id}/trace      stream a finished run's frames (ndjson) for Unity/dashboard

Two stores back these, by necessity rather than choice: ``experiment_run`` / ``episode`` /
``episode_kpi`` rows live in SQLite, but ``vehicle_snapshot`` was never populated (0 rows) - the
per-second vehicle history only ever went to JSONL trace files. So metadata and KPIs are served
from the database and frame streams from the trace files on disk, which is also what makes the
streams cheap: they are already ndjson, one ``sim_frame`` per line, and are relayed verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Episode, EpisodeKpi, ExperimentRun

router = APIRouter(tags=["replay"])

_KPI_FIELDS = (
    "avg_waiting_time",
    "avg_queue_length",
    "throughput",
    "num_stops",
    "wait_p95",
    "fairness_std",
    "worst_movement_max_wait",
)


def _run_summary(run: ExperimentRun) -> dict[str, Any]:
    """Flatten one ``experiment_run`` row into the list payload."""
    return {
        "run_id": run.run_id,
        "name": run.name,
        "mode": run.mode,
        "controller": run.controller,
        "created_at": str(run.created_at) if run.created_at else None,
        "version_chain": {
            "data_version": run.data_version,
            "lstm_version": run.lstm_version,
            "git_sha": run.git_sha,
            "sumo_version": run.sumo_version,
            "schema_version": run.schema_version,
        },
    }


def _episode_payload(ep: Episode, kpi: EpisodeKpi | None) -> dict[str, Any]:
    """One episode plus its KPI row, as served by /metadata."""
    return {
        "episode_id": ep.id,
        "index_in_run": ep.index_in_run,
        "scenario": ep.scenario,
        "seed": ep.seed,
        "total_reward": ep.total_reward,
        "done_reason": ep.done_reason,
        "loaded_count": ep.loaded_count,
        "departed_count": ep.departed_count,
        "arrived_count": ep.arrived_count,
        "insertion_backlog_fraction": ep.insertion_backlog_fraction,
        # Surfaced deliberately: a censored episode's KPIs are not comparable
        # (preregistration.md section 8) and any client showing them must be able to say so.
        "gridlock_censored": bool(ep.gridlock_censored),
        "kpis": None if kpi is None else {f: getattr(kpi, f) for f in _KPI_FIELDS},
    }


def _session(request: Request) -> Session:
    """Open a SQLAlchemy session against the app's configured engine."""
    engine = request.app.state.engine
    if engine is None:  # pragma: no cover - guarded at startup
        raise HTTPException(status_code=503, detail="results database not configured")
    return Session(engine)


def _find_run(session: Session, run_id: str) -> ExperimentRun:
    """Look a run up by its ``run_id`` string, 404ing if absent."""
    run = session.scalar(select(ExperimentRun).where(ExperimentRun.run_id == run_id))
    if run is None:
        raise HTTPException(status_code=404, detail="no run with run_id " + repr(run_id))
    return run


def _episodes_with_kpis(session: Session, run: ExperimentRun) -> list[tuple[Episode, EpisodeKpi | None]]:
    """All episodes of a run, in order, each paired with its KPI row if one exists."""
    episodes = session.scalars(
        select(Episode).where(Episode.run_id_fk == run.id).order_by(Episode.index_in_run)
    ).all()
    ids = [e.id for e in episodes] or [-1]
    kpis = {
        k.episode_id_fk: k
        for k in session.scalars(select(EpisodeKpi).where(EpisodeKpi.episode_id_fk.in_(ids))).all()
    }
    return [(e, kpis.get(e.id)) for e in episodes]


def _maybe_json(blob: str | None) -> Any:
    """Parse a JSON config blob, falling back to the raw string if it is not JSON."""
    if not blob:
        return None
    try:
        return json.loads(blob)
    except (TypeError, ValueError):
        return blob


@router.get("/runs")
def list_runs(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """List completed runs, newest first."""
    with _session(request) as session:
        total = len(session.scalars(select(ExperimentRun.id)).all())
        runs = session.scalars(
            select(ExperimentRun).order_by(ExperimentRun.id.desc()).limit(limit).offset(offset)
        ).all()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "runs": [_run_summary(r) for r in runs],
        }


@router.get("/runs/{run_id}/metadata")
def run_metadata(request: Request, run_id: str) -> dict[str, Any]:
    """Scenario, seeds, algorithm, final KPIs and the full version chain for one run."""
    with _session(request) as session:
        run = _find_run(session, run_id)
        pairs = _episodes_with_kpis(session, run)
        payload = _run_summary(run)
        payload["config"] = _maybe_json(run.config)
        payload["episode_count"] = len(pairs)
        payload["scenarios"] = sorted({e.scenario for e, _ in pairs if e.scenario})
        payload["seeds"] = sorted({e.seed for e, _ in pairs if e.seed is not None})
        payload["episodes"] = [_episode_payload(e, k) for e, k in pairs]
        return payload


@router.get("/runs/{run_id}/kpis")
def run_kpis(request: Request, run_id: str) -> dict[str, Any]:
    """The pre-computed KPI table for one run, one row per episode."""
    with _session(request) as session:
        run = _find_run(session, run_id)
        rows = []
        for ep, kpi in _episodes_with_kpis(session, run):
            row: dict[str, Any] = {
                "episode_id": ep.id,
                "scenario": ep.scenario,
                "seed": ep.seed,
                "gridlock_censored": bool(ep.gridlock_censored),
            }
            row.update({f: (None if kpi is None else getattr(kpi, f)) for f in _KPI_FIELDS})
            rows.append(row)
        return {
            "run_id": run_id,
            "columns": ["episode_id", "scenario", "seed", "gridlock_censored", *_KPI_FIELDS],
            "rows": rows,
        }


def _trace_files(
    trace_dirs: list[Path],
    run_id: str,
    scenario: str | None,
    seed: int | None,
    algo: str | None,
) -> list[Path]:
    """Find trace files for a run.

    Live sessions write ``<run_id>.jsonl``; the eval corpus uses the historical
    ``{scenario}_seed{seed}_{algo}.jsonl`` naming, so both are matched.
    """
    hits: list[Path] = []
    for directory in trace_dirs:
        if not directory.is_dir():
            continue
        hits.extend(sorted(directory.glob(run_id + "*.jsonl")))
        if scenario and seed is not None:
            tail = (algo + ".jsonl") if algo else "*.jsonl"
            hits.extend(sorted(directory.glob(scenario + "_seed" + str(seed) + "_" + tail)))
    out: list[Path] = []
    for hit in hits:
        # Containment check: `scenario`/`algo` reach glob() straight from the query
        # string, and Path.glob resolves ".." as real traversal, so a crafted scenario
        # can address files outside the trace dirs. Keep only hits that stay inside.
        try:
            resolved = hit.resolve()
            if not any(
                resolved.is_relative_to(d.resolve()) for d in trace_dirs if d.is_dir()
            ):
                continue
        except (OSError, ValueError):
            continue
        if resolved not in out:
            out.append(resolved)
    return out


def _stream_ndjson(path: Path, limit: int | None) -> Iterator[bytes]:
    """Relay a JSONL trace line by line without loading it into memory."""
    sent = 0
    with path.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield line if line.endswith(b"\n") else line + b"\n"
            sent += 1
            if limit is not None and sent >= limit:
                return


def _resolve_trace(request: Request, run_id: str, seed: int | None, scenario: str | None) -> Path:
    """Pick the trace file to stream for this run, 404ing with a useful message."""
    with _session(request) as session:
        run = _find_run(session, run_id)
        controller = run.controller
        episodes = session.scalars(select(Episode).where(Episode.run_id_fk == run.id)).all()
        scn = scenario or (episodes[0].scenario if episodes else None)
        sd = seed if seed is not None else (episodes[0].seed if episodes else None)
    files = _trace_files(request.app.state.trace_dirs, run_id, scn, sd, controller)
    if not files:
        raise HTTPException(
            status_code=404,
            detail=(
                "no trace file on disk for run " + repr(run_id) + " (looked for "
                + run_id + "*.jsonl and " + str(scn) + "_seed" + str(sd) + "_"
                + str(controller) + ".jsonl). Per-second vehicle history lives in JSONL "
                "traces, not in the database."
            ),
        )
    return files[0]


@router.get("/runs/{run_id}/trace")
def run_trace(
    request: Request,
    run_id: str,
    seed: int | None = None,
    scenario: str | None = None,
    limit: int | None = Query(None, ge=1),
) -> StreamingResponse:
    """Stream a finished run's 1 Hz ``sim_frame`` history as ndjson (Unity + dashboard replay)."""
    path = _resolve_trace(request, run_id, seed, scenario)
    return StreamingResponse(
        _stream_ndjson(path, limit),
        media_type="application/x-ndjson",
        headers={"X-Trace-File": path.name},
    )


@router.get("/runs/{run_id}/timeline")
def run_timeline(
    request: Request,
    run_id: str,
    seed: int | None = None,
    scenario: str | None = None,
    limit: int | None = Query(None, ge=1),
) -> StreamingResponse:
    """Full vehicle-snapshot history. Same frames as ``/trace``; kept as the documented alias."""
    return run_trace(request, run_id, seed=seed, scenario=scenario, limit=limit)
