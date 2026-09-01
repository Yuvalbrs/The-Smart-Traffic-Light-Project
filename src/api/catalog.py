"""Model catalogue and the cross-controller comparison, both read from what is on disk.

Two read-only endpoints behind the application's "train a model, then compare the models" flow:

* ``GET /models`` - what has actually been trained, discovered by walking ``runs/`` rather than
  from a hard-coded list, so a model the user trains inside the app appears next to the campaign's
  own cells without anything being registered anywhere.
* ``GET /comparison`` - every controller's KPIs on one scenario, averaged over the episodes in the
  results database.

**The comparison does not decide a winner.** It returns the numbers and which direction is better
per KPI, and lets the client mark the best cell. Nothing here privileges the project's own
controller: on the data this was written against, Webster beats the hybrid agent on SCN-05 average
wait while the DQN variants lead on SCN-01/03/04, and a comparison that could only ever show the
author's model winning would be worth nothing to a reader.

Every response carries a provenance block, because the database can hold rows from before a fix
landed. A table that cannot say which code produced it is exactly how a stale number gets quoted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNS_DIR = _REPO_ROOT / "runs"

router = APIRouter(tags=["catalog"])

#: The seven KPIs, with the direction that counts as better. ``lower_is_better`` is the whole
#: reason the client can mark a best cell without re-deriving domain knowledge in TypeScript.
KPI_SPEC: tuple[dict[str, Any], ...] = (
    {"key": "avg_waiting_time", "label": "Avg wait (s)", "lower_is_better": True},
    {"key": "throughput", "label": "Throughput (veh/h)", "lower_is_better": False},
    {"key": "avg_queue_length", "label": "Avg queue (veh)", "lower_is_better": True},
    {"key": "wait_p95", "label": "P95 wait (s)", "lower_is_better": True},
    {"key": "worst_movement_max_wait", "label": "Worst-movement max wait (s)", "lower_is_better": True},
    {"key": "num_stops", "label": "Stops / veh", "lower_is_better": True},
    {"key": "fairness_std", "label": "Fairness (SD across movements)", "lower_is_better": True},
)

#: Display labels, and which row is the project's own contribution. ``is_ours`` drives a highlight
#: in the UI - it marks the row, it does not mark it as the winner.
CONTROLLER_META: dict[str, dict[str, Any]] = {
    "hybrid": {"label": "DQN + forecast (ours)", "is_ours": True},
    "plain": {"label": "DQN, no forecast", "is_ours": False},
    "random-lstm": {"label": "DQN + random forecast (control)", "is_ours": False},
    "sel/plain": {"label": "DQN with regime selector", "is_ours": True},
    "dqn-plain": {"label": "DQN, no forecast (legacy rows)", "is_ours": False},
    "webster": {"label": "Webster fixed-time", "is_ours": False},
    "max_pressure": {"label": "Max-pressure", "is_ours": False},
    "actuated": {"label": "SUMO actuated", "is_ours": False},
}

_VARIANT_LABEL = {
    "plain": "DQN, no forecast",
    "hybrid": "DQN + forecast",
    "random-lstm": "DQN + random forecast (control)",
}


def _describe_run_dir(path: Path) -> dict[str, Any] | None:
    """One entry for ``GET /models``, or ``None`` if the directory holds no usable model."""
    ckpt_dir = path / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    checkpoints = sorted(ckpt_dir.glob("ep*.pt"), key=lambda p: int(p.stem[2:]))
    if not checkpoints:
        return None

    cfg: dict[str, Any] = {}
    cfg_path = path / "config.yaml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            cfg = {}  # a half-written config during a live run must not break the whole listing

    variant = cfg.get("variant") or path.name.split("_seed")[0]
    seed = cfg.get("seed")
    episodes = cfg.get("n_episodes")
    last_ep = int(checkpoints[-1].stem[2:])
    # "final" means the run reached its last episode. Without it the model is a partial run, which
    # is fine to show but must not be silently compared against a fully trained one.
    has_final = episodes is not None and last_ep >= int(episodes) - 1

    label = _VARIANT_LABEL.get(str(variant), str(variant))
    if seed is not None:
        label = f"{label} (seed {seed})"
    return {
        "id": path.name,
        "variant": variant,
        "seed": seed,
        "episodes": episodes,
        "episodes_trained": last_ep + 1,
        "obs_dim": cfg.get("obs_dim"),
        "lstm_version": cfg.get("lstm_version") or None,
        "git_sha": cfg.get("git_sha") or None,
        "has_final": bool(has_final),
        "label": label,
        # A run started from the app is named ui_*; anything else came from the training matrix.
        "source": "user" if path.name.startswith("ui_") else "matrix",
        # Relative when it can be (short and portable in the UI), absolute otherwise: a runs
        # directory pointed outside the repo must not raise and take the whole listing with it.
        "checkpoint": str(
            checkpoints[-1].relative_to(_REPO_ROOT)
            if checkpoints[-1].is_relative_to(_REPO_ROOT)
            else checkpoints[-1]
        ),
    }


@router.get("/models")
def list_models() -> dict[str, Any]:
    """Every trained model on disk, newest first, quarantined directories excluded."""
    if not _RUNS_DIR.is_dir():
        return {"models": [], "note": "no runs/ directory yet - train a model first"}
    models = []
    for path in sorted(_RUNS_DIR.iterdir()):
        # Leading underscore is this project's quarantine convention (_pre_a6_*, _crash*): those
        # hold models from a superseded world and must never be offered for comparison.
        if not path.is_dir() or path.name.startswith("_"):
            continue
        entry = _describe_run_dir(path)
        if entry is not None:
            models.append(entry)
    models.sort(key=lambda m: (m["source"] != "user", m["id"]))
    return {
        "models": models,
        "note": (
            "Models are discovered from runs/. 'matrix' entries come from the training campaign; "
            "'user' entries were trained from this application."
        ),
    }


_COMPARISON_SQL = text(
    """
    SELECT r.controller                        AS controller,
           COUNT(*)                            AS n_episodes,
           AVG(k.avg_waiting_time)             AS avg_waiting_time,
           AVG(k.throughput)                   AS throughput,
           AVG(k.avg_queue_length)             AS avg_queue_length,
           AVG(k.wait_p95)                     AS wait_p95,
           AVG(k.worst_movement_max_wait)      AS worst_movement_max_wait,
           AVG(k.num_stops)                    AS num_stops,
           AVG(k.fairness_std)                 AS fairness_std,
           AVG(CAST(e.gridlock_censored AS FLOAT)) AS gridlock_rate,
           MAX(r.git_sha)                      AS git_sha,
           MAX(r.created_at)                   AS newest_run_at
      FROM experiment_run r
      JOIN episode       e ON e.run_id_fk    = r.id
      JOIN episode_kpi   k ON k.episode_id_fk = e.id
     WHERE e.scenario = :scenario
       AND r.mode     = :mode
     GROUP BY r.controller
    """
)


@router.get("/comparison")
def comparison(
    request: Request,
    scenario: str = Query("SCN-05", description="scenario id, e.g. SCN-05"),
    mode: str = Query("eval", description="experiment_run.mode to compare over"),
) -> dict[str, Any]:
    """Average KPIs per controller on one scenario, straight from the results database."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:  # pragma: no cover - the app always builds one
        raise HTTPException(status_code=503, detail="results database is not available")

    try:
        with engine.connect() as conn:
            records = [dict(row) for row in conn.execute(
                _COMPARISON_SQL, {"scenario": scenario, "mode": mode}
            ).mappings()]
    except DatabaseError as exc:
        # A fresh install has a database file with no schema in it yet - the tables appear on the
        # first run that writes. That is an empty result, not a server fault, and it is exactly
        # what someone opening the Compare tab before running anything would hit.
        raise HTTPException(
            status_code=404,
            detail=(
                "the results database has no experiment tables yet. Run an episode or the "
                "evaluation campaign (scripts/eval_runner.py) to populate it."
            ),
        ) from exc

    if not records:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no {mode} episodes recorded for {scenario}. Run the evaluation campaign "
                f"(scripts/eval_runner.py) or pick another scenario."
            ),
        )

    rows = []
    for rec in records:
        meta = CONTROLLER_META.get(rec["controller"], {"label": rec["controller"], "is_ours": False})
        row: dict[str, Any] = {
            "controller": rec["controller"],
            "label": meta["label"],
            "is_ours": meta["is_ours"],
            "n_episodes": int(rec["n_episodes"]),
            "gridlock_rate": _round(rec["gridlock_rate"], 3),
        }
        for spec in KPI_SPEC:
            row[spec["key"]] = _round(rec[spec["key"]], 2)
        rows.append(row)
    # Order by the headline KPI so the table reads as a ranking without the client sorting it.
    rows.sort(key=lambda r: (r["avg_waiting_time"] is None, r["avg_waiting_time"] or 0.0))

    shas = sorted({r["git_sha"] for r in records if r["git_sha"]})
    newest = max((r["newest_run_at"] for r in records if r["newest_run_at"]), default=None)
    return {
        "scenario": scenario,
        "mode": mode,
        "rows": rows,
        "kpis": [dict(spec) for spec in KPI_SPEC],
        "provenance": {
            "git_shas": shas,
            "newest_run_at": str(newest) if newest else None,
            "source": "data/traffic.db",
        },
        "note": (
            "Averages over the episodes in the results database. Episodes are paired by seed "
            "across controllers. A high gridlock rate means the controller could not clear the "
            "demand, so its wait and throughput are not directly comparable to an uncensored "
            "row - that is why the rate is shown beside them."
        ),
    }


def _round(value: Any, digits: int) -> float | None:
    """Round, preserving null. A missing KPI must stay missing rather than become 0.0."""
    return None if value is None else round(float(value), digits)
