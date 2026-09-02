"""Evaluate one user-trained model, so a model trained in the app can reach the comparison.

Training a controller produces a checkpoint. The comparison reads EVALUATION rows out of the
results database. Without this module the two never meet: a model trained in the application had
no rows, so it could never appear beside the baselines - which made "train a model, then compare
it" impossible in the UI even though both halves existed.

The job runs ``scripts.eval_runner --model-dir ...``, which evaluates that one checkpoint against
the three baselines on shared seeds (the paired design the statistics require) and writes the rows
under a ``ui:`` controller name. That prefix is what keeps these rows distinguishable from the
pre-registered campaign's for the rest of their life: a model trained in the app used a different
episode budget, at a different time, on whatever code was current, and averaging it into a
campaign mean would corrupt a pre-registered result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.api.jobs import JobBusyError, ProcessJob, now_iso, spawn
from src.api.wire import SCHEMA_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNS_DIR = _REPO_ROOT / "runs"

EVALUATION_MSG = "evaluation_frame"

#: Seeds one UI evaluation may use. The campaign uses 15 (preregistration A1.3); more than that
#: from a web form is a long job started by a stray keystroke, and fewer than 1 is not a run.
MAX_UI_SEEDS = 15

# "[eval] SCN-05 seed7000 ui:demo            wait=  23.9 thru= 1827.0"
_EPISODE_RE = re.compile(r"^\[eval\]\s+(?P<scenario>SCN-\S+)\s+seed(?P<seed>\d+)\s+(?P<algo>\S+)\s+wait=")


@dataclass
class EvaluationStatus:
    """Everything the UI needs about one evaluation. Serialized as-is onto the wire."""

    job_id: str
    model_id: str
    label: str
    scenario: str
    seeds: int
    status: str = "running"  # running | done | failed | cancelled
    episodes_done: int = 0
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    error: str | None = None

    @property
    def episodes_total(self) -> int:
        """The model plus the three baselines, on every seed - eval_runner's paired design."""
        return self.seeds * 4

    @property
    def pct(self) -> float:
        total = self.episodes_total
        return round(100.0 * self.episodes_done / total, 1) if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "model_id": self.model_id,
            "label": self.label,
            "scenario": self.scenario,
            "seeds": self.seeds,
            "episodes_done": self.episodes_done,
            "episodes_total": self.episodes_total,
            "pct": self.pct,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    def as_frame(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "type": EVALUATION_MSG, **self.as_dict()}


def _label_for(model_id: str) -> str:
    """A short, filesystem- and SQL-safe label from a run-directory name."""
    label = model_id[3:] if model_id.startswith("ui_") else model_id
    return re.sub(r"[^A-Za-z0-9_.-]", "-", label)[:40] or "model"


class EvaluationManager:
    """One evaluation at a time, and the record of the most recent one."""

    def __init__(self, hub, runs_dir: Path | None = None, eval_seeds_start: int = 7000) -> None:
        self._hub = hub
        self._runs_dir = runs_dir or _RUNS_DIR
        self._seed0 = eval_seeds_start
        self._job: ProcessJob | None = None
        self._status: EvaluationStatus | None = None
        self._counter = 0

    @property
    def current(self) -> EvaluationStatus | None:
        return self._status

    @property
    def busy(self) -> bool:
        return self._job is not None and self._job.running

    def start(self, *, model_id: str, scenario: str, seeds: int) -> EvaluationStatus:
        """Evaluate one trained run directory. Raises :class:`JobBusyError` if one is running."""
        if not 1 <= seeds <= MAX_UI_SEEDS:
            raise ValueError(f"seeds must be between 1 and {MAX_UI_SEEDS}, got {seeds}")
        # Reject a path rather than a directory name: model_id reaches this from an HTTP body and
        # is spliced into a command line, so "../.." must not be able to point it anywhere.
        if "/" in model_id or "\\" in model_id or model_id in {"", ".", ".."}:
            raise ValueError(f"model_id must be a run-directory name, got {model_id!r}")
        run_dir = self._runs_dir / model_id
        if not (run_dir / "checkpoints").is_dir():
            raise FileNotFoundError(
                f"{model_id} has no checkpoints/ - train it before evaluating it"
            )
        if self.busy:
            raise JobBusyError(
                f"evaluation {self._status.job_id} is still running "  # type: ignore[union-attr]
                f"({self._status.episodes_done}/{self._status.episodes_total} episodes)"  # type: ignore[union-attr]
            )

        self._counter += 1
        label = _label_for(model_id)
        status = EvaluationStatus(
            job_id=f"e-{self._counter}", model_id=model_id, label=label,
            scenario=scenario, seeds=seeds,
        )
        cmd = [
            "python", "-u", "-m", "scripts.eval_runner",
            "--model-dir", str(run_dir),
            "--model-label", label,
            "--scenarios", scenario,
            # The campaign's own held-out seeds, taken in order, so a UI evaluation is paired
            # against the same traffic the baselines saw rather than a fresh random draw.
            "--eval-seeds", *[str(self._seed0 + i) for i in range(seeds)],
        ]
        import sys

        cmd[0] = sys.executable
        self._status = status
        self._job = ProcessJob(
            spawn(cmd), self._hub,
            on_line=lambda line: self._consume(line),
            status_frame=lambda: status.as_frame(),
            on_finish=lambda code, msg: self._finish(code, msg),
            name=f"eval-{status.job_id}",
        )
        return status

    def _consume(self, line: str) -> bool:
        """One episode line advances the counter; everything else is noise."""
        if self._status is None or not _EPISODE_RE.match(line):
            return False
        self._status.episodes_done += 1
        return True

    def _finish(self, code: int, message: str) -> None:
        status, job = self._status, self._job
        if status is None:
            return  # pragma: no cover - a job always has a status
        status.finished_at = now_iso()
        if job is not None and job.cancelled:
            status.status = "cancelled"
        elif code == 0:
            status.status = "done"
            status.episodes_done = status.episodes_total
        else:
            status.status = "failed"
            status.error = message

    def stop(self) -> bool | None:
        """Stop the running evaluation. ``False`` when none runs, ``None`` while it is dying."""
        job = self._job
        if job is None or not job.running:
            return False
        return True if job.cancel() else None
