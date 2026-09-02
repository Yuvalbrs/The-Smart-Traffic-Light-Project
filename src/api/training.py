"""In-app training jobs: run ``scripts.train_dqn`` as a child process and stream its progress.

The application has to be able to train each model variant on demand and show the reward curve
moving while it happens. Two constraints shape the design:

* **libsumo is single-instance per process.** A live episode is already driven inside the hub's
  own process, so training in-process would mean the two could never coexist and a stray import
  order could deadlock the demo. A child process gets its own libsumo and the question disappears.
* **The training loop is already written, tested and provenanced.** ``scripts/train_dqn.py`` writes
  ``config.yaml`` with the git SHA and the per-seed forecaster id (A6.4), checkpoints on a
  schedule, and logs one line per episode. Re-implementing any of that inside the API would create
  a second training path that could drift from the one the results came from. So the API drives the
  CLI rather than the library: the thing being demonstrated is the thing that produced the numbers.

Progress arrives by parsing the child's stdout, which is the same line the matrix logs::

      ep  12/300  SCN-01  reward=       -38.9  eps=0.994  buffer=4320

Parsing stdout is a contract with a sibling module, so :data:`_EPISODE_RE` is covered by a test that
feeds it a real line - if the log format ever changes, that test fails rather than the UI silently
flat-lining at 0%.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.api.hub import Hub
from src.api.wire import SCHEMA_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNS_DIR = _REPO_ROOT / "runs"

TRAINING_MSG = "training_frame"
VARIANTS = ("plain", "hybrid", "random-lstm")

#: Episodes a UI run is capped at. The full protocol is 300 episodes (~13 min); the app exists to
#: SHOW training, not to reproduce the campaign, and an unbounded box on a web form is an easy way
#: to start a two-hour job by accident during a demo.
MAX_UI_EPISODES = 300

# " ep  12/300  SCN-01  reward=       -38.9  eps=0.994  buffer=4320"
# reward is formatted "{:12,.1f}", so it can carry thousands separators and leading spaces.
_EPISODE_RE = re.compile(
    r"^\s*ep\s+(?P<ep>\d+)/(?P<total>\d+)\s+(?P<scenario>\S+)\s+"
    r"reward=\s*(?P<reward>-?[\d,]+\.?\d*)\s+eps=(?P<eps>[\d.]+)"
)


class TrainingBusyError(RuntimeError):
    """Raised when a training job is asked for while one is already running."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TrainingStatus:
    """Everything the UI needs about one job. Serialized as-is onto the wire."""

    job_id: str
    variant: str
    seed: int
    episodes: int
    label: str | None
    run_dir: str
    #: Scenarios whose demand drove this run; None means train_dqn's own default rotation.
    #: Reported so "this model was trained on real measured traffic" is a fact the UI reads
    #: back off the job rather than a claim someone types into the label.
    train_scenarios: list[str] | None = None
    status: str = "running"  # running | done | failed | cancelled
    episodes_done: int = 0
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    error: str | None = None
    curve: list[dict[str, float]] = field(default_factory=list)

    @property
    def pct(self) -> float:
        return round(100.0 * self.episodes_done / self.episodes, 1) if self.episodes else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "variant": self.variant,
            "seed": self.seed,
            "episodes": self.episodes,
            "label": self.label,
            "train_scenarios": list(self.train_scenarios) if self.train_scenarios else None,
            "episodes_done": self.episodes_done,
            "pct": self.pct,
            "curve": list(self.curve),
            "run_dir": self.run_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    def as_frame(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "type": TRAINING_MSG, **self.as_dict()}


def _build_command(variant: str, seed: int, episodes: int, episode_length_s: int | None,
                   run_dir: Path, train_scenarios: tuple[str, ...] | None = None) -> list[str]:
    """The ``train_dqn`` invocation for one variant.

    The variant flags mirror ``scripts/train_dqn.py``'s own argument handling exactly. ``hybrid``
    resolves the checkpoint through ``official_lstm_checked(seed)`` - the guarded, per-seed
    accessor (A6.4) - so a UI run cannot quietly train against a stale pin or another seed's
    forecaster the way a hard-coded path would.

    ``train_scenarios`` names the scenarios whose demand drives the episodes. ``train_dqn`` has
    accepted ``--train-scenarios`` all along; this builder simply never passed it, so every in-app
    run was pinned to the default SCN-01/02/03 rotation and the measured-demand scenario the
    ingest pipeline produces (SCN-R1) could not be reached from the application at all. ``None``
    keeps train_dqn's own default rather than restating it here - two copies of a default is how
    they drift.
    """
    cmd = [
        sys.executable, "-u", "-m", "scripts.train_dqn",
        "--seed", str(seed),
        "--variant", variant,
        "--episodes", str(episodes),
        "--run-dir", str(run_dir),
        # A demo run is short, so the default every-25 validation and every-50 checkpointing would
        # never fire and the run would end with no final checkpoint to compare against.
        "--validation-every", str(max(5, episodes // 3)),
        "--checkpoint-every", str(max(5, episodes // 3)),
        "--no-log-steps",  # per-step CSV is worth ~50 MB on a long run and the UI never reads it
    ]
    if episode_length_s:
        cmd += ["--episode-length", str(episode_length_s)]
    if train_scenarios:
        cmd += ["--train-scenarios", *train_scenarios]
    if variant == "hybrid":
        from src.provenance.official import official_lstm_checked

        cmd += ["--forecast-ckpt", str(official_lstm_checked(seed))]
    elif variant == "random-lstm":
        cmd += ["--random-lstm"]
    return cmd


class TrainingJob:
    """One running child process, its parsed progress, and the thread that reads it."""

    def __init__(self, status: TrainingStatus, proc: subprocess.Popen[str], hub: Hub) -> None:
        self.status = status
        self._proc = proc
        self._hub = hub
        self._tail: list[str] = []  # last stderr/stdout lines, for the failure message
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._pump, name=f"train-{status.job_id}", daemon=True)
        self._thread.start()

    @property
    def running(self) -> bool:
        return self._proc.poll() is None

    def _publish(self) -> None:
        self._hub.publish(self.status.as_frame())

    def _pump(self) -> None:
        """Read the child's output to EOF, then record how it exited.

        stdout and stderr are merged, so a traceback lands in the same stream as the progress and
        ends up in ``error`` instead of vanishing into a pipe nobody drains - an undrained stderr
        pipe is also how a child silently blocks once the buffer fills.
        """
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.rstrip("\n")
            with self._lock:
                self._tail.append(line)
                del self._tail[:-40]
            match = _EPISODE_RE.match(line)
            if not match:
                continue
            with self._lock:
                self.status.episodes_done = int(match["ep"])
                self.status.curve.append({
                    "ep": int(match["ep"]),
                    "reward": float(match["reward"].replace(",", "")),
                    "epsilon": float(match["eps"]),
                })
            self._publish()

        code = self._proc.wait()
        with self._lock:
            self.status.finished_at = _now()
            if self.status.status == "cancelled":
                pass  # a cancel already set the terminal state; a non-zero code is expected
            elif code == 0:
                self.status.status = "done"
                self.status.episodes_done = self.status.episodes
            else:
                self.status.status = "failed"
                self.status.error = self._failure_message(code)
        self._publish()

    def _failure_message(self, code: int) -> str:
        """A message that says what to do, built from the child's own last words."""
        tail = [t for t in self._tail if t.strip()][-6:]
        detail = " | ".join(tail) if tail else "no output"
        return f"training exited with code {code}: {detail}"

    def cancel(self) -> bool:
        """Ask the child to stop. Returns True once it is actually gone."""
        if not self.running:
            return True
        with self._lock:
            self.status.status = "cancelled"
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
            return True
        except subprocess.TimeoutExpired:
            # SUMO can sit inside a long native call; terminate is a request, kill is not.
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
                return True
            except subprocess.TimeoutExpired:  # pragma: no cover - the OS is not cooperating
                return False


class TrainingManager:
    """One training job at a time, and the record of the most recent one.

    Serialized deliberately: two concurrent 300-episode runs on a laptop that is also serving a
    live episode would make every part of the demo stutter, and the UI has one progress view.
    """

    def __init__(self, hub: Hub, runs_dir: Path | None = None) -> None:
        self._hub = hub
        self._runs_dir = runs_dir or _RUNS_DIR
        self._job: TrainingJob | None = None
        self._counter = 0
        self._lock = threading.Lock()

    @property
    def current(self) -> TrainingJob | None:
        return self._job

    @property
    def busy(self) -> bool:
        return self._job is not None and self._job.running

    def start(self, *, variant: str, seed: int, episodes: int,
              episode_length_s: int | None = None, label: str | None = None,
              train_scenarios: tuple[str, ...] | None = None) -> TrainingStatus:
        """Launch one job. Raises :class:`TrainingBusyError` if one is already running."""
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")
        if not 1 <= episodes <= MAX_UI_EPISODES:
            raise ValueError(f"episodes must be between 1 and {MAX_UI_EPISODES}, got {episodes}")
        with self._lock:
            if self.busy:
                raise TrainingBusyError(
                    f"training job {self._job.status.job_id} is still running "  # type: ignore[union-attr]
                    f"({self._job.status.episodes_done}/{self._job.status.episodes} episodes)"  # type: ignore[union-attr]
                )
            self._counter += 1
            job_id = f"t-{self._counter}"

        run_dir = self._runs_dir / f"ui_{variant}_seed{seed}_{job_id}"
        cmd = _build_command(variant, seed, episodes, episode_length_s, run_dir, train_scenarios)

        env = dict(os.environ)
        env["LIBSUMO_AS_TRACI"] = "1"  # the training backend, as in every other entry point
        env["PYTHONUNBUFFERED"] = "1"  # progress must arrive per line, not per 8 KB pipe buffer
        proc = subprocess.Popen(
            cmd, cwd=str(_REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        status = TrainingStatus(
            job_id=job_id, variant=variant, seed=seed, episodes=episodes, label=label,
            run_dir=str(run_dir.relative_to(_REPO_ROOT)) if run_dir.is_relative_to(_REPO_ROOT)
            else str(run_dir),
            train_scenarios=list(train_scenarios) if train_scenarios else None,
        )
        with self._lock:
            self._job = TrainingJob(status, proc, self._hub)
        return status

    def stop(self) -> bool | None:
        """Stop the running job. ``False`` when none is running, ``None`` when it is still dying."""
        job = self._job
        if job is None or not job.running:
            return False
        return True if job.cancel() else None
