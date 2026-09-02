"""Long-running child processes driven from the API, with their progress streamed to clients.

Both things the application runs on demand - training a controller and evaluating one - are the
same shape: spawn one of this project's existing CLI scripts, read its stdout line by line to
learn how far it has got, publish that to a WebSocket channel, and end in a terminal state.

They are child processes rather than in-process calls for two reasons that apply to both:

* **libsumo is single-instance per process.** A live episode already runs inside the hub's own
  process, so doing simulation work in-process would mean the two could never coexist.
* **The scripts are already tested and provenanced.** ``train_dqn`` and ``eval_runner`` write the
  config, the version chain and the database rows that the results depend on. Driving the CLI
  means the thing the application demonstrates is the thing that produced the numbers, instead of
  a second implementation that can drift from it.

This module holds what is common; :mod:`src.api.training` and :mod:`src.api.evaluation` supply the
command to run and the parser for its progress lines.
"""

from __future__ import annotations

import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _Publisher(Protocol):
    """The slice of :class:`src.api.hub.Hub` a job needs."""

    def publish(self, frame: dict[str, Any]) -> None: ...


class JobBusyError(RuntimeError):
    """Raised when a job is requested while one of the same kind is already running."""


def now_iso() -> str:
    """UTC timestamp, seconds resolution - the format every job status uses."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def spawn(cmd: list[str]) -> subprocess.Popen[str]:
    """Start one child in the repo root with the project's simulation backend selected.

    stdout and stderr are merged so a traceback arrives in the same stream as the progress and can
    be reported back to the user, rather than filling an undrained pipe - which is also how a child
    silently blocks once the buffer fills.
    """
    env = dict(os.environ)
    env["LIBSUMO_AS_TRACI"] = "1"  # the backend every other entry point uses
    env["PYTHONUNBUFFERED"] = "1"  # progress must arrive per line, not per 8 KB of pipe buffer
    return subprocess.Popen(
        cmd, cwd=str(_REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )


def repo_relative(path: Path) -> str:
    """``path`` relative to the repo when it is inside it, absolute otherwise.

    Short and portable in the UI for the normal case, without raising for a directory configured
    outside the repository - which would otherwise take down whatever was rendering it.
    """
    return str(path.relative_to(_REPO_ROOT) if path.is_relative_to(_REPO_ROOT) else path)


class ProcessJob:
    """One child process, the thread draining it, and how it ended.

    Parameters
    ----------
    proc : subprocess.Popen
        Already started, with merged text stdout.
    hub : Hub
        Channel the status frames are published to.
    on_line : callable
        Called for each output line. Returns ``True`` when the line changed the status and a frame
        should be published - so a job publishes once per unit of progress, not once per log line.
    status_frame : callable
        Builds the wire frame for the current status, called under the job's lock.
    on_finish : callable
        Called with the exit code once the child is gone, to set the terminal state.
    """

    def __init__(
        self,
        proc: subprocess.Popen[str],
        hub: _Publisher,
        *,
        on_line: Callable[[str], bool],
        status_frame: Callable[[], dict[str, Any]],
        on_finish: Callable[[int, str], None],
        name: str = "job",
    ) -> None:
        self._proc = proc
        self._hub = hub
        self._on_line = on_line
        self._status_frame = status_frame
        self._on_finish = on_finish
        self._tail: list[str] = []
        self.lock = threading.Lock()
        self.cancelled = False
        self._thread = threading.Thread(target=self._pump, name=name, daemon=True)
        self._thread.start()

    @property
    def running(self) -> bool:
        return self._proc.poll() is None

    def publish(self) -> None:
        self._hub.publish(self._status_frame())

    def _pump(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.rstrip("\n")
            with self.lock:
                self._tail.append(line)
                del self._tail[:-40]  # enough to explain a failure, bounded for a long run
            if self._on_line(line):
                self.publish()
        code = self._proc.wait()
        self._on_finish(code, self.failure_message(code))
        self.publish()

    def failure_message(self, code: int) -> str:
        """A message built from the child's own last words, so it says what actually went wrong."""
        with self.lock:
            tail = [t for t in self._tail if t.strip()][-6:]
        return f"exited with code {code}: {' | '.join(tail) if tail else 'no output'}"

    def cancel(self) -> bool:
        """Ask the child to stop; ``True`` once it is actually gone."""
        if not self.running:
            return True
        with self.lock:
            self.cancelled = True
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
            return True
        except subprocess.TimeoutExpired:
            # SUMO can sit inside a long native call. terminate is a request; kill is not.
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
                return True
            except subprocess.TimeoutExpired:  # pragma: no cover - the OS is not cooperating
                return False
