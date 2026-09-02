"""The hub can open the Unity 3-D window itself, and says so honestly when it cannot.

These endpoints shipped without tests and nothing called them, so "one launch" still meant
finding an .exe by hand. The dashboard now has an "open 3-D view" button wired to them, which
makes their failure modes user-visible: a machine with no build must get an explanation naming
the build command, not a stack trace or a dead button.

Nothing here starts a real process. ``ViewerManager`` takes its executable path as a constructor
argument precisely so a test can point it at a file that does not exist, and the one case that
must observe a launch fakes ``subprocess.Popen``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.api.viewer import BUILD_HINT, VIEWER_EXE, ViewerManager


class _FakeProc:
    """Stands in for a launched player. ``poll()`` is None while 'running'."""

    def __init__(self, *_a, **_kw) -> None:
        self.terminated = False
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def wait(self, timeout=None):  # noqa: ARG002 - signature match
        return 0


@pytest.fixture()
def missing_exe(tmp_path) -> Path:
    return tmp_path / "not-built" / "SmartTrafficViz.exe"


@pytest.fixture()
def built_exe(tmp_path) -> Path:
    exe = tmp_path / "SmartTrafficViz.exe"
    exe.write_bytes(b"MZ")  # contents are irrelevant; only is_file() is consulted
    return exe


# ------------------------------------------------------------------------------------------
# Not built - the state a fresh clone is in
# ------------------------------------------------------------------------------------------


def test_status_reports_unavailable_and_names_the_build_command(missing_exe) -> None:
    status = ViewerManager(missing_exe).status()
    assert status["available"] is False
    assert status["running"] is False
    assert status["hint"] == BUILD_HINT
    assert "BuildWindows" in status["hint"], "the hint must name the command that fixes this"


def test_starting_an_unbuilt_viewer_raises_rather_than_launching_nothing(missing_exe) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        ViewerManager(missing_exe).start()
    assert "BuildWindows" in str(exc.value)


def test_the_editor_route_is_always_offered(missing_exe) -> None:
    """Even with no player build there is a way to get the same view; say so."""
    status = ViewerManager(missing_exe).status()
    assert "Play" in status["editor_hint"]


# ------------------------------------------------------------------------------------------
# Built - launching, idempotence, stopping
# ------------------------------------------------------------------------------------------


def test_start_launches_once_and_is_idempotent(built_exe, monkeypatch) -> None:
    """A second click while the window is open must not spawn a second player."""
    spawned: list[list[str]] = []

    def fake_popen(cmd, **_kw):
        spawned.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    mgr = ViewerManager(built_exe)

    assert mgr.start()["running"] is True
    assert mgr.start()["running"] is True
    assert len(spawned) == 1, "the second start spawned another process"
    assert spawned[0] == [str(built_exe)]


def test_stop_is_false_when_nothing_is_running(built_exe) -> None:
    assert ViewerManager(built_exe).stop() is False


def test_stop_terminates_a_running_viewer(built_exe, monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **_kw: _FakeProc())
    mgr = ViewerManager(built_exe)
    mgr.start()
    assert mgr.stop() is True
    assert mgr.running is False


# ------------------------------------------------------------------------------------------
# The executable path is fixed, never taken from a request
# ------------------------------------------------------------------------------------------


def test_default_executable_is_the_build_scripts_output_path() -> None:
    """A path from an HTTP body would make this endpoint a way to start any process."""
    assert VIEWER_EXE.name == "SmartTrafficViz.exe"
    assert VIEWER_EXE.parent.name == "Build"


# ------------------------------------------------------------------------------------------
# HTTP surface - the shape the dashboard button reads
# ------------------------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(db_path=tmp_path / "empty.db"))


def test_get_viewer_returns_every_field_the_button_needs(client) -> None:
    body = client.get("/viewer").json()
    assert set(body) >= {"available", "running", "path", "hint", "editor_hint"}
    assert isinstance(body["available"], bool)


def test_stopping_a_viewer_that_is_not_running_404s(client) -> None:
    assert client.delete("/viewer").status_code == 404
