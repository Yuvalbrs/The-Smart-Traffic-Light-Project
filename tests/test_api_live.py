"""T-05-01 - WebSocket delivery and the single-session lock.

Still no SUMO: the session lock is exercised with a stub standing in for
:class:`~src.api.live.LiveSession`, because what is under test is the locking contract
(libsumo is single-instance, so a second concurrent episode must be refused), not the
simulation itself.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import live as live_mod
from src.api.hub import MAX_QUEUE
from src.api.live import SessionBusyError, SessionManager
from src.api.server import create_app


class _StubSession:
    """Stands in for LiveSession: never touches SUMO, reports itself as alive until stopped."""

    def __init__(self, **kwargs) -> None:
        self.controller = kwargs.get("controller", "webster")
        self.scenario = kwargs.get("scenario", "SCN-05")
        self.seed = kwargs.get("seed", 7000)
        self._alive = False
        self.started = False
        self.stopped = False

        class _Status:
            def as_dict(_self) -> dict:
                return {"run_id": "stub", "controller": self.controller, "state": "running"}

        self.status = _Status()

    def start(self) -> None:
        self.started = True
        self._alive = True

    def stop(self, timeout: float = 20.0) -> None:
        self.stopped = True
        self._alive = False

    @property
    def alive(self) -> bool:
        return self._alive


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(live_mod, "LiveSession", _StubSession)
    app = create_app(db_path=tmp_path / "none.db", trace_dirs=[tmp_path])
    with TestClient(app) as c:
        yield c


def test_dashboard_socket_greets_then_streams(client) -> None:
    """A client gets a hello (so it can check the schema) and then live frames."""
    app = client.app
    with client.websocket_connect("/ws/dashboard") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["channel"] == "dash"
        assert hello["schema_version"]

        app.state.dash_hub.publish({"type": "dashboard_frame", "seq": 1, "sim_time": 1.0})
        got = ws.receive_json()
        assert got["seq"] == 1


def test_unity_socket_relays_sim_frames_untouched(client) -> None:
    """Unity must receive exactly the tracer's envelope, so replay and live agree byte for byte."""
    app = client.app
    frame = {
        "schema_version": "1.1.0",
        "type": "sim_frame",
        "seq": 7,
        "sim_time": 7.0,
        "payload": {"vehicles": [{"id": "v0", "movement_id": "M3"}], "signal": {"phase_index": 2}},
    }
    with client.websocket_connect("/ws/unity") as ws:
        ws.receive_json()  # hello
        app.state.unity_hub.publish(frame)
        assert ws.receive_json() == frame


def test_health_counts_subscribers_and_drops(client) -> None:
    app = client.app
    with client.websocket_connect("/ws/dashboard") as ws:
        ws.receive_json()  # hello
        for i in range(MAX_QUEUE * 4):
            app.state.dash_hub.publish({"seq": i})
        health = client.get("/health").json()
        assert health["channels"]["dash"]["subscribers"] == 1
        assert health["channels"]["dash"]["published"] >= MAX_QUEUE * 4

    after = client.get("/health").json()
    assert after["channels"]["dash"]["subscribers"] == 0, "disconnect must unsubscribe"


def test_second_session_is_refused_while_one_runs(client) -> None:
    """libsumo holds a single global simulation - the API must say 409, not corrupt it."""
    first = client.post("/sessions", json={"controller": "webster", "scenario": "SCN-05"})
    assert first.status_code == 201

    second = client.post("/sessions", json={"controller": "dqn-plain", "scenario": "SCN-05"})
    assert second.status_code == 409
    assert "already running" in second.json()["detail"]
    assert "DELETE /sessions/current" in second.json()["detail"]


def test_stopping_frees_the_slot(client) -> None:
    assert client.post("/sessions", json={"controller": "webster"}).status_code == 201
    assert client.delete("/sessions/current").status_code == 204
    assert client.post("/sessions", json={"controller": "actuated"}).status_code == 201


def test_manager_raises_session_busy(monkeypatch) -> None:
    """The lock lives in the manager, not only in the route handler."""
    monkeypatch.setattr(live_mod, "LiveSession", _StubSession)
    from src.api.hub import Hub

    manager = SessionManager(Hub("unity"), Hub("dash"))
    manager.start(controller="webster", scenario="SCN-05", seed=7000)
    with pytest.raises(SessionBusyError):
        manager.start(controller="webster", scenario="SCN-05", seed=7000)

    assert manager.stop() is True
    assert manager.stop() is False, "nothing left to stop"


def test_unknown_controller_is_rejected_before_any_simulation() -> None:
    """Validation happens up front - a bad name must never reach the SUMO layer."""
    from src.api.live import LiveSession
    from src.api.hub import Hub

    with pytest.raises(ValueError, match="unknown controller"):
        LiveSession(
            controller="ppo",
            scenario="SCN-05",
            seed=7000,
            unity_hub=Hub("unity"),
            dash_hub=Hub("dash"),
        )
