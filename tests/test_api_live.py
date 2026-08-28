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
        self.speed = kwargs.get("speed", 0.0)
        self._alive = False
        self.started = False
        self.stopped = False
        self.stop_hangs = False  # set True to simulate a worker that will not unwind

        class _Status:
            def as_dict(_self) -> dict:
                return {"run_id": "stub", "controller": self.controller, "state": "running"}

        self.status = _Status()

    def start(self) -> None:
        self.started = True
        self._alive = True

    def stop(self, timeout: float = 20.0) -> bool:
        """Mirror LiveSession.stop: returns whether the worker actually finished."""
        self.stopped = True
        if self.stop_hangs:  # simulate a worker stuck in a native SUMO call
            return False
        self._alive = False
        return True

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


def test_stop_reports_202_when_the_worker_has_not_unwound(client) -> None:
    """A stop that did not actually stop must not report 204.

    The worker can be inside a long native SUMO call rather than the interruptible
    pacing wait, and Python threads cannot be killed. Reporting a clean stop we did
    not achieve hides a still-running simulation from the operator
    (decisions.md 2026-08-28).
    """
    assert client.post("/sessions", json={"controller": "webster"}).status_code == 201
    current = client.app.state.sessions.current
    current.stop_hangs = True  # worker will not unwind within the join timeout

    resp = client.delete("/sessions/current")
    assert resp.status_code == 202
    assert resp.json()["status"] == "stopping"
    # the slot is NOT freed, because the old session still holds the TraCI connection
    assert client.post("/sessions", json={"controller": "actuated"}).status_code == 409


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


# --- real-time pacing (T-05-02 demo fix) ---------------------------------------------------


def _session(**kwargs):
    """A LiveSession that has not been started - safe to poke at without SUMO."""
    from src.api.hub import Hub
    from src.api.live import LiveSession

    kwargs.setdefault("controller", "webster")
    kwargs.setdefault("scenario", "SCN-04")
    kwargs.setdefault("seed", 7000)
    return LiveSession(unity_hub=Hub("unity"), dash_hub=Hub("dash"), **kwargs)


def test_unpaced_is_the_default_and_never_sleeps() -> None:
    """speed=0 must keep the original as-fast-as-possible behaviour for tests and corpus runs."""
    import time

    session = _session()
    assert session.speed == 0.0

    session.status.frames = 500
    started = time.perf_counter()
    for _ in range(50):
        session._pace()
    assert time.perf_counter() - started < 0.05, "unpaced sessions must not throttle"


def test_pacing_throttles_to_the_requested_speed() -> None:
    """At speed=20, ten simulated seconds must take about half a wall-clock second."""
    import time

    session = _session(speed=20.0)
    session.status.frames = 1
    session._pace()  # first frame only takes the origin

    started = time.perf_counter()
    for frame in range(2, 12):
        session.status.frames = frame
        session._pace()
    elapsed = time.perf_counter() - started

    assert 0.3 < elapsed < 0.9, f"expected ~0.5s of pacing for 10 frames at 20x, got {elapsed:.3f}s"


def test_pacing_wakes_immediately_when_the_session_is_stopped() -> None:
    """DELETE /sessions/current must not block for a whole frame interval mid-wait."""
    import time

    session = _session(speed=0.5)  # 2 wall-seconds per simulated second
    session.status.frames = 1
    session._pace()

    session._stop.set()
    session.status.frames = 2
    started = time.perf_counter()
    session._pace()
    assert time.perf_counter() - started < 0.5, "a stopped session must abandon the pacing wait"


def test_speed_reaches_the_session_manager(client) -> None:
    """The wire field must actually arrive at LiveSession, not be dropped in the route."""
    resp = client.post(
        "/sessions",
        json={"controller": "webster", "scenario": "SCN-04", "seed": 7000, "speed": 5.0},
    )
    assert resp.status_code == 201
    assert client.app.state.sessions.current.speed == 5.0


def test_speed_must_not_be_negative(client) -> None:
    assert client.post("/sessions", json={"controller": "webster", "speed": -1.0}).status_code == 422


# --- live-run KPIs (replay browser fix) ----------------------------------------------------


def test_no_kpis_when_the_episode_was_stopped_early(tmp_path) -> None:
    """A truncated trip-info file would yield real-looking numbers that mean nothing."""
    session = _session()
    session.status.state = "stopped"
    session._tripinfo_path = tmp_path / "t.xml"
    session._tripinfo_path.write_text("<tripinfos/>", encoding="utf-8")
    session.status.trace_path = str(tmp_path / "t.jsonl")

    assert session._compute_kpis() is None


def test_no_kpis_when_tripinfo_was_never_written(tmp_path) -> None:
    """trace=False sessions have no trip-info, and extract_kpis cannot run without one."""
    session = _session(trace=False)
    session.status.state = "finished"
    session.status.trace_path = str(tmp_path / "t.jsonl")

    assert session._tripinfo_path is None
    assert session._compute_kpis() is None
