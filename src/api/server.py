"""T-05-01 - The FastAPI hub: WebSocket live channels, session control, REST replay.

Routes::

    GET    /health                  liveness + loop lag + per-channel backpressure counters
    GET    /controllers             what can be driven, and on which scenarios
    POST   /sessions                start a live episode  (409 if one is already running)
    GET    /sessions/current        status of the running / last session
    DELETE /sessions/current        stop the running session
    WS     /ws/dashboard            1 Hz derived dashboard frames
    WS     /ws/unity                1 Hz raw sim_frame envelopes
    GET    /runs...                 replay endpoints (see :mod:`src.api.replay`)

No auth, single-machine deployment, both WebSocket channels at 1 Hz with client-side
interpolation - all per the locked architecture (``decisions.md`` 2026-06-18, F10 amendment).

Run it::

    LIBSUMO_AS_TRACI=1 .venv/Scripts/python -m uvicorn src.api.server:app --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.hub import MAX_QUEUE, Hub
from src.api.live import CONTROLLERS, SessionBusyError, SessionManager
from src.api.replay import router as replay_router
from src.api.wire import SCHEMA_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _REPO_ROOT / "data" / "traffic.db"
_TRACE_DIRS = [_REPO_ROOT / "data" / "live", _REPO_ROOT / "data" / "eval"]

SCENARIOS = ("SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05")


class StartSession(BaseModel):
    """Body of ``POST /sessions``."""

    controller: str = Field(..., description="one of GET /controllers")
    scenario: str = Field("SCN-05", description="scenario id, e.g. SCN-05")
    seed: int = Field(7000, description="SUMO vehicle seed")
    episode_length_s: int = Field(3600, ge=60, le=3600, description="seconds of simulated time")
    trace: bool = Field(True, description="write the JSONL trace + a provenanced run row")
    speed: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="simulated seconds per wall second; 0 = as fast as possible, 1 = real time",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Bind the hubs to the running event loop and open the results database."""
    loop = asyncio.get_running_loop()
    app.state.unity_hub.bind_loop(loop)
    app.state.dash_hub.bind_loop(loop)
    yield
    manager: SessionManager = app.state.sessions
    manager.stop()


def create_app(db_path: Path | None = None, trace_dirs: list[Path] | None = None) -> FastAPI:
    """Build the application.

    Parameters
    ----------
    db_path : Path, optional
        SQLite results database. Defaults to ``data/traffic.db``.
    trace_dirs : list of Path, optional
        Directories searched for JSONL traces when replaying. Defaults to ``data/live`` then
        ``data/eval``.
    """
    from src.db.engine import create_db_engine

    app = FastAPI(
        title="Smart Traffic Intersection - hub",
        version=SCHEMA_VERSION,
        lifespan=lifespan,
    )
    # Single-machine, no-auth deployment (project-rules.md, system-architecture-overview.md
    # "Deployment topology"): the React dev server runs on its own port (5173) from the FastAPI
    # hub (8000), so the browser enforces CORS on every REST call unless the hub allows it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.unity_hub = Hub("unity")
    app.state.dash_hub = Hub("dash")
    app.state.sessions = SessionManager(app.state.unity_hub, app.state.dash_hub)
    app.state.trace_dirs = list(trace_dirs) if trace_dirs is not None else list(_TRACE_DIRS)
    path = Path(db_path) if db_path is not None else _DB_PATH
    # Create the engine unconditionally: SQLite makes the file on first write, and gating
    # on path.exists() at import time permanently 503'd every replay endpoint on a fresh
    # deployment even after live runs had populated the database. One engine per process,
    # shared with LiveSession._persist so sessions do not each leak their own pool.
    path.parent.mkdir(parents=True, exist_ok=True)
    app.state.engine = create_db_engine(path)
    app.state.db_path = path
    app.include_router(replay_router)

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness plus the two numbers that matter under load: loop lag and dropped frames."""
        session = app.state.sessions.current
        return {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "database": str(path) if app.state.engine is not None else None,
            "session": session.status.as_dict() if session is not None else None,
            "channels": {
                hub.name: {
                    "subscribers": hub.subscriber_count,
                    "published": hub.published,
                    "dropped": hub.dropped_total,
                    "queue_maxsize": MAX_QUEUE,
                }
                for hub in (app.state.dash_hub, app.state.unity_hub)
            },
        }

    @app.get("/controllers")
    def controllers() -> dict[str, Any]:
        """What can be driven live, and the scenarios available."""
        return {
            "controllers": list(CONTROLLERS),
            "scenarios": list(SCENARIOS),
            "default": {"controller": "sel/plain", "scenario": "SCN-05", "seed": 7000},
            "note": (
                "sel/plain is the shipped product (episode-level selector over plain DQN with a "
                "Webster fallback). Running the same scenario+seed under two controllers is the "
                "comparison demo."
            ),
        }

    @app.post("/sessions", status_code=201)
    def start_session(body: StartSession) -> dict[str, Any]:
        """Start one live episode. 409 while another is running (libsumo is single-instance)."""
        if body.controller not in CONTROLLERS:
            raise HTTPException(status_code=422, detail="unknown controller " + body.controller)
        if body.scenario not in SCENARIOS:
            raise HTTPException(status_code=422, detail="unknown scenario " + body.scenario)
        try:
            session = app.state.sessions.start(
                controller=body.controller,
                scenario=body.scenario,
                seed=body.seed,
                episode_length_s=body.episode_length_s,
                trace=body.trace,
                speed=body.speed,
            )
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return session.status.as_dict()

    @app.get("/sessions/current")
    def current_session() -> dict[str, Any]:
        """Status of the running (or most recent) session."""
        session = app.state.sessions.current
        if session is None:
            raise HTTPException(status_code=404, detail="no session has been started")
        return session.status.as_dict()

    @app.delete("/sessions/current")
    def stop_session(response: Response) -> dict[str, Any] | None:
        """Stop the running session.

        204 when the worker actually stopped; 202 when the stop was accepted but the
        worker is still unwinding (it can be inside a long native SUMO call, and Python
        threads cannot be killed). Reporting 204 unconditionally, as this did before,
        told the operator the simulation had stopped when it had not.
        """
        outcome = app.state.sessions.stop()
        if outcome is False:
            raise HTTPException(status_code=404, detail="no session is running")
        if outcome is None:
            response.status_code = 202
            return {
                "status": "stopping",
                "detail": "stop requested; the worker has not finished yet",
            }
        response.status_code = 204
        return None

    async def _pump(websocket: WebSocket, hub: Hub, name: str) -> None:
        """Relay one hub's frames to one client until it disconnects."""
        await websocket.accept()
        sub = hub.subscribe(name)
        try:
            await websocket.send_json(
                {"schema_version": SCHEMA_VERSION, "type": "hello", "channel": hub.name}
            )
            while True:
                frame = await sub.get()
                await websocket.send_json(frame)
        except WebSocketDisconnect:
            pass
        except RuntimeError:  # pragma: no cover - socket torn down mid-send
            pass
        finally:
            hub.unsubscribe(sub)

    @app.websocket("/ws/dashboard")
    async def ws_dashboard(websocket: WebSocket) -> None:
        """1 Hz derived dashboard frames."""
        await _pump(websocket, app.state.dash_hub, "dashboard")

    @app.websocket("/ws/unity")
    async def ws_unity(websocket: WebSocket) -> None:
        """1 Hz raw ``sim_frame`` envelopes - the same frames the JSONL trace records."""
        await _pump(websocket, app.state.unity_hub, "unity")

    return app


app = create_app()
