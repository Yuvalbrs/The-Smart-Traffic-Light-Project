"""T-05-01 - The FastAPI hub: WebSocket live channels, session control, REST replay.

Routes::

    GET    /health                  liveness + loop lag + per-channel backpressure counters
    GET    /controllers             what can be driven, and on which scenarios
    GET    /models                  every trained model on disk (see :mod:`src.api.catalog`)
    GET    /comparison              per-controller KPIs on one scenario, from the database
    POST   /training                start a training job (409 if one is already running)
    GET    /training/current        progress of the running / last training job
    DELETE /training/current        cancel the running training job
    WS     /ws/training             live training progress frames
    POST   /evaluation              evaluate one user-trained model against the baselines
    GET    /evaluation/current      progress of the running / last evaluation
    DELETE /evaluation/current      cancel the running evaluation
    WS     /ws/evaluation           live evaluation progress frames
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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.api.catalog import router as catalog_router
from src.api.evaluation import MAX_UI_SEEDS, EvaluationManager
from src.api.hub import MAX_QUEUE, Hub
from src.api.live import CONTROLLERS, SessionBusyError, SessionManager
from src.api.replay import router as replay_router
from src.api.jobs import JobBusyError
from src.api.training import MAX_UI_EPISODES, VARIANTS, TrainingBusyError, TrainingManager
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


class StartTraining(BaseModel):
    """Body of ``POST /training``."""

    variant: str = Field("plain", description="plain | hybrid | random-lstm")
    seed: int = Field(42, ge=0, description="training seed; hybrid loads THIS seed's forecaster")
    episodes: int = Field(
        30, ge=1, le=MAX_UI_EPISODES,
        description=f"episodes to train (1-{MAX_UI_EPISODES}); 30 is a demo, 300 the full protocol",
    )
    episode_length_s: int | None = Field(
        None, ge=60, le=3600, description="shorten each episode for a faster demo run"
    )
    label: str | None = Field(None, max_length=80, description="free-text label for this run")


class StartEvaluation(BaseModel):
    """Body of ``POST /evaluation``."""

    model_id: str = Field(..., description="a run-directory name from GET /models")
    scenario: str = Field("SCN-05", description="scenario to evaluate on")
    seeds: int = Field(
        5, ge=1, le=MAX_UI_SEEDS,
        description=f"held-out eval seeds to use (1-{MAX_UI_SEEDS}); the campaign uses 15",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Bind the hubs to the running event loop and open the results database."""
    loop = asyncio.get_running_loop()
    app.state.unity_hub.bind_loop(loop)
    app.state.dash_hub.bind_loop(loop)
    app.state.training_hub.bind_loop(loop)
    app.state.evaluation_hub.bind_loop(loop)
    yield
    manager: SessionManager = app.state.sessions
    manager.stop()
    # A training child process outlives its parent unless it is asked to stop, which would leave
    # an orphan holding a SUMO instance and a run directory after the server is gone.
    app.state.training.stop()
    app.state.evaluation.stop()


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
    app.state.training_hub = Hub("training")
    app.state.evaluation_hub = Hub("evaluation")
    app.state.sessions = SessionManager(app.state.unity_hub, app.state.dash_hub)
    app.state.training = TrainingManager(app.state.training_hub)
    app.state.evaluation = EvaluationManager(app.state.evaluation_hub)
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
    app.include_router(catalog_router)

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
                for hub in (app.state.dash_hub, app.state.unity_hub,
                            app.state.training_hub, app.state.evaluation_hub)
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


    @app.post("/training", status_code=201)
    def start_training(body: StartTraining) -> dict[str, Any]:
        """Start one training job as a child process. 409 while another is running.

        A live episode may run at the same time: the job is a separate process with its own
        libsumo, so the single-instance rule that serializes live sessions does not apply here.
        Both will simply be slower while they share the machine.
        """
        if body.variant not in VARIANTS:
            raise HTTPException(
                status_code=422,
                detail=f"unknown variant {body.variant}; expected one of {list(VARIANTS)}",
            )
        try:
            status = app.state.training.start(
                variant=body.variant,
                seed=body.seed,
                episodes=body.episodes,
                episode_length_s=body.episode_length_s,
                label=body.label,
            )
        except TrainingBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, KeyError) as exc:
            # KeyError is the per-seed forecaster pin refusing an unpinned seed (A6.4) - a 422,
            # not a 500: the request named a seed the project has no forecaster for.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return status.as_dict()

    @app.get("/training/current")
    def current_training() -> dict[str, Any]:
        """Progress of the running (or most recent) training job."""
        job = app.state.training.current
        if job is None:
            raise HTTPException(status_code=404, detail="no training job has been started")
        return job.status.as_dict()

    @app.delete("/training/current")
    def stop_training(response: Response) -> dict[str, Any] | None:
        """Cancel the running training job.

        Mirrors DELETE /sessions/current: 204 once the child is gone, 202 while it is still
        unwinding inside a native SUMO call.
        """
        outcome = app.state.training.stop()
        if outcome is False:
            raise HTTPException(status_code=404, detail="no training job is running")
        if outcome is None:
            response.status_code = 202
            return {"status": "stopping", "detail": "cancel requested; the job has not exited yet"}
        response.status_code = 204
        return None


    @app.post("/evaluation", status_code=201)
    def start_evaluation(body: StartEvaluation) -> dict[str, Any]:
        """Evaluate one user-trained model against the baselines on shared seeds.

        This is what lets a model trained in the app reach the comparison at all: the comparison
        reads evaluation rows, and training alone produces none.
        """
        if body.scenario not in SCENARIOS:
            raise HTTPException(status_code=422, detail="unknown scenario " + body.scenario)
        try:
            status = app.state.evaluation.start(
                model_id=body.model_id, scenario=body.scenario, seeds=body.seeds
            )
        except JobBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return status.as_dict()

    @app.get("/evaluation/current")
    def current_evaluation() -> dict[str, Any]:
        """Progress of the running (or most recent) evaluation."""
        status = app.state.evaluation.current
        if status is None:
            raise HTTPException(status_code=404, detail="no evaluation has been started")
        return status.as_dict()

    @app.delete("/evaluation/current")
    def stop_evaluation(response: Response) -> dict[str, Any] | None:
        """Cancel the running evaluation (204 gone, 202 still unwinding)."""
        outcome = app.state.evaluation.stop()
        if outcome is False:
            raise HTTPException(status_code=404, detail="no evaluation is running")
        if outcome is None:
            response.status_code = 202
            return {"status": "stopping", "detail": "cancel requested; the job has not exited yet"}
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


    @app.websocket("/ws/evaluation")
    async def ws_evaluation(websocket: WebSocket) -> None:
        """Evaluation progress frames, one per completed episode."""
        await _pump(websocket, app.state.evaluation_hub, "evaluation")

    @app.websocket("/ws/training")
    async def ws_training(websocket: WebSocket) -> None:
        """Training progress frames, one per completed episode."""
        await _pump(websocket, app.state.training_hub, "training")


    # The built single-page app is mounted LAST and only if it exists. Mounting at "/" catches
    # every path not already claimed above, so registering it earlier would shadow the whole API;
    # and a missing dist/ must leave the API perfectly usable, because that is the state during
    # frontend development, when Vite serves the SPA on its own port instead.
    dist = _REPO_ROOT / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="app")
    app.state.spa_dir = dist if dist.is_dir() else None

    return app


app = create_app()
