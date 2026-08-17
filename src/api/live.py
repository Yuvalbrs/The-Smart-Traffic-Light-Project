"""T-05-01 - The live bridge: one SUMO episode driven in a worker thread, streamed at 1 Hz.

Locked constraint: **SUMO is the single source of truth and is never blocked by clients**
(``decisions.md`` 2026-06-18). So the simulation runs on its own thread and pushes frames into
:class:`~src.api.hub.Hub` mailboxes that drop frames rather than apply backpressure upstream. A
stalled WebSocket client can only starve itself.

Only one session may exist at a time, by construction rather than by policy: ``libsumo`` runs
in-process and holds a single global simulation, so a second concurrent episode would corrupt
the first. :class:`SessionManager` enforces that with a lock and answers 409 while one is live.

Every live session is provenanced like an eval run (``project-rules.md`` rule 7): it writes its
own JSONL trace under ``data/live/`` and an ``experiment_run`` + ``episode`` row carrying
``(run_id, git_sha, data_version, lstm_version)``, so anything demonstrated live is immediately
replayable through the same REST endpoints as the recorded corpus. Live rows are written with
``mode="live"`` so they can never be mistaken for evaluation results.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Imported at module scope, on the MAIN thread, deliberately. When the first `import traci`
# happens inside the simulation worker thread instead, the libsumo backend probe fails
# ("DLL load failed while importing _libsumo") and traci silently falls back to the pure-python
# socket client - functional, but ~10x slower, which is the difference between a live demo that
# keeps up and one that crawls. Importing here forces the fast in-process backend to be resolved
# once, up front, under LIBSUMO_AS_TRACI=1.
import traci  # noqa: F401  (side-effecting import: selects the libsumo backend)

from src.api.hub import Hub
from src.api.wire import dashboard_frame

_REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_TRACE_DIR = _REPO_ROOT / "data" / "live"

# The controllers offerable from the API. sel/plain is the project's shipped product; the
# baselines are what it is judged against. Keys are the wire names clients send.
CONTROLLERS = (
    "sel/plain",
    "dqn-plain",
    "dqn-hybrid",
    "webster",
    "max_pressure",
    "actuated",
)


class SessionBusyError(RuntimeError):
    """Raised when a session is requested while another is already running."""


@dataclass
class SessionStatus:
    """Snapshot of the running (or last) session, as served by ``/sessions/current``."""

    run_id: str
    controller: str
    scenario: str
    seed: int
    state: str = "starting"  # starting | running | finished | failed | stopped
    sim_time: float = 0.0
    frames: int = 0
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    trace_path: str | None = None
    max_loop_lag_s: float = 0.0
    last_loop_lag_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {
            "run_id": self.run_id,
            "controller": self.controller,
            "scenario": self.scenario,
            "seed": self.seed,
            "state": self.state,
            "sim_time": self.sim_time,
            "frames": self.frames,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "trace_path": self.trace_path,
            "loop_lag_s": {"last": self.last_loop_lag_s, "max": self.max_loop_lag_s},
        }


def _running_kpis(env: Any, sim_time: float) -> tuple[float, float]:
    """Running (avg_wait_so_far, throughput_so_far) estimates for the dashboard channel.

    These are live indicators only. The project's KPIs come from
    :func:`src.metrics.kpi_extractor.extract_kpis` over SUMO trip-info after the episode - one
    implementation, no drift (``project-rules.md``). Waiting time here is the mean accumulated
    wait over vehicles currently in the network, which is why it is named ``*_so_far``.
    """
    ids = traci.vehicle.getIDList()
    if ids:
        total = sum(traci.vehicle.getAccumulatedWaitingTime(v) for v in ids)
        avg_wait = total / len(ids)
    else:
        avg_wait = 0.0
    hours = max(sim_time, 1.0) / 3600.0
    return float(avg_wait), float(env.arrived_count / hours)


class LiveSession:
    """One episode, driven on a worker thread, streamed to the two hubs.

    Parameters
    ----------
    controller : str
        One of :data:`CONTROLLERS`.
    scenario : str
        Scenario id, e.g. ``"SCN-05"``.
    seed : int
        SUMO vehicle seed for the episode.
    unity_hub, dash_hub : Hub
        Fan-out channels for the raw ``sim_frame`` and the derived dashboard frame.
    episode_length_s : int, optional
        Episode length; shorter values make a demo loop quickly.
    trace : bool, optional
        Write the JSONL trace + database rows (default ``True``).
    """

    def __init__(
        self,
        *,
        controller: str,
        scenario: str,
        seed: int,
        unity_hub: Hub,
        dash_hub: Hub,
        episode_length_s: int = 3600,
        trace: bool = True,
    ) -> None:
        if controller not in CONTROLLERS:
            raise ValueError(
                "unknown controller " + repr(controller) + "; expected one of " + str(CONTROLLERS)
            )
        from src.provenance.versions import new_run_id

        self.controller = controller
        self.scenario = scenario
        self.seed = seed
        self.episode_length_s = episode_length_s
        self._unity, self._dash = unity_hub, dash_hub
        self._trace = trace
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._env: Any = None
        self._last_tick = 0.0
        self.status = SessionStatus(
            run_id=new_run_id(), controller=controller, scenario=scenario, seed=seed
        )

    # --- lifecycle ---

    def start(self) -> None:
        """Launch the worker thread (returns immediately)."""
        self._thread = threading.Thread(target=self._run, name="sumo-live", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 20.0) -> None:
        """Ask the episode to end and wait for the worker to unwind."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        """Whether the worker thread is still running."""
        return self._thread is not None and self._thread.is_alive()

    # --- the worker ---

    def _on_frame(self, frame: dict[str, Any]) -> None:
        """Called by SUMOEnv once per simulated second, inside the stepping loop.

        Must not block: both hubs hand off to bounded queues and return. Also records loop lag -
        the wall-clock gap between consecutive simulated seconds - which ``/health`` reports as
        the honest measure of whether the sim is keeping up with real time.
        """
        now = time.perf_counter()
        if self._last_tick:
            lag = now - self._last_tick
            self.status.last_loop_lag_s = lag
            self.status.max_loop_lag_s = max(self.status.max_loop_lag_s, lag)
        self._last_tick = now

        self.status.frames += 1
        self.status.sim_time = float(frame.get("sim_time", 0.0))
        self._unity.publish(frame)

        env = self._env
        try:
            queues, _counts = env.movement_features()
            pressures = env.movement_pressures()
            avg_wait, throughput = _running_kpis(env, self.status.sim_time)
            signal = frame.get("payload", frame).get("signal", {})
            phase = int(signal.get("phase_index", 0))
            self._dash.publish(
                dashboard_frame(
                    sim_time=self.status.sim_time,
                    current_phase=phase,
                    last_action=int(getattr(self, "_last_action", phase)),
                    queue_lengths=np.asarray(queues).tolist(),
                    pressures=np.asarray(pressures).tolist(),
                    avg_wait_so_far=avg_wait,
                    throughput_so_far=throughput,
                    forecast_next_30s=getattr(self, "_forecast", None),
                    seq=int(frame.get("seq", self.status.frames)),
                    episode_id=int(frame.get("episode_id", 0)),
                )
            )
        except Exception as exc:  # a dashboard hiccup must never kill the simulation
            self.status.error = "dashboard frame failed: " + repr(exc)

    def _build(self) -> tuple[Any, Any]:
        """Construct the env + the policy for this session. Returns ``(env, algo)``."""
        from scripts.build_network import build_net
        from scripts.build_routes import write_routes
        from scripts.eval_runner import Algo, _baseline_algos, _load_agent
        from src.env.sumo_env import SUMOEnv
        from src.ml.dqn import OBS_DIM
        from src.ml.hybrid_wrapper import HybridStateWrapper
        from src.ml.supervisor import EpisodeLevelSelector
        from src.scenarios.config import SCENARIO_DIR, load_scenario

        build_net()
        scn = load_scenario(SCENARIO_DIR / ("scn_" + self.scenario.split("-")[1] + ".yaml"))
        baselines = {a.name: a for a in _baseline_algos(scn)}

        algo: Any
        if self.controller in ("webster", "max_pressure", "actuated"):
            algo = baselines[self.controller]
        elif self.controller in ("dqn-plain", "sel/plain"):
            ckpt = _REPO_ROOT / "runs" / "plain_seed42" / "checkpoints" / "ep299.pt"
            algo = Algo(
                self.controller,
                "dqn",
                agent=_load_agent(ckpt, OBS_DIM),
                variant="plain",
                train_seed=42,
                ckpt=str(ckpt),
            )
        else:  # dqn-hybrid
            from scripts.eval_runner import _OFFICIAL_LSTM, load_forecaster
            from src.ml.hybrid_wrapper import HYBRID_OBS_DIM

            ckpt = _REPO_ROOT / "runs" / "hybrid_seed42" / "checkpoints" / "ep299.pt"
            algo = Algo(
                self.controller,
                "dqn",
                agent=_load_agent(ckpt, HYBRID_OBS_DIM),
                forecaster=load_forecaster(str(_OFFICIAL_LSTM)),
                variant="hybrid",
                train_seed=42,
                lstm_version=_OFFICIAL_LSTM.name,
                ckpt=str(ckpt),
            )

        trace_path = None
        if self._trace:
            LIVE_TRACE_DIR.mkdir(parents=True, exist_ok=True)
            trace_path = LIVE_TRACE_DIR / (self.status.run_id + ".jsonl")
            self.status.trace_path = str(trace_path)

        env: Any = SUMOEnv(
            write_routes(scn, self.seed),
            episode_length_s=min(self.episode_length_s, scn.duration_s),
            sumo_seed=self.seed,
            signal_mode=algo.signal_mode,
            trace_path=trace_path,
            on_frame=self._on_frame,
        )
        if algo.forecaster is not None:
            env = HybridStateWrapper(env, algo.forecaster)

        if self.controller == "sel/plain":
            algo.controller = EpisodeLevelSelector(
                algo.agent, baselines["webster"].controller, threshold=150
            )
        self._env = env
        return env, algo

    def _run(self) -> None:
        """Thread body: drive the episode to completion (or until stopped)."""
        env = None
        try:
            env, algo = self._build()
            self.status.state = "running"
            obs, info = env.reset()
            if algo.controller is not None and hasattr(algo.controller, "reset"):
                algo.controller.reset(env)
            done = False
            while not done and not self._stop.is_set():
                mask = info["mask"]
                if algo.controller is not None:
                    action = int(algo.controller.select_action(obs, mask))
                elif algo.agent is not None:
                    action = int(algo.agent.act(obs, mask, epsilon=0.0))
                else:  # actuated: SUMO drives the lights, the action is ignored
                    action = 0
                self._last_action = action
                obs, _r, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
            self.status.state = "stopped" if self._stop.is_set() else "finished"
        except Exception as exc:  # surface it in /sessions/current rather than dying silently
            self.status.state = "failed"
            self.status.error = repr(exc)
        finally:
            self.status.finished_at = time.time()
            if env is not None:
                try:
                    env.close()
                except Exception:  # pragma: no cover - close is best-effort
                    pass
            self._env = None
            if self._trace and self.status.state in ("finished", "stopped"):
                try:
                    self._persist()
                except Exception as exc:  # provenance failure must not mask a good run
                    self.status.error = "persist failed: " + repr(exc)

    def _persist(self) -> None:
        """Record this live run in the results DB so it shows up in ``GET /runs``.

        Written with ``mode="live"`` and no KPI row: the confirmatory KPIs require SUMO trip-info
        (``extract_kpis``), which a stopped-early demo episode need not have. What the row does
        carry is the full version chain, which is what makes the trace replayable and citable.
        """
        import json

        from sqlalchemy.orm import Session

        from src.db.engine import create_db_engine, init_db
        from src.db.models import Episode, ExperimentRun
        from src.provenance.versions import git_sha, sumo_version

        db_path = _REPO_ROOT / "data" / "traffic.db"
        engine = create_db_engine(db_path)
        init_db(engine)
        with Session(engine) as session:
            run = ExperimentRun(
                name="live-" + self.controller,
                mode="live",
                controller=self.controller,
                config=json.dumps(
                    {
                        "scenario": self.scenario,
                        "seed": self.seed,
                        "episode_length_s": self.episode_length_s,
                        "frames": self.status.frames,
                        "source": "api/live",
                    }
                ),
                run_id=self.status.run_id,
                git_sha=git_sha(short=True),
                sumo_version=sumo_version(),
            )
            session.add(run)
            session.flush()
            session.add(
                Episode(
                    run_id_fk=run.id,
                    index_in_run=0,
                    seed=self.seed,
                    scenario=self.scenario,
                    sim_duration=self.status.sim_time,
                    done_reason=self.status.state,
                )
            )
            session.commit()


class SessionManager:
    """Owns the single live session slot."""

    def __init__(self, unity_hub: Hub, dash_hub: Hub) -> None:
        self._lock = threading.Lock()
        self._current: LiveSession | None = None
        self._unity, self._dash = unity_hub, dash_hub

    @property
    def current(self) -> LiveSession | None:
        """The active (or most recently finished) session, if any."""
        return self._current

    def start(
        self, *, controller: str, scenario: str, seed: int, episode_length_s: int = 3600,
        trace: bool = True,
    ) -> LiveSession:
        """Start a session, or raise :class:`SessionBusyError` if one is already running."""
        with self._lock:
            if self._current is not None and self._current.alive:
                raise SessionBusyError(
                    "a session is already running (" + self._current.controller + " on "
                    + self._current.scenario + "); stop it with DELETE /sessions/current"
                )
            session = LiveSession(
                controller=controller,
                scenario=scenario,
                seed=seed,
                unity_hub=self._unity,
                dash_hub=self._dash,
                episode_length_s=episode_length_s,
                trace=trace,
            )
            self._current = session
            session.start()
            return session

    def stop(self) -> bool:
        """Stop the running session. Returns ``False`` if there was nothing to stop."""
        with self._lock:
            session = self._current
            if session is None or not session.alive:
                return False
            session.stop()
            return True
