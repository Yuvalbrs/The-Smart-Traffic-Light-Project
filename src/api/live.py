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

import logging
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

# There was no logger anywhere in the service layer: a failed run's cause lived in one
# in-memory string that the next session start overwrote, and failed runs were excluded
# from persistence - so the evidence was gone twice over.
_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_TRACE_DIR = _REPO_ROOT / "data" / "live"

# Same KPI warm-up as `eval_runner --warmup` so a live row is comparable to an eval row.
_WARMUP_S = 300.0

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

# The DQN training seed the live demo runs. Picks BOTH which run-dir checkpoint is loaded and
# which official forecaster is paired with it (src/provenance/official.py A6.4: the pin is per
# DQN training seed) - one constant so the two selections cannot drift to different seeds.
LIVE_TRAIN_SEED = 42


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
    speed: float = 0.0  # simulated seconds per wall second; 0 = unpaced
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
            "speed": self.speed,
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
    speed : float, optional
        Simulated seconds per wall-clock second. ``0`` (the default) runs the episode as fast as
        the machine allows - right for tests and for filling the replay corpus. ``1.0`` paces it
        to real time so a human can watch the charts move; ``5.0`` is 5x real time. See
        :meth:`_pace`.
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
        speed: float = 0.0,
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
        self.speed = max(0.0, float(speed))
        self._unity, self._dash = unity_hub, dash_hub
        self._trace = trace
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._env: Any = None
        self._last_tick = 0.0
        self._pace_origin: float | None = None
        self._pace_frame0 = 0
        self._tripinfo_path: Path | None = None
        self._counters: dict[str, int] | None = None
        self.status = SessionStatus(
            run_id=new_run_id(),
            controller=controller,
            scenario=scenario,
            seed=seed,
            speed=self.speed,
        )

    # --- lifecycle ---

    def start(self) -> None:
        """Launch the worker thread (returns immediately)."""
        self._thread = threading.Thread(target=self._run, name="sumo-live", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 20.0) -> bool:
        """Ask the episode to end and wait for the worker to unwind.

        Returns whether the worker actually finished. It may not: the thread can be
        inside a long native ``simulationStep`` rather than the interruptible pacing
        wait, and Python threads cannot be killed. Reporting a clean stop we did not
        achieve is worse than reporting the truth, so the caller gets the real answer.
        """
        self._stop.set()
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        stopped = not self._thread.is_alive()
        if not stopped:
            _log.warning(
                "run %s did not stop within %.0fs; worker still alive "
                "(likely inside a native SUMO call)", self.status.run_id, timeout
            )
        return stopped

    @property
    def alive(self) -> bool:
        """Whether the worker thread is still running."""
        return self._thread is not None and self._thread.is_alive()

    # --- the worker ---

    def _on_frame(self, frame: dict[str, Any]) -> None:
        """Called by SUMOEnv once per simulated second, inside the stepping loop.

        Must never block *on a client*: both hubs hand off to bounded queues and return, so a
        stalled WebSocket subscriber can only starve itself. It may block on the session's own
        clock - see :meth:`_pace`.

        Also records loop lag - the wall-clock gap between consecutive simulated seconds - which
        ``/health`` reports as the honest measure of whether the sim is keeping up. The lag clock
        is stopped before :meth:`_pace` and restarted after it, so deliberate pacing sleep is
        excluded: under ``speed=1.0`` loop lag must still read the few milliseconds of real work,
        not the ~1 s we chose to wait.
        """
        now = time.perf_counter()
        if self._last_tick:
            lag = now - self._last_tick
            self.status.last_loop_lag_s = lag
            self.status.max_loop_lag_s = max(self.status.max_loop_lag_s, lag)

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
                    forecast_next_30s=self._latest_forecast(),
                    seq=int(frame.get("seq", self.status.frames)),
                    episode_id=int(frame.get("episode_id", 0)),
                )
            )
        except Exception as exc:  # a dashboard hiccup must never kill the simulation
            self.status.error = "dashboard frame failed: " + repr(exc)

        self._pace()
        self._last_tick = time.perf_counter()

    def _latest_forecast(self) -> list[float] | None:
        """The frozen forecaster's most recent prediction, flattened for the wire.

        This used to read ``getattr(self, "_forecast", None)`` - an attribute NOTHING in the
        codebase ever assigned. The default silently turned that into None on every frame, so the
        dashboard's forecast panel said "no forecaster attached" for every controller including
        the hybrid one, and the forecast the agent was actually consuming was never shown.

        The value was always one attribute away: when a forecaster is attached, ``self._env`` IS
        the :class:`HybridStateWrapper`, which records each prediction in ``last_forecast``.
        Controllers without a forecaster have no such attribute, so they still yield None - which
        is the honest answer for them, and what the panel is built to render.

        None is also correct for the first eleven decisions of a hybrid episode: the wrapper needs
        twelve steps of history before it can predict, and reports None until it has them. Zeros
        would read as a confident forecast of no traffic.
        """
        pred = getattr(getattr(self, "_env", None), "last_forecast", None)
        if pred is None:
            return None
        return np.asarray(pred, dtype=float).reshape(-1).tolist()

    def _pace(self) -> None:
        """Throttle the stepping loop to ``speed`` simulated seconds per wall-clock second.

        Without this the episode runs as fast as libsumo can step - measured at ~700x real time
        on this machine - so a 1 Hz "live" stream arrives in one burst and the dashboard charts
        snap to their final state instead of moving. Unpaced (``speed=0``) stays the default:
        tests and corpus-filling runs want the machine's full speed.

        Waits on :attr:`_stop` rather than sleeping, so ``DELETE /sessions/current`` stays
        responsive while a real-time run is mid-wait. The origin is taken on the first frame,
        not at thread start, so SUMO's startup cost is not charged to the schedule.
        """
        if self.speed <= 0.0:
            return
        if self._pace_origin is None:
            self._pace_origin = time.perf_counter()
            self._pace_frame0 = self.status.frames
            return
        elapsed_sim = self.status.frames - self._pace_frame0
        delay = (self._pace_origin + elapsed_sim / self.speed) - time.perf_counter()
        if delay > 0:
            self._stop.wait(timeout=delay)

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
            ckpt = _REPO_ROOT / "runs" / f"plain_seed{LIVE_TRAIN_SEED}" / "checkpoints" / "ep299.pt"
            algo = Algo(
                self.controller,
                "dqn",
                agent=_load_agent(ckpt, OBS_DIM),
                variant="plain",
                train_seed=LIVE_TRAIN_SEED,
                ckpt=str(ckpt),
            )
        else:  # dqn-hybrid
            from scripts.eval_runner import load_forecaster
            from src.provenance.official import official_lstm_checked, official_lstm_filename
            from src.ml.hybrid_wrapper import HYBRID_OBS_DIM

            ckpt = _REPO_ROOT / "runs" / f"hybrid_seed{LIVE_TRAIN_SEED}" / "checkpoints" / "ep299.pt"
            algo = Algo(
                self.controller,
                "dqn",
                agent=_load_agent(ckpt, HYBRID_OBS_DIM),
                forecaster=load_forecaster(str(official_lstm_checked(LIVE_TRAIN_SEED))),
                variant="hybrid",
                train_seed=LIVE_TRAIN_SEED,
                lstm_version=official_lstm_filename(LIVE_TRAIN_SEED),
                ckpt=str(ckpt),
            )

        trace_path = None
        if self._trace:
            LIVE_TRACE_DIR.mkdir(parents=True, exist_ok=True)
            trace_path = LIVE_TRACE_DIR / (self.status.run_id + ".jsonl")
            # SUMO only writes trip-info when asked at start-up, and extract_kpis cannot run
            # without it - which is why live runs used to persist a KPI-less episode row.
            self._tripinfo_path = LIVE_TRACE_DIR / (self.status.run_id + ".tripinfo.xml")
            self.status.trace_path = str(trace_path)

        env: Any = SUMOEnv(
            write_routes(scn, self.seed),
            episode_length_s=min(self.episode_length_s, scn.duration_s),
            sumo_seed=self.seed,
            signal_mode=algo.signal_mode,
            trace_path=trace_path,
            tripinfo_path=self._tripinfo_path,
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
            # loaded_count is the only source of the insertion backlog (trip-info holds completed
            # trips only), so the counters must be read before the env is closed.
            self._counters = info.get("episode")
            self.status.state = "stopped" if self._stop.is_set() else "finished"
        except Exception as exc:  # surface it in /sessions/current rather than dying silently
            self.status.state = "failed"
            self.status.error = repr(exc)
            # The in-memory error string is erased by the next session start, so without
            # this the only record of WHY a run died is gone. Collisions now raise from
            # the env, which makes this path live rather than theoretical.
            _log.exception(
                "live run %s failed (%s scenario=%s seed=%s)",
                self.status.run_id, self.controller, self.scenario, self.seed,
            )
        finally:
            self.status.finished_at = time.time()
            if env is not None:
                try:
                    env.close()
                except Exception:  # pragma: no cover - close is best-effort
                    pass
            self._env = None
            # A failed run is persisted too (KPIs stay NULL): excluding it meant a crashed
            # run left no DB row, so its on-disk trace was undiscoverable through the API
            # and the failure was invisible the moment a new session started.
            if self._trace and self.status.state in ("finished", "stopped", "failed"):
                try:
                    self._persist()
                except Exception as exc:  # provenance failure must not mask a good run
                    self.status.error = "persist failed: " + repr(exc)
                    _log.exception("persist failed for run %s", self.status.run_id)

    def _compute_kpis(self) -> Any | None:
        """Extract this episode's KPIs, or ``None`` when they would not be meaningful.

        Deliberately returns ``None`` unless the episode ran to its natural end: an operator who
        hits ``DELETE /sessions/current`` half way through has a truncated trip-info file, and a
        KPI row computed from it would be a real-looking number that means nothing. The original
        design skipped KPIs for *every* live run for this reason; the narrower rule keeps that
        protection while letting a completed demo carry its numbers.

        Uses :func:`src.metrics.kpi_extractor.extract_kpis` - the same one implementation the
        eval corpus used (``project-rules.md``: one KPI implementation, no drift) - and mirrors
        ``eval_runner``'s warm-up convention so a live run is comparable to an eval row.
        """
        if self.status.state != "finished":
            return None
        if self._tripinfo_path is None or not self._tripinfo_path.exists():
            return None
        if self.status.trace_path is None:
            return None

        from src.metrics.kpi_extractor import extract_kpis

        # eval_runner: `warmup = args.warmup if args.warmup < episode_length_s else 0.0`.
        # Short demo episodes therefore keep every second rather than discarding all of them.
        episode_length_s = float(self.status.sim_time)
        warmup_s = _WARMUP_S if _WARMUP_S < episode_length_s else 0.0
        return extract_kpis(
            self.status.trace_path,
            self._tripinfo_path,
            episode_counters=self._counters,
            warmup_s=warmup_s,
            episode_length_s=episode_length_s,
        )

    def _persist(self) -> None:
        """Record this live run in the results DB so it shows up in ``GET /runs``.

        Written with ``mode="live"`` so it can never be mistaken for evaluation data, and
        carrying the full version chain, which is what makes the trace replayable and citable.
        A completed episode also gets its 1:1 KPI row (see :meth:`_compute_kpis`); one that was
        stopped early gets the episode row alone, with KPI columns left NULL.
        """
        import json

        from sqlalchemy.orm import Session

        from src.db.engine import create_db_engine, init_db
        from src.db.models import Episode, EpisodeKpi, ExperimentRun
        from src.provenance.versions import git_sha, sumo_version

        kpis = self._compute_kpis()

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
            episode = Episode(
                run_id_fk=run.id,
                index_in_run=0,
                seed=self.seed,
                scenario=self.scenario,
                sim_duration=self.status.sim_time,
                done_reason=self.status.state,
            )
            if kpis is not None:
                episode.insertion_backlog_fraction = kpis.insertion_backlog_fraction
                episode.gridlock_censored = bool(kpis.gridlock_censored)
                episode.kpi = EpisodeKpi(
                    avg_waiting_time=kpis.avg_waiting_time,
                    avg_queue_length=kpis.avg_queue_length,
                    throughput=kpis.throughput,
                    num_stops=kpis.num_stops,
                    wait_p95=kpis.wait_p95,
                    fairness_std=kpis.fairness_std,
                    per_movement_max_wait=kpis.per_movement_max_wait,
                    per_movement_p95_wait=kpis.per_movement_p95_wait,
                    worst_movement_max_wait=kpis.worst_movement_max_wait,
                )
            session.add(episode)
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
        trace: bool = True, speed: float = 0.0,
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
                speed=speed,
            )
            self._current = session
            session.start()
            return session

    def stop(self) -> bool | None:
        """Stop the running session.

        ``False`` - nothing was running. ``True`` - the worker actually finished.
        ``None`` - the stop was requested but the worker is still unwinding, so the
        caller must not claim it stopped (it previously always returned ``True``).
        """
        with self._lock:
            session = self._current
            if session is None or not session.alive:
                return False
            return True if session.stop() else None
