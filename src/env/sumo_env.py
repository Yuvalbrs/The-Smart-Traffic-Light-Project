"""T-02-01 / Step-2 - SUMOEnv: a decentralized multi-agent SUMO wrapper.

The environment the DQN (and every baseline) plugs into. One ``step`` = one round
of agent decisions = each controlled TLS picks a NEMA green phase (``Discrete(8)``)
and the simulation advances ``decision_interval_s`` seconds ONCE. Observation is the
locked 20-dim base vector PER AGENT (12 normalized pressures + 8-dim current-phase
one-hot, state-space.md); reward is the LOCAL ``-sum|pressure| - lambda*1[switched]``
per TLS (reward-function.md, unnormalized pressures).

Two operating modes share one implementation:

* **Single-agent (legacy, default)** - constructed with the scalar ``tls_id`` (default
  ``"C"``). ``reset()`` returns ``(obs_array, info_dict)``; ``step(action: int)`` returns
  the Gym 5-tuple of scalars; ``get_action_mask()`` returns one length-8 array. This is
  the exact pre-Step-2 API every existing caller (RL, actuated, baselines, tracer)
  depends on - preserved byte-for-byte.
* **Multi-agent (arterial, PettingZoo-parallel STYLE)** - constructed with
  ``tls_ids=["C1", "C2", ...]``. ``reset()`` returns ``(obs, info)`` dicts keyed by
  tls_id; ``step(actions: dict[str, int])`` returns ``(obs, rewards, terminations,
  truncations, infos)`` all dict-keyed; ``get_action_mask(tls_id)`` is per-TLS. No
  ``pettingzoo`` dependency is added - only the dict-keyed shape is borrowed.

The yellow/all-red transition machine runs INDEPENDENTLY per TLS (each has its own
yellow timer, all-red timer, and pending target phase), while simulated time advances
globally once per decision window: within the window each TLS follows its own per-tick
schedule (yellow -> optional all-red -> new green), so C1 switching never perturbs C2's
timing or mask. Free right turns stay green throughout.

Two B3 guards are baked in (research-sumo.md Round 3 / open-items B3) and stay
NETWORK-GLOBAL - when the network is done, every agent terminates:

1. ``reset()`` uses ``traci.load(args)`` (not ``close``+``start``) to flush the
   insertion buffer between episodes in the same process.
2. ``getMinExpectedNumber() == 0`` terminates the episode naturally (the insertion
   buffer is empty), so leftover routed vehicles never leak across episodes.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from sumolib import checkBinary

import traci
from src.env.intersection import (
    N_MOVEMENTS,
    N_PHASES,
    _VAULT_MOVEMENTS,
    Intersection,
)
from src.env.masking import barrier_crossing_mask, compute_mask
from src.trace import JsonlWriter, MovementResolver, build_sim_frame

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NET_FILE = _REPO_ROOT / "config" / "network" / "intersection.net.xml"
_ACTUATED_ADD_FILE = _REPO_ROOT / "config" / "network" / "actuated.add.xml"
_ACTUATED_PROGRAM = "actuated"

_OBS_DIM = 20
_PRESSURE_CLIP = 10.0  # clip pressures to +/-10 then /10 -> [-1, 1] (state-space.md)


def gridlock_penalty(max_queue: float, mu: float, threshold: float) -> float:
    """Anti-gridlock reward shaping (v2 ablation, off by default).

    Penalizes the WORST movement's queue beyond a saturation ``threshold`` - the locked reward
    ``-sum|pressure|`` is a sum and so is blind to a single-movement queue runaway, which is the
    proximate cause of gridlock cascades. This term gives the agent a direct gradient away from
    that state. ``mu <= 0`` -> 0.0 (no-op: the pre-registered reward is exactly unchanged).
    """
    if mu <= 0.0:
        return 0.0
    return mu * max(0.0, float(max_queue) - threshold)


class SUMOEnv(gym.Env):
    """Gymnasium env wrapping one OR several SUMO intersections under TraCI control.

    Parameters
    ----------
    route_file : str or Path
        The ``.rou.xml`` for this run (one (scenario, seed); see build_routes).
    net_file : str or Path, optional
        The network file. Defaults to the real single-intersection net. Pass the
        arterial net when driving ``["C1", "C2"]``.
    tls_id : str, optional
        Single-agent traffic-light id. Default ``"C"``. Ignored when ``tls_ids`` is
        given.
    tls_ids : list of str, optional
        When given, the env runs in MULTI-AGENT mode over exactly these TLS ids and
        the reset/step/mask API becomes dict-keyed. When ``None`` (default) the env
        runs the legacy single-agent scalar API over ``tls_id``.
    binding_file : str or Path, optional
        The link-index binding yaml. Defaults to :meth:`Intersection.from_traci`'s
        own default (the legacy single-TLS artifact). Pass the multi-TLS arterial
        binding when driving ``["C1", "C2"]``.
    episode_length_s : int, optional
        Episode horizon in simulated seconds. Default 3600.
    decision_interval_s : int, optional
        Simulated seconds advanced per ``step``. Default 10.
    switch_penalty : float, optional
        ``lambda`` in the reward. Default 0.1.
    sumo_seed : int, optional
        SUMO RNG seed (determinism). Default 42.
    use_gui : bool, optional
        Launch ``sumo-gui`` instead of headless ``sumo``. Default ``False``.
    movements_path : str or Path, optional
        Path to ``movements.yaml`` (vault SSOT by default).
    signal_mode : str, optional
        ``"rl"`` (default) - Python commands the lights every step (DQN + the
        Webster/max-pressure baselines). ``"actuated"`` - SUMO's own actuated
        program drives the lights (T-02-06, single-agent only): the env loads the
        actuated additional-file, switches the TLS to it, and ``step`` just advances
        the window and reads metrics; the ``action`` argument is ignored.
    additional_file : str or Path, optional
        The actuated additional-file (program + detectors). Defaults to the
        committed ``actuated.add.xml`` when ``signal_mode == "actuated"``.
    trace_path : str or Path, optional
        When set, write one ``sim_frame`` per simulated second to this JSONL file
        (the T-01-04 tracer, single-agent only). The file is rewritten fresh each
        ``reset``. Off by default (no tracing overhead).
    on_frame : callable, optional
        When set, called with each 1 Hz ``sim_frame`` dict as it is produced - the SAME
        frame the tracer writes, so the live API stream and the JSONL trace can never
        drift apart (T-05-01). Independent of ``trace_path``: either, both or neither may
        be active. Single-agent only. The callback runs inside the simulation loop, so it
        must not block (the API side pushes into bounded queues and returns immediately).
    tripinfo_path : str or Path, optional
        When set, tell SUMO to write per-vehicle trip-info XML (``--tripinfo-output``)
        to this path - the second artifact the KPI extractor needs. SUMO finalizes the
        file when the simulation ends, so read it only after ``close()``. Off by
        default. For multi-episode reuse of one env, pass a fresh path each episode
        via ``reset(options={"tripinfo_path": ...})`` (a plain reload reuses the same
        path and would overwrite it; ``reset`` raises if you forget).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        route_file: str | Path,
        *,
        net_file: str | Path = _NET_FILE,
        tls_id: str = "C",
        tls_ids: list[str] | None = None,
        binding_file: str | Path | None = None,
        episode_length_s: int = 3600,
        decision_interval_s: int = 10,
        switch_penalty: float = 0.1,
        gridlock_penalty_mu: float = 0.0,
        gridlock_queue_threshold: float = 20.0,
        sumo_seed: int = 42,
        use_gui: bool = False,
        movements_path: str | Path = _VAULT_MOVEMENTS,
        signal_mode: str = "rl",
        additional_file: str | Path | None = None,
        trace_path: str | Path | None = None,
        tripinfo_path: str | Path | None = None,
        on_frame: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__()
        if signal_mode not in ("rl", "actuated"):
            raise ValueError(f"signal_mode must be 'rl' or 'actuated', got {signal_mode!r}")
        self._multi_agent = tls_ids is not None
        self._tls_ids: list[str] = list(tls_ids) if tls_ids is not None else [tls_id]
        if not self._tls_ids:
            raise ValueError("tls_ids must contain at least one TLS id")
        self._tls_id = self._tls_ids[0]  # primary TLS (tracing / actuated / legacy)
        if self._multi_agent and signal_mode == "actuated":
            raise ValueError("actuated signal_mode is single-agent only")
        if self._multi_agent and trace_path is not None:
            raise ValueError("per-frame tracing is single-agent only")
        if self._multi_agent and on_frame is not None:
            raise ValueError("on_frame streaming is single-agent only")

        self._route_file = Path(route_file)
        self._net_file = Path(net_file)
        self._binding_file = Path(binding_file) if binding_file is not None else None
        self._episode_length_s = episode_length_s
        self._decision_interval_s = decision_interval_s
        self._switch_penalty = switch_penalty
        self._gridlock_penalty_mu = gridlock_penalty_mu
        self._gridlock_queue_threshold = gridlock_queue_threshold
        self._sumo_seed = sumo_seed
        self._use_gui = use_gui
        self._movements_path = movements_path
        self._signal_mode = signal_mode
        self._additional_file = (
            Path(additional_file)
            if additional_file is not None
            else (_ACTUATED_ADD_FILE if signal_mode == "actuated" else None)
        )
        # optional per-second JSONL tracing (T-01-04 tracer wired in, single-agent).
        self._trace_path = Path(trace_path) if trace_path is not None else None
        # SUMO-native per-vehicle trip-info (the KPI extractor's second input);
        # written by SUMO at simulation end, read after close().
        self._tripinfo_path = Path(tripinfo_path) if tripinfo_path is not None else None
        # T-05-01: optional live sink for the SAME 1 Hz sim_frame the tracer writes, so the
        # API streams the tracer's frames rather than rebuilding a parallel (driftable) one.
        self._on_frame = on_frame
        self._tracer: JsonlWriter | None = None
        self._resolver: MovementResolver | None = None
        self._seq = 0
        self._episode_id = 0

        self.action_space = gym.spaces.Discrete(N_PHASES)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32
        )

        self._started = False
        # per-TLS model + bookkeeping (dict-keyed even in single-agent mode).
        self._intersections: dict[str, Intersection] = {}
        self._last_action: dict[str, int] = {t: 0 for t in self._tls_ids}
        self._time_in_phase: dict[str, float] = {t: 0.0 for t in self._tls_ids}
        # NEMA phase stamped on trace frames in the current window (per TLS).
        self._trace_phase: dict[str, int] = {t: 0 for t in self._tls_ids}
        # global (network-wide) simulation counters.
        self._sim_time = 0.0
        self._loaded = 0
        self._departed = 0
        self._arrived = 0
        self._collisions = 0  # junction/rear-end collisions; must stay 0 (see _tick)

    # --- SUMO command ---

    def _sumo_args(self) -> list[str]:
        """The deterministic SUMO argument list (no binary), shared by start/load."""
        args = [
            "-n", str(self._net_file),
            "-r", str(self._route_file),
            "--step-length", "1.0",
            "--seed", str(self._sumo_seed),
            "--time-to-teleport", "-1",  # B3: gridlock stays visible, deterministic
            "--step-method.ballistic",  # deterministic car-following
            "--threads", "1",
            "--no-step-log", "true",
            "--no-warnings", "true",
            # SUMO does NOT check collisions inside junctions by default; that silence hid
            # the free-right-turn priority defect through 291 tests and a full campaign
            # (decisions.md 2026-08-28). "warn" keeps physics untouched — the default
            # "teleport" would silently remove colliding vehicles and alter traffic.
            "--collision.check-junctions", "true",
            "--collision.action", "warn",
        ]
        if self._additional_file is not None:  # actuated program + detectors
            args += ["-a", str(self._additional_file)]
        if self._tripinfo_path is not None:  # per-vehicle KPIs (wait/stops/p95)
            args += ["--tripinfo-output", str(self._tripinfo_path)]
        return args

    def _build_intersection(self, tls_id: str) -> Intersection:
        """Construct the :class:`Intersection` model for ``tls_id`` from the live net."""
        kwargs: dict[str, Any] = {"movements_path": self._movements_path}
        if self._binding_file is not None:
            kwargs["binding_path"] = self._binding_file
        return Intersection.from_traci(traci, tls_id, **kwargs)

    # --- Gym API ---

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, Any]:
        """Start or reload the simulation and return ``(obs, info)``.

        In single-agent mode ``obs`` is the 20-dim array and ``info`` the per-step
        dict (legacy contract). In multi-agent mode both are dicts keyed by tls_id.

        ``options`` may carry per-episode output paths ``{"trace_path": ...,
        "tripinfo_path": ...}``. Supplying them is REQUIRED when reusing one env
        across episodes with a file output configured: ``reset()`` reloads via
        ``traci.load``, which finalizes+truncates any fixed output file, so without a
        fresh path per episode only the last episode's trace/trip-info would survive.
        """
        super().reset(seed=seed)
        opts = options or {}
        if "trace_path" in opts:
            self._trace_path = Path(opts["trace_path"]) if opts["trace_path"] else None
        if "tripinfo_path" in opts:
            self._tripinfo_path = Path(opts["tripinfo_path"]) if opts["tripinfo_path"] else None
        if self._started:  # reuse via traci.load would overwrite a fixed output file
            if self._trace_path is not None and "trace_path" not in opts:
                raise RuntimeError(
                    "reset() reuse would overwrite the fixed trace_path; pass a fresh "
                    "reset(options={'trace_path': ...}) per episode, or use a new env."
                )
            if self._tripinfo_path is not None and "tripinfo_path" not in opts:
                raise RuntimeError(
                    "reset() reuse would overwrite the fixed tripinfo_path; pass a fresh "
                    "reset(options={'tripinfo_path': ...}) per episode, or use a new env."
                )
        args = self._sumo_args()
        if not self._started:
            binary = checkBinary("sumo-gui" if self._use_gui else "sumo")
            traci.start([binary] + args)
            self._started = True
        else:
            traci.load(args)  # B3 guard #1: flush the insertion buffer, reuse process

        if not self._intersections:  # net is constant -> build each model once
            for tls_id in self._tls_ids:
                self._intersections[tls_id] = self._build_intersection(tls_id)

        if self._trace_path is not None or self._on_frame is not None:
            if self._resolver is None:  # needed by _build_frame for tracing AND streaming
                self._resolver = MovementResolver.from_traci(traci, self._tls_id)
        if self._trace_path is not None:  # fresh JSONL trace per episode (single-agent)
            if self._tracer is not None:
                self._tracer.close()
            self._tracer = JsonlWriter(self._trace_path).__enter__()
            self._seq = 0
            self._episode_id += 1

        for tls_id in self._tls_ids:
            self._last_action[tls_id] = 0
            self._time_in_phase[tls_id] = 0.0
            self._trace_phase[tls_id] = 0
        self._sim_time = 0.0
        self._loaded = self._departed = self._arrived = self._collisions = 0

        if self._signal_mode == "actuated":
            # hand the lights to SUMO's actuated program; we never command them.
            traci.trafficlight.setProgram(self._tls_id, _ACTUATED_PROGRAM)
        else:
            for tls_id in self._tls_ids:
                traci.trafficlight.setRedYellowGreenState(
                    tls_id, self._intersections[tls_id].green_state(0)
                )

        if self._multi_agent:
            obs = {t: self._observe(t) for t in self._tls_ids}
            info = {t: self._info(t) for t in self._tls_ids}
            return obs, info
        return self._observe(self._tls_id), self._info(self._tls_id)

    def step(self, action: int | dict[str, int]) -> tuple[Any, Any, Any, Any, Any]:
        """Apply ``action``(s), advance the decision window ONCE, return results.

        Single-agent: ``action`` is an int and the return is the Gym 5-tuple of
        scalars. Multi-agent: ``action`` is a ``{tls_id: int}`` dict and the return
        is ``(obs, rewards, terminations, truncations, infos)`` all dict-keyed.

        On a phase change a TLS spends its window as: 3 s yellow, then (only when the
        change crosses the NS<->EW barrier) 2 s all-red, then the new green for the
        remainder - so the simulation always advances ``decision_interval_s`` and a
        phase never snaps green->green (T-02-03). Each TLS runs this schedule
        independently; free right turns stay green throughout.
        """
        if not self._intersections:
            raise RuntimeError("step() called before reset()")
        if self._signal_mode == "actuated":
            return self._step_actuated()
        if isinstance(action, dict):
            return self._step_core(action)
        obs, rewards, terms, truncs, infos = self._step_core({self._tls_id: int(action)})
        t = self._tls_id
        return obs[t], rewards[t], terms[t], truncs[t], infos[t]

    def _step_core(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Per-TLS transition + one global window advance; returns dict-keyed results.

        Builds an independent per-tick RYG schedule for every acting TLS (yellow ->
        optional all-red -> new green), then advances ``decision_interval_s`` ticks
        once, applying each TLS's schedule in lock-step. Reward, mask timer, and phase
        one-hot are all computed LOCALLY per TLS; the B3 termination is global.
        """
        schedules: dict[str, list[str]] = {}
        for tls_id, raw_action in actions.items():
            action = int(raw_action)
            self._trace_phase[tls_id] = action  # frames in this window carry the choice
            ix = self._intersections[tls_id]
            prev = self._last_action[tls_id]
            ticks: list[str] = []
            if action != prev:
                ticks += [ix.yellow_state(prev, action)] * ix.yellow_s
                if ix.is_barrier_crossing(prev, action):
                    ticks += [ix.all_red_state()] * ix.all_red_s
            green = ix.green_state(action)
            ticks += [green] * max(0, self._decision_interval_s - len(ticks))
            schedules[tls_id] = ticks[: self._decision_interval_s]

        terminated = self._advance_window(actions.keys(), schedules)
        truncated = (not terminated) and self._sim_time >= self._episode_length_s
        done = terminated or truncated

        obs: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terms: dict[str, bool] = {}
        truncs: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}
        for tls_id, raw_action in actions.items():
            action = int(raw_action)
            ix = self._intersections[tls_id]
            prev = self._last_action[tls_id]
            switched = action != prev
            pressures = ix.pressures(traci)  # unnormalized, LOCAL, for reward
            reward = float(-np.abs(pressures).sum())
            if switched:
                reward -= self._switch_penalty
            if self._gridlock_penalty_mu > 0.0:  # v2 anti-gridlock shaping (off by default)
                reward -= gridlock_penalty(
                    float(np.max(ix.movement_queues(traci))),
                    self._gridlock_penalty_mu,
                    self._gridlock_queue_threshold,
                )

            # Green timer for the NEXT mask: a switch resets it to the green run that
            # actually elapsed this window (window minus this TLS's own transition); a
            # hold accumulates the full window. Independent per TLS.
            if switched:
                transition_s = ix.yellow_s + (
                    ix.all_red_s if ix.is_barrier_crossing(prev, action) else 0
                )
                self._time_in_phase[tls_id] = float(
                    max(0, self._decision_interval_s - transition_s)
                )
            else:
                self._time_in_phase[tls_id] += self._decision_interval_s
            self._last_action[tls_id] = action  # one-hot / info reflect the NEW phase

            obs[tls_id] = self._observe(tls_id, pressures)
            rewards[tls_id] = reward
            terms[tls_id] = terminated
            truncs[tls_id] = truncated
            infos[tls_id] = self._info(tls_id, done=done)
        return obs, rewards, terms, truncs, infos

    @property
    def departed_count(self) -> int:
        """Cumulative vehicles inserted into the network so far this episode.

        A controller-INDEPENDENT demand signal (insertion is driven by the route file's arrival
        process, not by how the lights are timed - unless gridlock blocks insertion). Used by the
        episode-level selector to judge demand from a short safe probe, where queue/pressure would
        instead reflect how well the probe controller coped rather than how much traffic arrived.
        """
        return self._departed

    @property
    def arrived_count(self) -> int:
        """Cumulative vehicles that completed their trip so far this episode."""
        return self._arrived

    @property
    def loaded_count(self) -> int:
        """Cumulative vehicles the demand has tried to insert so far this episode.

        ``loaded - departed`` is the insertion backlog: demand that exists but could not enter
        the network. On a saturated scenario this is the quantity that actually diverges.
        """
        return self._loaded

    def movement_pressures(self) -> np.ndarray:
        """Return the primary TLS's unnormalized ``pressure[12]`` at the current state.

        The observation carries these clipped to +/-10 and scaled; the live dashboard wants the
        raw values (T-05-01), so read them straight off the intersection model.
        """
        ix = self._intersections.get(self._tls_id)
        assert ix is not None, "movement_pressures() called before reset()"
        return ix.pressures(traci)

    def movement_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(queue[12], count[12])`` for the primary TLS at the current state.

        The per-step LSTM features (T-01-05 data generation): queue is the halting
        count, count is the vehicle count, both over each movement's incoming lanes.
        """
        ix = self._intersections.get(self._tls_id)
        assert ix is not None, "movement_features() called before reset()"
        return ix.movement_queues(traci), ix.movement_counts(traci)

    def _step_actuated(
        self,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one decision window while SUMO's actuated program drives the TLS.

        The env issues no light commands and applies no mask/transition logic - it
        just steps the window and reads the same pressures/metrics every other
        controller is scored on. Reward carries no switch penalty (the agent makes
        no choice here). The phase one-hot is recovered from SUMO's live state.
        Single-agent only.
        """
        tls_id = self._tls_id
        ix = self._intersections[tls_id]
        self._trace_phase[tls_id] = self._last_action[tls_id]  # SUMO owns the lights
        terminated = False
        for _ in range(self._decision_interval_s):
            if self._tick():
                terminated = True
                break
        truncated = (not terminated) and self._sim_time >= self._episode_length_s
        pressures = ix.pressures(traci)
        reward = float(-np.abs(pressures).sum())
        if self._gridlock_penalty_mu > 0.0:  # v2 anti-gridlock shaping (off by default)
            reward -= gridlock_penalty(
                float(np.max(ix.movement_queues(traci))),
                self._gridlock_penalty_mu,
                self._gridlock_queue_threshold,
            )
        live = traci.trafficlight.getRedYellowGreenState(tls_id)
        action = ix.action_for_state(live)  # None during yellow/all-red -> hold last
        if action is not None:
            self._last_action[tls_id] = action
        obs = self._observe(tls_id, pressures)
        return obs, reward, terminated, truncated, self._info(tls_id, done=terminated or truncated)

    def get_action_mask(self, tls_id: str | None = None) -> np.ndarray:
        """Return the length-8 boolean action mask for a TLS's current decision point.

        Forbids switching before min-green (10 s) and forces a switch at max-green
        (60 s); free choice in between. ``mask.any()`` always holds. In actuated
        mode the mask is meaningless (SUMO owns the timing) - all actions read valid.
        ``tls_id=None`` returns the primary TLS's mask (legacy single-agent API).
        """
        if tls_id is None:
            tls_id = self._tls_id
        ix = self._intersections.get(tls_id)
        assert ix is not None, "get_action_mask() called before reset()"
        if self._signal_mode == "actuated":
            return np.ones(N_PHASES, dtype=bool)
        return compute_mask(
            self._last_action[tls_id],
            self._time_in_phase[tls_id],
            min_green=ix.min_green_s,
            max_green=ix.max_green_s,
        )

    def _advance_window(
        self, tls_ids: Iterable[str], schedules: dict[str, list[str]]
    ) -> bool:
        """Step the full decision window once, applying each TLS's per-tick schedule.

        At each tick every TLS's RYG is (re)applied only when it changes from the
        prior tick (segment boundaries), so a held green is set once and a switching
        TLS advances yellow -> all-red -> green in place. Returns ``terminated`` (the
        B3 guard #2 condition ``getMinExpectedNumber() == 0``, network-global).
        """
        tls_list = list(tls_ids)
        prev_state: dict[str, str | None] = {t: None for t in tls_list}
        for tick in range(self._decision_interval_s):
            for tls_id in tls_list:
                ryg = schedules[tls_id][tick]
                if ryg != prev_state[tls_id]:
                    traci.trafficlight.setRedYellowGreenState(tls_id, ryg)
                    prev_state[tls_id] = ryg
            if self._tick():
                return True
        return False

    def _tick(self) -> bool:
        """Advance one simulated second, update global counters + tracing.

        Returns ``True`` when the network has emptied (B3 guard #2).
        """
        traci.simulationStep()
        self._sim_time = traci.simulation.getTime()
        self._loaded += traci.simulation.getLoadedNumber()
        self._departed += traci.simulation.getDepartedNumber()
        self._arrived += traci.simulation.getArrivedNumber()
        colliding = traci.simulation.getCollidingVehiclesNumber()
        if colliding:
            # Fail LOUDLY. A collision means a right-of-way/physics modelling error —
            # the class of bug that silently invalidated the entire first campaign.
            self._collisions += colliding
            raise RuntimeError(
                f"SUMO reported {colliding} colliding vehicle(s) at t={self._sim_time:.0f}s "
                f"({traci.simulation.getCollidingVehiclesIDList()}). The environment model "
                "is wrong; investigate — do not suppress (see decisions.md 2026-08-28)."
            )
        if self._tracer is not None or self._on_frame is not None:
            frame = self._build_frame()  # one 1 Hz sim_frame per tick (single-agent)
            if self._tracer is not None:
                self._tracer.write(frame)
            if self._on_frame is not None:
                self._on_frame(frame)
        return traci.simulation.getMinExpectedNumber() == 0

    def _build_frame(self) -> dict[str, Any]:
        """Assemble + count one ``sim_frame`` for the current tick (tracing on)."""
        assert self._resolver is not None
        frame = build_sim_frame(
            traci,
            self._tls_id,
            seq=self._seq,
            episode_id=self._episode_id,
            phase_index=self._trace_phase[self._tls_id],
            resolver=self._resolver,
            sim_time=self._sim_time,
        )
        self._seq += 1
        return frame

    def close(self) -> None:
        """Close the trace writer + TraCI connection (idempotent)."""
        if self._tracer is not None:
            self._tracer.close()
            self._tracer = None
        if self._started:
            try:
                traci.close()
            except Exception:  # noqa: BLE001 - closing must never raise
                pass
            self._started = False

    # --- helpers ---

    def _observe(self, tls_id: str, pressures: np.ndarray | None = None) -> np.ndarray:
        """Build ``tls_id``'s 20-dim observation (12 normalized pressures + one-hot)."""
        ix = self._intersections[tls_id]
        if pressures is None:
            pressures = ix.pressures(traci)
        norm = np.clip(pressures, -_PRESSURE_CLIP, _PRESSURE_CLIP) / _PRESSURE_CLIP
        one_hot = np.zeros(N_PHASES, dtype=np.float32)
        one_hot[self._last_action[tls_id]] = 1.0
        return np.concatenate([norm.astype(np.float32), one_hot])

    def _info(self, tls_id: str, done: bool = False) -> dict[str, Any]:
        """Per-step info for ``tls_id``: timer + mask; on episode end, global counters."""
        ix = self._intersections[tls_id]
        info: dict[str, Any] = {
            "sim_time": self._sim_time,
            "phase": self._last_action[tls_id],
            "time_in_phase": self._time_in_phase[tls_id],
            "mask": self.get_action_mask(tls_id),
            "barrier_crossing": barrier_crossing_mask(ix, self._last_action[tls_id]),
        }
        if done:  # the episode counters are network-global (shared by all agents)
            backlog = (self._loaded - self._departed) / self._loaded if self._loaded else 0.0
            info["episode"] = {
                "loaded_count": self._loaded,
                "departed_count": self._departed,
                "arrived_count": self._arrived,
                "insertion_backlog_fraction": backlog,
                "collision_count": self._collisions,  # 0 unless the RuntimeError was caught
            }
        return info
