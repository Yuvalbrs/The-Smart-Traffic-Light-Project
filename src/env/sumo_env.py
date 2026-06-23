"""T-02-01 - SUMOEnv: a Gymnasium-style wrapper around the SUMO intersection.

The environment the DQN (and every baseline) plugs into. One ``step`` = one agent
decision = pick a NEMA green phase (``Discrete(8)``) and advance the simulation
``decision_interval_s`` seconds. Observation is the locked 20-dim base vector
(12 normalized pressures + 8-dim current-phase one-hot, state-space.md); reward is
``-sum|pressure| - lambda*1[switched]`` (reward-function.md, unnormalized pressures).

Two B3 guards are baked in (research-sumo.md Round 3 / open-items B3) - without
them every multi-episode run silently corrupts:

1. ``reset()`` uses ``traci.load(args)`` (not ``close``+``start``) to flush the
   insertion buffer between episodes in the same process.
2. ``getMinExpectedNumber() == 0`` terminates the episode naturally (the insertion
   buffer is empty), so leftover routed vehicles never leak across episodes.

Scope (T-02-01): the core env only. Action masking (min/max-green) is T-02-02 and
yellow/all-red transitions are T-02-03 - here a phase change is applied instantly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NET_FILE = _REPO_ROOT / "config" / "network" / "intersection.net.xml"
_ACTUATED_ADD_FILE = _REPO_ROOT / "config" / "network" / "actuated.add.xml"
_ACTUATED_PROGRAM = "actuated"

_OBS_DIM = 20
_PRESSURE_CLIP = 10.0  # clip pressures to +/-10 then /10 -> [-1, 1] (state-space.md)


class SUMOEnv(gym.Env):
    """Gymnasium env wrapping a single SUMO intersection under TraCI control.

    Parameters
    ----------
    route_file : str or Path
        The ``.rou.xml`` for this run (one (scenario, seed); see build_routes).
    net_file : str or Path, optional
        The network file. Defaults to the real intersection net.
    tls_id : str, optional
        Traffic-light id. Default ``"C"``.
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
        program drives the lights (T-02-06): the env loads the actuated
        additional-file, switches ``C`` to it, and ``step`` just advances the
        window and reads metrics; the ``action`` argument is ignored.
    additional_file : str or Path, optional
        The actuated additional-file (program + detectors). Defaults to the
        committed ``actuated.add.xml`` when ``signal_mode == "actuated"``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        route_file: str | Path,
        *,
        net_file: str | Path = _NET_FILE,
        tls_id: str = "C",
        episode_length_s: int = 3600,
        decision_interval_s: int = 10,
        switch_penalty: float = 0.1,
        sumo_seed: int = 42,
        use_gui: bool = False,
        movements_path: str | Path = _VAULT_MOVEMENTS,
        signal_mode: str = "rl",
        additional_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        if signal_mode not in ("rl", "actuated"):
            raise ValueError(f"signal_mode must be 'rl' or 'actuated', got {signal_mode!r}")
        self._route_file = Path(route_file)
        self._net_file = Path(net_file)
        self._tls_id = tls_id
        self._episode_length_s = episode_length_s
        self._decision_interval_s = decision_interval_s
        self._switch_penalty = switch_penalty
        self._sumo_seed = sumo_seed
        self._use_gui = use_gui
        self._movements_path = movements_path
        self._signal_mode = signal_mode
        self._additional_file = (
            Path(additional_file)
            if additional_file is not None
            else (_ACTUATED_ADD_FILE if signal_mode == "actuated" else None)
        )

        self.action_space = gym.spaces.Discrete(N_PHASES)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32
        )

        self._started = False
        self._intersection: Intersection | None = None
        self._last_action = 0
        self._time_in_phase = 0.0  # green seconds the current phase has been active
        self._sim_time = 0.0
        self._loaded = 0
        self._departed = 0
        self._arrived = 0

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
        ]
        if self._additional_file is not None:  # actuated program + detectors
            args += ["-a", str(self._additional_file)]
        return args

    # --- Gym API ---

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start or reload the simulation and return ``(obs, info)``."""
        super().reset(seed=seed)
        args = self._sumo_args()
        if not self._started:
            binary = checkBinary("sumo-gui" if self._use_gui else "sumo")
            traci.start([binary] + args)
            self._started = True
        else:
            traci.load(args)  # B3 guard #1: flush the insertion buffer, reuse process

        if self._intersection is None:  # net is constant -> build the model once
            self._intersection = Intersection.from_traci(
                traci, self._tls_id, movements_path=self._movements_path
            )

        self._last_action = 0
        self._time_in_phase = 0.0
        self._sim_time = 0.0
        self._loaded = self._departed = self._arrived = 0
        if self._signal_mode == "actuated":
            # hand the lights to SUMO's actuated program; we never command them.
            traci.trafficlight.setProgram(self._tls_id, _ACTUATED_PROGRAM)
        else:
            traci.trafficlight.setRedYellowGreenState(
                self._tls_id, self._intersection.green_state(0)
            )
        return self._observe(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply ``action``, advance the decision window, return the Gym 5-tuple.

        On a phase change the window is spent as: 3 s yellow, then (only when the
        change crosses the NS<->EW barrier) 2 s all-red, then the new green for the
        remainder - so the simulation always advances ``decision_interval_s`` and
        a phase never snaps green->green (T-02-03). Free right turns stay green
        throughout.
        """
        if self._intersection is None:
            raise RuntimeError("step() called before reset()")
        if self._signal_mode == "actuated":
            return self._step_actuated()
        action = int(action)
        ix = self._intersection
        prev = self._last_action
        switched = action != prev

        terminated = False
        remaining = self._decision_interval_s

        if switched:
            traci.trafficlight.setRedYellowGreenState(self._tls_id, ix.yellow_state(prev, action))
            terminated, remaining = self._advance(ix.yellow_s, remaining)
            if not terminated and ix.is_barrier_crossing(prev, action):
                traci.trafficlight.setRedYellowGreenState(self._tls_id, ix.all_red_state())
                terminated, remaining = self._advance(ix.all_red_s, remaining)

        if not terminated:  # the (new) green for the rest of the window
            traci.trafficlight.setRedYellowGreenState(self._tls_id, ix.green_state(action))
            terminated, remaining = self._advance(remaining, remaining)

        truncated = (not terminated) and self._sim_time >= self._episode_length_s
        pressures = ix.pressures(traci)  # unnormalized, for reward
        reward = float(-np.abs(pressures).sum())
        if switched:
            reward -= self._switch_penalty

        # Update the green timer for the NEXT mask (point: timer reflects state AFTER
        # the yellow/all-red insertion). A switch resets it to the green run that
        # actually elapsed this window; a hold accumulates the full window.
        if switched:
            transition_s = ix.yellow_s + (ix.all_red_s if ix.is_barrier_crossing(prev, action) else 0)
            self._time_in_phase = float(max(0, self._decision_interval_s - transition_s))
        else:
            self._time_in_phase += self._decision_interval_s
        self._last_action = action

        obs = self._observe(pressures)
        return obs, reward, terminated, truncated, self._info(done=terminated or truncated)

    def _step_actuated(
        self,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one decision window while SUMO's actuated program drives ``C``.

        The env issues no light commands and applies no mask/transition logic - it
        just steps the window and reads the same pressures/metrics every other
        controller is scored on. Reward carries no switch penalty (the agent makes
        no choice here). The phase one-hot is recovered from SUMO's live state.
        """
        ix = self._intersection
        assert ix is not None
        terminated, _ = self._advance(self._decision_interval_s, self._decision_interval_s)
        truncated = (not terminated) and self._sim_time >= self._episode_length_s
        pressures = ix.pressures(traci)
        reward = float(-np.abs(pressures).sum())
        live = traci.trafficlight.getRedYellowGreenState(self._tls_id)
        action = ix.action_for_state(live)  # None during yellow/all-red -> hold last
        if action is not None:
            self._last_action = action
        obs = self._observe(pressures)
        return obs, reward, terminated, truncated, self._info(done=terminated or truncated)

    def get_action_mask(self) -> np.ndarray:
        """Return the length-8 boolean action mask for the current decision point.

        Forbids switching before min-green (10 s) and forces a switch at max-green
        (60 s); free choice in between. ``mask.any()`` always holds. In actuated
        mode the mask is meaningless (SUMO owns the timing) - all actions read valid.
        """
        ix = self._intersection
        assert ix is not None, "get_action_mask() called before reset()"
        if self._signal_mode == "actuated":
            return np.ones(N_PHASES, dtype=bool)
        return compute_mask(
            self._last_action,
            self._time_in_phase,
            min_green=ix.min_green_s,
            max_green=ix.max_green_s,
        )

    def _advance(self, n_ticks: int, remaining: int) -> tuple[bool, int]:
        """Step the sim up to ``n_ticks`` (capped by ``remaining``), updating counters.

        Returns ``(terminated, remaining)`` where ``terminated`` is the B3 guard #2
        condition (``getMinExpectedNumber() == 0``).
        """
        terminated = False
        for _ in range(min(n_ticks, remaining)):
            traci.simulationStep()
            self._sim_time = traci.simulation.getTime()
            self._loaded += traci.simulation.getLoadedNumber()
            self._departed += traci.simulation.getDepartedNumber()
            self._arrived += traci.simulation.getArrivedNumber()
            remaining -= 1
            if traci.simulation.getMinExpectedNumber() == 0:  # B3 guard #2
                terminated = True
                break
        return terminated, remaining

    def close(self) -> None:
        """Close the TraCI connection (idempotent)."""
        if self._started:
            try:
                traci.close()
            except Exception:  # noqa: BLE001 - closing must never raise
                pass
            self._started = False

    # --- helpers ---

    def _observe(self, pressures: np.ndarray | None = None) -> np.ndarray:
        """Build the 20-dim observation (12 normalized pressures + phase one-hot)."""
        assert self._intersection is not None
        if pressures is None:
            pressures = self._intersection.pressures(traci)
        norm = np.clip(pressures, -_PRESSURE_CLIP, _PRESSURE_CLIP) / _PRESSURE_CLIP
        one_hot = np.zeros(N_PHASES, dtype=np.float32)
        one_hot[self._last_action] = 1.0
        return np.concatenate([norm.astype(np.float32), one_hot])

    def _info(self, done: bool = False) -> dict[str, Any]:
        """Per-step info: timer + action mask; on episode end, gridlock counters."""
        assert self._intersection is not None
        info: dict[str, Any] = {
            "sim_time": self._sim_time,
            "phase": self._last_action,
            "time_in_phase": self._time_in_phase,
            "mask": self.get_action_mask(),
            "barrier_crossing": barrier_crossing_mask(self._intersection, self._last_action),
        }
        if done:
            backlog = (self._loaded - self._departed) / self._loaded if self._loaded else 0.0
            info["episode"] = {
                "loaded_count": self._loaded,
                "departed_count": self._departed,
                "arrived_count": self._arrived,
                "insertion_backlog_fraction": backlog,
            }
        return info
