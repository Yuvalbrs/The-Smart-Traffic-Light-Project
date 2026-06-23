"""T-02-01 - Physical model of the controlled intersection.

Separates the *intersection physics* (movements, the 8 NEMA phases, pressure
computation, and green-state synthesis) from the RL bookkeeping in
:class:`src.env.sumo_env.SUMOEnv`. The max-pressure (T-02-05) and Webster
(T-02-04) baselines reuse this model, so there is one implementation of
"pressure" and "what green looks like for action a" - no drift.

Built from two sources, composed at episode start:

* the **logical spec** ``specs/movements.yaml`` (vault SSOT): the 8 phases'
  green-movement sets and which movements are free (uncontrolled right turns);
* the **resolved link indices** ``config/network/link_index_binding.yaml`` (the
  T-01-02 artifact) plus the live ``getControlledLinks`` for each movement's
  incoming/outgoing lanes.

Pressure follows the MPLight definition (state-space.md, research-sumo.md s3):
``pressure(m) = sum(count incoming lanes) - sum(count outgoing lanes)`` using
``lane.getLastStepVehicleNumber`` (vehicle count, NOT halting count).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BINDING_FILE = _REPO_ROOT / "config" / "network" / "link_index_binding.yaml"

# The logical movement/phase spec is the vault SSOT (anti-drift rule: the code
# repo does not vendor a copy - same coupling as scripts/build_network.py).
# Override via the constructor for CI portability.
_VAULT_MOVEMENTS = Path(
    r"C:\Year3\Obsidian\Yuval\30_Projects\smart-traffic-rl\specs\movements.yaml"
)

N_MOVEMENTS = 12
N_PHASES = 8


def load_phase_movements(
    movements_path: str | Path = _VAULT_MOVEMENTS,
) -> dict[int, tuple[int, ...]]:
    """Map each action (0..7) to the canonical movement indices (0..11) it greens.

    A pure read of the ``movements.yaml`` SSOT - no live SUMO connection needed,
    unlike :meth:`Intersection.from_traci` (which also resolves link indices). The
    max-pressure baseline (T-02-05) uses this to sum pressure over each phase's
    served movements; it reads the same ``phases[p]["green"]`` sets the env uses,
    so "which movements a phase serves" has a single definition.

    Free right turns (``controlled: false``) are not in any phase's ``green`` and
    so are excluded here, exactly as in the env's green-state synthesis.
    """
    spec = yaml.safe_load(Path(movements_path).read_text(encoding="utf-8"))
    movement_ids = sorted(spec["movements"], key=lambda m: int(m[1:]))  # M0..M11
    index = {mid: i for i, mid in enumerate(movement_ids)}
    phases = spec["phases"]
    return {int(p): tuple(index[m] for m in phases[p]["green"]) for p in phases}


class Intersection:
    """Movements, phases, pressure, and per-action green-state for one TLS."""

    def __init__(
        self,
        *,
        tls_id: str,
        movement_ids: list[str],
        phase_green: dict[int, list[str]],
        phase_group: dict[int, str],
        free_movements: list[str],
        movement_links: dict[str, list[int]],
        movement_in_lanes: dict[str, list[str]],
        movement_out_lanes: dict[str, list[str]],
        n_links: int,
        yellow_s: int,
        all_red_s: int,
        min_green_s: int,
        max_green_s: int,
    ) -> None:
        self.tls_id = tls_id
        self.movement_ids = movement_ids  # canonical M0..M11 order
        self._phase_green = phase_green
        self._phase_group = phase_group  # action -> "NS" | "EW" (NEMA barrier side)
        self._free = free_movements
        self._links = movement_links
        self._in_lanes = movement_in_lanes
        self._out_lanes = movement_out_lanes
        self._n_links = n_links
        self.yellow_s = yellow_s
        self.all_red_s = all_red_s
        self.min_green_s = min_green_s
        self.max_green_s = max_green_s
        # free (uncontrolled right-turn) links stay green through every transition
        self._free_links: set[int] = set()
        for mid in free_movements:
            self._free_links.update(movement_links[mid])

    # --- construction ---

    @classmethod
    def from_traci(
        cls,
        conn: Any,
        tls_id: str,
        *,
        movements_path: str | Path = _VAULT_MOVEMENTS,
        binding_path: str | Path = _BINDING_FILE,
    ) -> "Intersection":
        """Build the model from the live connection + the spec/binding files."""
        spec = yaml.safe_load(Path(movements_path).read_text(encoding="utf-8"))
        movements = spec["movements"]
        phases = spec["phases"]
        binding = yaml.safe_load(Path(binding_path).read_text(encoding="utf-8"))["link_indices"]

        movement_ids = sorted(movements, key=lambda m: int(m[1:]))  # M0..M11
        free = [m for m in movement_ids if not movements[m]["controlled"]]
        phase_green = {int(p): list(phases[p]["green"]) for p in phases}
        phase_group = {int(p): phases[p]["group"] for p in phases}
        transitions = spec.get("transitions", {})
        yellow_s = int(transitions.get("yellow_s", 3))
        all_red_s = int(transitions.get("all_red_s", 2))
        safety = spec.get("safety", {})
        min_green_s = int(safety.get("min_green_s", 10))
        # max-green == max-red anti-starvation bound (60s, safety-masking.md / decisions.md)
        max_green_s = int(safety.get("max_green_s", safety.get("max_red_s", 60)))

        controlled_links = conn.trafficlight.getControlledLinks(tls_id)
        n_links = len(controlled_links)

        in_lanes: dict[str, list[str]] = {}
        out_lanes: dict[str, list[str]] = {}
        for mid in movement_ids:
            ins: list[str] = []
            outs: list[str] = []
            for idx in binding[mid]:
                for in_lane, out_lane, _via in controlled_links[idx]:
                    ins.append(in_lane)
                    outs.append(out_lane)
            # all links of a movement share one incoming lane
            in_lanes[mid] = sorted(set(ins))
            out_lanes[mid] = outs

        return cls(
            tls_id=tls_id,
            movement_ids=movement_ids,
            phase_green=phase_green,
            phase_group=phase_group,
            free_movements=free,
            movement_links={m: list(binding[m]) for m in movement_ids},
            movement_in_lanes=in_lanes,
            movement_out_lanes=out_lanes,
            n_links=n_links,
            yellow_s=yellow_s,
            all_red_s=all_red_s,
            min_green_s=min_green_s,
            max_green_s=max_green_s,
        )

    @property
    def controlled_in_lanes(self) -> list[str]:
        """Unique incoming lanes of the controlled (signalized) movements, sorted.

        The actuated baseline (T-02-06) places one gap detector per such lane; the
        free right-turn lanes are excluded (they are green in every phase and never
        gate the actuated logic).
        """
        lanes: set[str] = set()
        for mid in self.movement_ids:
            if mid not in self._free:
                lanes.update(self._in_lanes[mid])
        return sorted(lanes)

    def action_for_state(self, ryg: str) -> int | None:
        """Return the action whose green-state equals ``ryg``, or ``None``.

        Used in actuated mode to recover our 0..7 phase one-hot from SUMO's live
        light string (which cycles through an 18-phase actuated program, not our
        action indices). Yellow/all-red strings match no green and return ``None``.
        """
        for action in range(N_PHASES):
            if self.green_state(action) == ryg:
                return action
        return None

    # --- pressure (unnormalized) ---

    def pressures(self, conn: Any) -> np.ndarray:
        """Return the 12 movement pressures (unnormalized), canonical M0..M11 order.

        ``pressure(m) = sum(count incoming lanes) - sum(count outgoing lanes)``.
        """
        count = conn.lane.getLastStepVehicleNumber
        out = np.empty(N_MOVEMENTS, dtype=np.float64)
        for i, mid in enumerate(self.movement_ids):
            inc = sum(count(l) for l in self._in_lanes[mid])
            outc = sum(count(l) for l in self._out_lanes[mid])
            out[i] = inc - outc
        return out

    # --- green-state synthesis for an action ---

    def _links_for(self, movements: list[str]) -> set[int]:
        """Union of TLS link indices for a set of movements."""
        out: set[int] = set()
        for mid in movements:
            out.update(self._links[mid])
        return out

    def green_state(self, action: int) -> str:
        """Return the SUMO RYG string for ``action``'s green phase.

        A link is ``G`` if it belongs to one of the action's green movements or to
        a free (always-permitted) right turn; otherwise ``r``.
        """
        self._check_action(action)
        green = self._links_for(self._phase_green[action]) | self._free_links
        return "".join("G" if i in green else "r" for i in range(self._n_links))

    def is_barrier_crossing(self, prev_action: int, next_action: int) -> bool:
        """True if switching ``prev_action -> next_action`` crosses the NEMA barrier.

        A barrier crossing is any change between an NS-group phase (0-3) and an
        EW-group phase (4-7); it requires the extra all-red clearance (T-02-03).
        """
        self._check_action(prev_action)
        self._check_action(next_action)
        return self._phase_group[prev_action] != self._phase_group[next_action]

    def yellow_state(self, prev_action: int, next_action: int) -> str:
        """RYG string for the yellow tick between ``prev_action`` and ``next_action``.

        Greens that are ending (green in ``prev`` but not ``next``) show ``y``;
        greens continuing through the change stay ``G``; free rights stay ``G``;
        everything else is ``r`` (research-sumo.md s1: yellow replaces just-ended
        greens with ``y``).
        """
        prev_green = self._links_for(self._phase_green[prev_action])
        next_green = self._links_for(self._phase_green[next_action])
        chars = []
        for i in range(self._n_links):
            if i in self._free_links:
                chars.append("G")
            elif i in prev_green and i not in next_green:
                chars.append("y")
            elif i in prev_green:
                chars.append("G")  # green in both -> no need to clear
            else:
                chars.append("r")
        return "".join(chars)

    def all_red_state(self) -> str:
        """RYG string for the all-red clearance tick (free rights stay green)."""
        return "".join("G" if i in self._free_links else "r" for i in range(self._n_links))

    def _check_action(self, action: int) -> None:
        if not 0 <= action < N_PHASES:
            raise ValueError(f"action {action} out of range 0..{N_PHASES - 1}")
