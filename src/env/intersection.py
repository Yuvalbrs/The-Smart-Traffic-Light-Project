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
# The movement + NEMA phase spec. Authored in the Obsidian vault (the SSOT for the written
# design) and SHIPPED HERE, because the runtime must not depend on a folder outside the repo.
#
# It used to point at C:\Year3\Obsidian\... directly. build_net() is called on EVERY live
# session start, so on any machine but the author's - a clone, a marker's laptop, CI - every
# episode died with FileNotFoundError before it began, for every controller including the
# baselines that need no checkpoint. tests/test_network.py has always skipped when the vault was
# absent, so nothing caught it. tests/test_movements_spec.py now proves the two copies agree.
MOVEMENTS_SPEC = _REPO_ROOT / "config" / "movements.yaml"

N_MOVEMENTS = 12
N_PHASES = 8


def load_phase_movements(
    movements_path: str | Path = MOVEMENTS_SPEC,
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



# --- observation squashing (single definition; see decisions.md 2026-08-30) ----------
# The observation used to clip pressure to +/-10 then divide by 10. Measured live, that
# pinned 26.9% of the 12 dims at exactly +/-1.0 (33.3% in the worst-10% reward states)
# with |pressure| reaching 36, making congested and gridlocked states OBSERVATIONALLY
# IDENTICAL. tanh is monotone everywhere so ordering survives at every magnitude, and it
# stays strictly inside (-1, 1) so the Box(-1, 1) observation contract still holds.
# These live here, beside `pressures()`, so the env and the max-pressure baseline cannot
# drift apart about what an observation means.
PRESSURE_SCALE = 20.0  # tanh(10/20)=0.46, tanh(20/20)=0.76, tanh(36/20)=0.95
_TANH_LIMIT = 0.999999  # float32-safe: arctanh(0.999999)*20 ~= 145 veh, well past reality


def squash_pressures(pressures: "np.ndarray") -> "np.ndarray":
    """Map raw movement pressures into (-1, 1), monotonically and without clipping."""
    return np.tanh(np.asarray(pressures, dtype=np.float64) / PRESSURE_SCALE)


def unsquash_pressures(squashed: "np.ndarray") -> "np.ndarray":
    """Exact inverse of :func:`squash_pressures`.

    Needed because tanh is monotone but NONLINEAR: summing squashed values across a
    phase's movements does not preserve the ordering of the summed raw pressures, so a
    max-pressure controller reading the observation must invert the transform first.
    """
    z = np.clip(np.asarray(squashed, dtype=np.float64), -_TANH_LIMIT, _TANH_LIMIT)
    return np.arctanh(z) * PRESSURE_SCALE

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
        max_red_s: int = 120,
    ) -> None:
        self.tls_id = tls_id
        self.movement_ids = movement_ids  # canonical M0..M11 order
        self._phase_green = phase_green
        self._phase_group = phase_group  # action -> "NS" | "EW" (NEMA barrier side)
        self._free = free_movements
        self._links = movement_links
        self._in_lanes = movement_in_lanes
        self._out_lanes = {m: sorted(set(ls)) for m, ls in movement_out_lanes.items()}
        # --- shared-lane apportionment (added 2026-08-30, decisions.md) ---------------
        # pressure(m) = sum(incoming) - sum(outgoing) presupposes that the movements
        # PARTITION the lanes. The locked MPLight allocation breaks that presupposition:
        # the rightmost lane of each approach is "through + right", so it is claimed by
        # BOTH the through movement and the free right (n_t_0 by M1 and M2, and likewise
        # e_t_0, s_t_0, w_t_0). Summing over 12 movements therefore traversed 16
        # lane-counts for 12 physical lanes and every vehicle in a shared lane was
        # weighted 2x in the reward - measured at a median 30.3% and a max 64.0% of
        # |reward| on SCN-02. The exit side had the SAME defect with the opposite sign
        # (t_s_0 is claimed by M1 and M11, etc.) and was additionally not deduplicated,
        # so a movement with two links onto one exit lane counted it twice again.
        # Weighting each lane by 1/(movements claiming it) restores the partition:
        # sum_m incoming(m) is once again the vehicle count on the approach.
        def _weights(lane_map: dict[str, list[str]]) -> dict[str, list[float]]:
            claims: dict[str, int] = {}
            for lanes in lane_map.values():
                for lane in lanes:
                    claims[lane] = claims.get(lane, 0) + 1
            return {m: [1.0 / claims[l] for l in lanes] for m, lanes in lane_map.items()}

        self._in_w = _weights(self._in_lanes)
        self._out_w = _weights(self._out_lanes)
        self._n_links = n_links
        self.yellow_s = yellow_s
        self.all_red_s = all_red_s
        self.min_green_s = min_green_s
        self.max_green_s = max_green_s
        # Anti-starvation bound: the longest a controlled movement may stay red. Distinct
        # from max_green_s (which bounds the CURRENT phase); conflating the two left the
        # bound unenforced - see decisions.md 2026-08-28.
        self.max_red_s = max_red_s
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
        movements_path: str | Path = MOVEMENTS_SPEC,
        binding_path: str | Path = _BINDING_FILE,
    ) -> "Intersection":
        """Build the model from the live connection + the spec/binding files.

        Supports both binding schemas: the multi-TLS arterial artifact
        (``{tls_id: {link_indices: {...}}, ...}``, looked up by ``tls_id``) and
        the legacy single-TLS one (``{tls_id: C, link_indices: {...}}``).
        """
        spec = yaml.safe_load(Path(movements_path).read_text(encoding="utf-8"))
        movements = spec["movements"]
        phases = spec["phases"]
        raw_binding = yaml.safe_load(Path(binding_path).read_text(encoding="utf-8"))
        if tls_id in raw_binding:  # multi-TLS schema (arterial)
            binding = raw_binding[tls_id]["link_indices"]
        elif "link_indices" in raw_binding:  # legacy single-TLS schema
            binding = raw_binding["link_indices"]
        else:
            raise ValueError(
                f"binding file {binding_path} has no entry for TLS {tls_id!r} "
                "and no legacy 'link_indices' key"
            )

        movement_ids = sorted(movements, key=lambda m: int(m[1:]))  # M0..M11
        free = [m for m in movement_ids if not movements[m]["controlled"]]
        phase_green = {int(p): list(phases[p]["green"]) for p in phases}
        phase_group = {int(p): phases[p]["group"] for p in phases}
        transitions = spec.get("transitions", {})
        yellow_s = int(transitions.get("yellow_s", 3))
        all_red_s = int(transitions.get("all_red_s", 2))
        safety = spec.get("safety", {})
        min_green_s = int(safety.get("min_green_s", 10))
        # max_green_s bounds the CURRENT phase; max_red_s bounds how long any controlled
        # movement may stay unserved. They are DIFFERENT bounds and 60/60 is infeasible
        # (see movements.yaml safety: the arithmetic puts worst-case red at 196s).
        # Amended 2026-08-30 to 60 / 120. These fallbacks are last-resort only - the
        # vault movements.yaml is the SSOT and is what every real build reads.
        max_green_s = int(safety.get("max_green_s", 60))
        max_red_s = int(safety.get("max_red_s", 120))

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
            max_red_s=max_red_s,
        )

    @property
    def free_movements(self) -> list[str]:
        """The free (never-signalized) right-turn movement ids."""
        return list(self._free)

    def phase_green(self, action: int) -> list[str]:
        """The movement ids an action gives protected green."""
        self._check_action(action)
        return list(self._phase_green[action])

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

        ``pressure(m) = sum(count incoming) - sum(count outgoing)``, with each lane
        APPORTIONED by ``1 / (movements claiming it)`` so the shared through+right lane
        is not counted twice (see the apportionment note in ``__init__``).
        """
        count = conn.lane.getLastStepVehicleNumber
        out = np.empty(N_MOVEMENTS, dtype=np.float64)
        for i, mid in enumerate(self.movement_ids):
            inc = sum(count(l) * w for l, w in zip(self._in_lanes[mid], self._in_w[mid]))
            outc = sum(count(l) * w for l, w in zip(self._out_lanes[mid], self._out_w[mid]))
            out[i] = inc - outc
        return out

    def movement_queues(self, conn: Any) -> np.ndarray:
        """Return the 12 per-movement queue lengths (halting count), M0..M11 order.

        Queue = ``lane.getLastStepHaltingNumber`` summed over a movement's incoming
        lanes (halting = speed < 0.1 m/s, matching the KPI threshold). This is the
        LSTM forecast target / input feature - distinct from pressure's vehicle
        *count* (research-sumo.md "queue vs count" gotcha).
        """
        halting = conn.lane.getLastStepHaltingNumber
        out = np.empty(N_MOVEMENTS, dtype=np.float64)
        for i, mid in enumerate(self.movement_ids):
            out[i] = sum(halting(l) * w for l, w in zip(self._in_lanes[mid], self._in_w[mid]))
        return out

    def movement_counts(self, conn: Any) -> np.ndarray:
        """Return the 12 per-movement incoming vehicle counts, M0..M11 order.

        Count = ``lane.getLastStepVehicleNumber`` summed over a movement's incoming
        lanes (an LSTM input feature alongside the queue).
        """
        count = conn.lane.getLastStepVehicleNumber
        out = np.empty(N_MOVEMENTS, dtype=np.float64)
        for i, mid in enumerate(self.movement_ids):
            out[i] = sum(count(l) * w for l, w in zip(self._in_lanes[mid], self._in_w[mid]))
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

        A link in one of the action's green movements is ``G`` - green *major*, holding right of
        way. A free (always-permitted) right turn is ``g`` - green *minor*, permitted only when
        the way is clear.

        That distinction is load-bearing, not cosmetic. The four free rights are green in EVERY
        phase, so marking them ``G`` gave them priority over the movements they cross, and SUMO
        drove them straight through conflicting traffic. Measured on SCN-02 seed 7000: 11 junction
        collisions with ``G``, 0 with ``g``. On SCN-05 across the five held-out eval seeds it was
        the difference between plain DQN gridlocking 4/5 episodes and 0/5 (see decisions.md
        2026-08-28). A permitted right turn yields; ``g`` is how SUMO says so.
        """
        self._check_action(action)
        green = self._links_for(self._phase_green[action])
        return "".join(
            "G" if i in green else "g" if i in self._free_links else "r"
            for i in range(self._n_links)
        )

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
                chars.append("g")  # permitted, must yield - see green_state
            elif i in prev_green and i not in next_green:
                chars.append("y")
            elif i in prev_green:
                chars.append("G")  # green in both -> no need to clear
            else:
                chars.append("r")
        return "".join(chars)

    def all_red_state(self) -> str:
        """RYG string for the all-red clearance tick (free rights stay permitted, yielding)."""
        return "".join("g" if i in self._free_links else "r" for i in range(self._n_links))

    def _check_action(self, action: int) -> None:
        if not 0 <= action < N_PHASES:
            raise ValueError(f"action {action} out of range 0..{N_PHASES - 1}")
