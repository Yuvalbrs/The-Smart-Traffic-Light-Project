"""T-01-04 - Build a schema-1.1.0 ``sim_frame`` envelope from a live TraCI run.

A ``sim_frame`` carries, per simulated second, every vehicle (position, kinematics
and the **movement** M0-M11 it is executing) plus the intersection ``signal``
block (data-schema.md s4). The movement of a vehicle is decided by its current
lane: at this intersection each approach lane maps to exactly one movement
(leftmost -> left, middle -> through, rightmost -> the free through+right
movement), so the lane id alone resolves it.

``MovementResolver`` builds the ``lane -> movement`` map at episode start from
``config/network/link_index_binding.yaml`` (the T-01-02 artifact) composed with
the live ``getControlledLinks`` - so it needs no vault file at runtime and stays
correct if the link indices ever change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.schema.validate import SCHEMA_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BINDING_FILE = _REPO_ROOT / "config" / "network" / "link_index_binding.yaml"

# SUMO getRedYellowGreenState chars -> our 3 colors. Anything not green/yellow
# (r, s=off-red, u, etc.) is treated as red (data-schema.md signal_colors domain).
_CHAR_COLOR = {"G": "green", "g": "green", "y": "yellow", "Y": "yellow"}


def _char_to_color(char: str) -> str:
    """Map one ``getRedYellowGreenState`` char to ``green``/``yellow``/``red``."""
    return _CHAR_COLOR.get(char, "red")


class MovementResolver:
    """Resolve a lane id to its movement, and a RYG string to per-movement colors.

    Construct via :meth:`from_traci` after the SUMO connection is up; the maps are
    fixed for the lifetime of the network.
    """

    def __init__(
        self, lane_to_movement: dict[str, str], movement_to_links: dict[str, list[int]]
    ) -> None:
        self._lane_to_movement = lane_to_movement
        self._movement_to_links = movement_to_links

    @classmethod
    def from_traci(
        cls, conn: Any, tls_id: str, *, binding_path: str | Path = _BINDING_FILE
    ) -> "MovementResolver":
        """Build the resolver from the link-index binding + live controlled links.

        Parameters
        ----------
        conn : Any
            A TraCI connection (the ``traci`` module or a ``Connection``).
        tls_id : str
            The traffic-light id (``"C"``).
        binding_path : str or Path, optional
            Path to ``link_index_binding.yaml``. Defaults to the repo artifact.
        """
        binding = yaml.safe_load(Path(binding_path).read_text(encoding="utf-8"))
        movement_to_links: dict[str, list[int]] = {
            mid: list(idxs) for mid, idxs in binding["link_indices"].items()
        }

        # link index -> incoming lane (a movement's links all share one in-lane).
        idx_to_lane: dict[int, str] = {}
        for idx, conns in enumerate(conn.trafficlight.getControlledLinks(tls_id)):
            for in_lane, _out_lane, _via in conns:
                idx_to_lane[idx] = in_lane

        lane_to_movement: dict[str, str] = {}
        for mid, idxs in movement_to_links.items():
            for idx in idxs:
                lane = idx_to_lane.get(idx)
                if lane is not None:
                    lane_to_movement[lane] = mid
        return cls(lane_to_movement, movement_to_links)

    def for_lane(self, lane_id: str) -> str | None:
        """Return the movement M0-M11 for ``lane_id``, or ``None`` off-approach.

        Internal junction lanes (``:C_*``) and outgoing edges have no movement.
        """
        return self._lane_to_movement.get(lane_id)

    def signal_colors(self, ryg_state: str) -> dict[str, str]:
        """Map a ``getRedYellowGreenState`` string to ``{Mk: color}`` for M0-M11.

        A movement with several links (the free through+right lanes) takes the
        color of its first link; they share a state under any valid program.
        """
        colors: dict[str, str] = {}
        for mid, idxs in self._movement_to_links.items():
            chars = [ryg_state[i] for i in idxs if i < len(ryg_state)]
            colors[mid] = _char_to_color(chars[0]) if chars else "red"
        return colors


def build_sim_frame(
    conn: Any,
    tls_id: str,
    *,
    seq: int,
    episode_id: int,
    phase_index: int,
    resolver: MovementResolver,
    sim_time: float | None = None,
) -> dict[str, Any]:
    """Assemble one ``sim_frame`` envelope from the live simulation state.

    Parameters
    ----------
    conn : Any
        A TraCI connection (the ``traci`` module or a ``Connection``).
    tls_id : str
        The traffic-light id (``"C"``).
    seq : int
        Monotonic frame sequence number (lets clients detect dropped frames).
    episode_id : int
        The episode this frame belongs to.
    phase_index : int
        The agent's last action (NEMA phase 0-7) - NOT SUMO's internal phase
        index, which points at transition phases during yellow.
    resolver : MovementResolver
        Built once per episode via :meth:`MovementResolver.from_traci`.
    sim_time : float, optional
        SUMO time for this frame; read from ``conn`` if omitted.

    Returns
    -------
    dict
        A schema-1.1.0 ``sim_frame`` envelope (passes ``validate_envelope``).
    """
    if sim_time is None:
        sim_time = conn.simulation.getTime()

    vehicles: list[dict[str, Any]] = []
    for vid in conn.vehicle.getIDList():
        x, y = conn.vehicle.getPosition(vid)
        lane = conn.vehicle.getLaneID(vid)
        vehicles.append(
            {
                "id": vid,
                "x": x,
                "y": y,
                "angle": conn.vehicle.getAngle(vid),
                "speed": conn.vehicle.getSpeed(vid),
                "lane": lane,
                "type": conn.vehicle.getTypeID(vid),
                "movement_id": resolver.for_lane(lane),
            }
        )

    ryg = conn.trafficlight.getRedYellowGreenState(tls_id)
    signal = {
        "phase_index": phase_index,
        "signal_colors": resolver.signal_colors(ryg),
        "sumo_state": ryg,
        "phase_remaining_s": conn.trafficlight.getNextSwitch(tls_id) - sim_time,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "type": "sim_frame",
        "sim_time": sim_time,
        "seq": seq,
        # yellow present on the wire == a transition tick (so Unity shows the change).
        "transition": "y" in ryg.lower(),
        "episode_id": episode_id,
        "payload": {"vehicles": vehicles, "signal": signal},
    }
