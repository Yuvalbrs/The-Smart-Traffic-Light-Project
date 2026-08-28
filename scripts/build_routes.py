"""T-01-08 - Generate deterministic SUMO route files from the scenarios.

Turns each scenario (``config/scenarios/scn_*.yaml``, T-01-01) into one
``.rou.xml`` per ``(scenario, seed)``. Vehicles arrive as a non-homogeneous
Poisson process whose instantaneous rate is the scenario's per-axis demand
(``AxisDemand.rate_at``); each arrival is assigned a turn from ``turn_split`` and
routed across the intersection.

**Determinism contract** (unity-sumo-integration.md §Determinism, repro B-series):
seed a single ``random.Random(seed)``, emit **explicit ``<vehicle>`` elements**
(never ``<flow>``) sorted by departure time, with fixed float formatting. Same
``(scenario, seed)`` -> byte-identical file. This is what makes the JSONL traces
downstream reproducible.

Geometry (which approach+turn leads where, and which lane carries it) is reused
from :mod:`scripts.build_network` so there is a single source of truth.

Run::

    python -m scripts.build_routes                 # all scenarios x all seeds
    python -m scripts.build_routes --scenario SCN-02 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import traci
from sumolib import checkBinary

from scripts.build_network import (
    GEOMETRY,
    IN_EDGE,
    LANE_FOR_TURN,
    OUT_EDGE,
    _NET_FILE,
    build_net,
)
from src.scenarios.config import AxisDemand, Scenario, load_all, load_scenario

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ROUTES_DIR = _REPO_ROOT / "config" / "routes"

# Each approach belongs to one demand axis; both approaches on an axis emit at
# that axis's rate (N and S each at the N/S rate). ASSUMPTION: the demand table
# is per-approach, not split across the pair - see harsh-review / flag.
_APPROACH_AXIS: dict[str, str] = {"N": "ns", "E": "ew", "S": "ns", "W": "ew"}
_TURN_ORDER = ("left", "through", "right")  # fixed iteration order = determinism


@dataclass(frozen=True)
class Trip:
    """One generated vehicle: when it departs, where it enters, and its turn."""

    depart: float
    approach: str
    turn: str

    @property
    def route_edges(self) -> str:
        """The two-edge route ``"<in_edge> <out_edge>"`` across the junction."""
        dest = GEOMETRY[(self.approach, self.turn)]
        return f"{IN_EDGE[self.approach]} {OUT_EDGE[dest]}"

    @property
    def depart_lane(self) -> int:
        """Lane index this turn must depart from (left=2, through=1, right=0)."""
        return LANE_FOR_TURN[self.turn]


# --- arterial corridor geometry (mirrors config/network/arterial.*, step-1 naming) ---
# Entry points: (first edge, junction reached, heading of travel, demand attribute).
_ARTERIAL_ENTRIES: dict[str, tuple[str, str, str, str]] = {
    "W1": ("w1_c1", "C1", "E", "through"),
    "E2": ("e2_c2", "C2", "W", "through"),
    "N1": ("n1_c1", "C1", "S", "cross_c1"),
    "S1": ("s1_c1", "C1", "N", "cross_c1"),
    "N2": ("n2_c2", "C2", "S", "cross_c2"),
    "S2": ("s2_c2", "C2", "N", "cross_c2"),
}
# (junction, heading, turn) -> (out_edge, next_junction | None, next_heading | None).
# Compass turns: heading E -> left=N, right=S; heading W -> left=S, right=N;
# heading S -> left=E, right=W; heading N -> left=W, right=E. An out-edge onto the
# coupling pair (c1_c2 / c2_c1) continues to the other junction and samples again.
_ARTERIAL_TURNS: dict[tuple[str, str, str], tuple[str, str | None, str | None]] = {
    ("C1", "E", "left"): ("c1_n1", None, None),
    ("C1", "E", "through"): ("c1_c2", "C2", "E"),
    ("C1", "E", "right"): ("c1_s1", None, None),
    ("C1", "W", "left"): ("c1_s1", None, None),
    ("C1", "W", "through"): ("c1_w1", None, None),
    ("C1", "W", "right"): ("c1_n1", None, None),
    ("C1", "S", "left"): ("c1_c2", "C2", "E"),
    ("C1", "S", "through"): ("c1_s1", None, None),
    ("C1", "S", "right"): ("c1_w1", None, None),
    ("C1", "N", "left"): ("c1_w1", None, None),
    ("C1", "N", "through"): ("c1_n1", None, None),
    ("C1", "N", "right"): ("c1_c2", "C2", "E"),
    ("C2", "E", "left"): ("c2_n2", None, None),
    ("C2", "E", "through"): ("c2_e2", None, None),
    ("C2", "E", "right"): ("c2_s2", None, None),
    ("C2", "W", "left"): ("c2_s2", None, None),
    ("C2", "W", "through"): ("c2_c1", "C1", "W"),
    ("C2", "W", "right"): ("c2_n2", None, None),
    ("C2", "S", "left"): ("c2_e2", None, None),
    ("C2", "S", "through"): ("c2_s2", None, None),
    ("C2", "S", "right"): ("c2_c1", "C1", "W"),
    ("C2", "N", "left"): ("c2_c1", "C1", "W"),
    ("C2", "N", "through"): ("c2_n2", None, None),
    ("C2", "N", "right"): ("c2_e2", None, None),
}


@dataclass(frozen=True)
class CorridorTrip:
    """One arterial vehicle: departure, full multi-edge route, and entry lane."""

    depart: float
    edges: tuple[str, ...]
    first_turn: str

    @property
    def route_edges(self) -> str:
        """The space-joined multi-edge route across one or both junctions."""
        return " ".join(self.edges)

    @property
    def depart_lane(self) -> int:
        """Lane index for the FIRST junction's turn (left=2, through=1, right=0)."""
        return LANE_FOR_TURN[self.first_turn]


def generate_corridor_trips(scenario: Scenario, seed: int) -> list[CorridorTrip]:
    """Generate the sorted vehicle list for one arterial ``(scenario, seed)``.

    Same non-homogeneous Poisson thinning as :func:`generate_trips`, but a turn is
    sampled at EACH junction the vehicle crosses: a corridor-through or turned-in
    vehicle that continues onto the coupling edge samples again at the far TLS.
    """
    rng = random.Random(seed)
    trips: list[CorridorTrip] = []
    for entry in sorted(_ARTERIAL_ENTRIES):  # deterministic order
        first_edge, junction, heading, demand_attr = _ARTERIAL_ENTRIES[entry]
        demand: AxisDemand = getattr(scenario, demand_attr)
        lam_max = _lambda_max(demand, scenario.duration_s)
        if lam_max <= 0:
            continue
        t = 0.0
        while True:
            t += rng.expovariate(lam_max / 3600.0)  # vph -> veh/s
            if t >= scenario.duration_s:
                break
            if rng.random() > demand.rate_at(t) / lam_max:
                continue
            edges = [first_edge]
            node: str | None = junction
            head: str | None = heading
            first_turn = ""
            while node is not None:
                turn = _sample_turn(rng, scenario.turn_split)
                first_turn = first_turn or turn
                out_edge, node, head = _ARTERIAL_TURNS[(node, head, turn)]
                edges.append(out_edge)
            trips.append(CorridorTrip(depart=t, edges=tuple(edges), first_turn=first_turn))
    trips.sort(key=lambda tr: (tr.depart, tr.edges))  # depart-sorted, byte-stable ties
    return trips


def _lambda_max(demand: AxisDemand, duration: int) -> float:
    """Upper bound on the arrival rate over ``[0, duration]`` (grid scan, 1 s)."""
    return max(demand.rate_at(float(t)) for t in range(0, duration + 1))


def _sample_turn(rng: random.Random, turn_split: dict[str, float]) -> str:
    """Pick a turn from the split (fixed cumulative order for reproducibility)."""
    r = rng.random()
    cum = 0.0
    for turn in _TURN_ORDER:
        cum += turn_split[turn]
        if r <= cum:
            return turn
    return _TURN_ORDER[-1]  # float-rounding fallback


def generate_trips(scenario: Scenario, seed: int) -> list[Trip]:
    """Generate the sorted vehicle list for one ``(scenario, seed)``.

    Non-homogeneous Poisson via thinning: propose arrivals at the per-approach
    ``lambda_max`` and accept each with probability ``rate_at(t) / lambda_max``.
    """
    rng = random.Random(seed)
    trips: list[Trip] = []
    for approach, axis in sorted(_APPROACH_AXIS.items()):  # deterministic order
        demand = scenario.ns if axis == "ns" else scenario.ew
        lam_max = _lambda_max(demand, scenario.duration_s)
        if lam_max <= 0:
            continue
        t = 0.0
        while True:
            t += rng.expovariate(lam_max / 3600.0)  # vph -> veh/s
            if t >= scenario.duration_s:
                break
            if rng.random() <= demand.rate_at(t) / lam_max:
                trips.append(Trip(depart=t, approach=approach, turn=_sample_turn(rng, scenario.turn_split)))
    # SUMO requires depart-sorted vehicles; tie-break for byte-stable ordering.
    trips.sort(key=lambda tr: (tr.depart, tr.approach, tr.turn))
    return trips


def _route_xml(scenario: Scenario, seed: int, trips: list[Trip]) -> str:
    """Render the ``.rou.xml`` text (fixed formatting -> byte-reproducible)."""
    lines = [
        f"<!-- generated by scripts/build_routes.py (T-01-08): {scenario.id} seed {seed}. "
        "Do not edit by hand; regenerate. -->",
        "<routes>",
        # jmTimegapMinor: minimum time gap a vehicle on a minor (yielding, 'g')
        # link keeps to an approaching foe. SUMO's 1.0s default is far below the
        # ~4.5-6s critical gaps drivers actually accept (HCM) and let free right
        # turns dart in front of through platoons ~1.2s away, ending in a
        # junction collision about once per 100 episodes (decisions.md
        # 2026-08-28). 3.0s keeps the free rights merging without the darting.
        f'    <vType id="{scenario.vehicle_type}" vClass="passenger" jmTimegapMinor="3.0"/>',
    ]
    for i, tr in enumerate(trips):
        lines.append(
            f'    <vehicle id="v{i}" type="{scenario.vehicle_type}" depart="{tr.depart:.2f}" '
            f'departLane="{tr.depart_lane}" departSpeed="max">'
        )
        lines.append(f'        <route edges="{tr.route_edges}"/>')
        lines.append("    </vehicle>")
    lines.append("</routes>")
    return "\n".join(lines) + "\n"


def write_routes(scenario: Scenario, seed: int, out_dir: Path = _ROUTES_DIR) -> Path:
    """Generate and write one route file; returns its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trips = generate_corridor_trips(scenario, seed) if scenario.is_arterial else generate_trips(scenario, seed)
    scn_num = scenario.id.split("-")[1].lower()
    out_path = out_dir / f"scn_{scn_num}_seed_{seed:02d}.rou.xml"
    out_path.write_text(_route_xml(scenario, seed, trips), encoding="utf-8")
    return out_path


def main() -> None:
    """Generate route files for the requested scenarios x seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", help="single scenario id (e.g. SCN-02); default all")
    parser.add_argument("--seed", type=int, help="single seed; default all in the scenario")
    args = parser.parse_args()

    scenarios = [load_scenario(_REPO_ROOT / "config" / "scenarios"
                               / f"scn_{args.scenario.split('-')[1]}.yaml")] if args.scenario else load_all()

    total = 0
    for scenario in scenarios:
        seeds = [args.seed] if args.seed is not None else scenario.seeds
        for seed in seeds:
            path = write_routes(scenario, seed)
            n = sum(1 for _ in path.read_text(encoding="utf-8").splitlines() if "<vehicle " in _)
            print(f"[routes] {scenario.id} seed {seed:02d} -> {path.relative_to(_REPO_ROOT)} ({n} vehicles)")
            total += 1
    print(f"[routes] OK - wrote {total} route file(s).")
    sys.exit(0)


if __name__ == "__main__":
    main()
