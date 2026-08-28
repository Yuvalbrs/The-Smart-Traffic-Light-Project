"""T-01-02 - Build the intersection networks and assert their wiring.

Deterministically compiles a network (``.nod``/``.edg``/``.con`` source via
``netconvert``), then verifies the result against the authoritative movement
spec (``specs/movements.yaml`` in the vault):

1. ``netOffset == (0, 0)`` (required for the SUMO->Unity coordinate transform;
   research-sumo.md R4-3).
2. At EVERY traffic light, the 12 movements M0-M11 are wired correctly - every
   controlled movement maps to exactly one TLS link with the right approach +
   turn, and every free (rightmost) lane carries its through + right links.
   This is the ``getControlledLinks`` assertion that catches a silent
   link-index permutation (research-sumo.md "the silent killer", gotcha 1).

Two networks are defined (the arterial pivot keeps the legacy net intact):

* ``intersection`` - the original single 4-way TLS ``C`` at the origin. Output
  files (``intersection.net.xml``, ``link_index_binding.yaml``) are unchanged
  and byte-stable, so the single-intersection eval path stays valid.
* ``arterial`` - TWO TLS: ``C1`` at the origin (same spot as ``C``) and ``C2``
  at ``+ARTERIAL_SPACING_M`` on x, coupled by the shared edge pair
  ``c1_c2``/``c2_c1`` (each is one junction's out-edge and the other's
  in-edge - the back-pressure coupling). Outputs ``arterial.net.xml`` +
  ``arterial_link_index_binding.yaml`` (multi-TLS schema, keyed by tls_id).

On success it writes the resolved binding yaml (closing the ``link_index: TBD``
open flag B1) and prints the M -> link-index table per TLS.

Run::

    python -m scripts.build_network                # both networks
    python -m scripts.build_network --network intersection
    python -m scripts.build_network --network arterial

Spec authority: ``specs/movements.yaml`` (the SSOT). Note: ``03-simulation.md``
Sections 4.3-4.4 carry a STALE, contradictory movement order - do not use it
(open-items.md A2).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import traci
import yaml
from sumolib import checkBinary

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NET_DIR = _REPO_ROOT / "config" / "network"
_NET_FILE = _NET_DIR / "intersection.net.xml"
_BINDING_FILE = _NET_DIR / "link_index_binding.yaml"

# Authoritative movement spec lives in the Obsidian vault (the SSOT). The code
# repo intentionally does NOT vendor a copy (anti-drift rule, decisions.md);
# override with --movements if the vault moves. See harsh-review note on this
# coupling.
_VAULT_MOVEMENTS = Path(
    r"C:\Year3\Obsidian\Yuval\30_Projects\smart-traffic-rl\specs\movements.yaml"
)

TLS_ID = "C"  # legacy single-intersection TLS

# Centre-to-centre spacing of the two arterial junctions (C1 at the origin,
# C2 at (+ARTERIAL_SPACING_M, 0)). 300 m is a physically sensible urban block
# pair and leaves the 300 m shared edge long enough to hold a realistic queue.
ARTERIAL_SPACING_M = 300.0

# --- Geometry of a '+' intersection (authoritative, derived from approach layout) ---
# For an incoming approach, which outgoing approach each turn leads to. This and
# LANE_FOR_TURN are IDENTICAL at every TLS (all junctions are the same '+'
# layout); only the approach->edge maps below differ per TLS.
GEOMETRY: dict[tuple[str, str], str] = {
    ("N", "left"): "E", ("N", "through"): "S", ("N", "right"): "W",
    ("E", "left"): "S", ("E", "through"): "W", ("E", "right"): "N",
    ("S", "left"): "W", ("S", "through"): "N", ("S", "right"): "E",
    ("W", "left"): "N", ("W", "through"): "E", ("W", "right"): "S",
}
# MPLight allocation in SUMO lane indices (0 = rightmost .. 2 = leftmost):
#   left turn  -> leftmost lane (2);  through -> middle (1);  right -> rightmost (0).
LANE_FOR_TURN: dict[str, int] = {"left": 2, "through": 1, "right": 0}

# Legacy single-intersection approach->edge maps (TLS "C"); kept as the module
# defaults so existing callers (tests, repro reference) are untouched.
IN_EDGE: dict[str, str] = {"N": "n_t", "E": "e_t", "S": "s_t", "W": "w_t"}
OUT_EDGE: dict[str, str] = {"N": "t_n", "E": "t_e", "S": "t_s", "W": "t_w"}


@dataclass(frozen=True)
class TLSSpec:
    """One traffic light's wiring: its id + approach->edge maps.

    ``GEOMETRY`` and ``LANE_FOR_TURN`` are shared by all TLS (same '+' layout);
    only the edge naming differs per junction.
    """

    tls_id: str
    in_edge: dict[str, str]
    out_edge: dict[str, str]


@dataclass(frozen=True)
class NetworkSpec:
    """A buildable network: source/output file stem + its traffic lights."""

    name: str  # stem of the .nod/.edg/.con/.net files in config/network/
    binding_name: str  # binding yaml filename in config/network/
    tls: tuple[TLSSpec, ...]

    @property
    def net_file(self) -> Path:
        return _NET_DIR / f"{self.name}.net.xml"

    @property
    def binding_file(self) -> Path:
        return _NET_DIR / self.binding_name


SINGLE = NetworkSpec(
    name="intersection",
    binding_name="link_index_binding.yaml",
    tls=(TLSSpec(tls_id=TLS_ID, in_edge=IN_EDGE, out_edge=OUT_EDGE),),
)

ARTERIAL = NetworkSpec(
    name="arterial",
    binding_name="arterial_link_index_binding.yaml",
    tls=(
        TLSSpec(
            tls_id="C1",
            in_edge={"N": "n1_c1", "E": "c2_c1", "S": "s1_c1", "W": "w1_c1"},
            out_edge={"N": "c1_n1", "E": "c1_c2", "S": "c1_s1", "W": "c1_w1"},
        ),
        TLSSpec(
            tls_id="C2",
            in_edge={"N": "n2_c2", "E": "e2_c2", "S": "s2_c2", "W": "c1_c2"},
            out_edge={"N": "c2_n2", "E": "c2_e2", "S": "c2_s2", "W": "c2_c1"},
        ),
    ),
)

NETWORKS: dict[str, NetworkSpec] = {"intersection": SINGLE, "arterial": ARTERIAL}


def build_net(network: NetworkSpec = SINGLE, movements_path: Path = _VAULT_MOVEMENTS) -> None:
    """Compile ``network`` with ``netconvert`` (normalization disabled so the
    origin junction stays at (0,0) and ``netOffset`` is ``(0, 0)``).

    TWO passes. Pass 1 lets netconvert auto-generate a TLS program; pass 2
    rebuilds with the REAL 8-phase program (``{name}.tll.xml``, synthesized from
    movements.yaml). This matters because netconvert bakes the junction's
    right-of-way ``response`` matrix from the phases it is given: with its auto
    program the free rights and the shared-lane throughs are never green
    together, so their merge precedence came out arbitrary - two of the four
    free rights held priority over the throughs they merge with, and the
    runtime ``g`` could not make them yield (the response matrix rules).
    Baking the real phases makes every free right yield to every foe
    (decisions.md 2026-08-28).

    The output is made byte-stable across rebuilds by normalizing the only volatile
    field netconvert writes - the ``generated on <timestamp>`` line in the header
    comment - so regenerating the net (every test run / data-gen run does) never
    dirties git with a meaningless timestamp diff.
    """
    cmd = [
        checkBinary("netconvert"),
        "--node-files", str(_NET_DIR / f"{network.name}.nod.xml"),
        "--edge-files", str(_NET_DIR / f"{network.name}.edg.xml"),
        "--connection-files", str(_NET_DIR / f"{network.name}.con.xml"),
        "--output-file", str(network.net_file),
        "--no-turnarounds", "true",
        "--tls.default-type", "static",
        "--offset.disable-normalization", "true",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    _normalize_generated_timestamp(network.net_file)
    if Path(movements_path).exists():
        tll = _write_tll(network, movements_path)
        subprocess.run(
            cmd + ["--tllogic-files", str(tll)], check=True, capture_output=True, text=True
        )
        _normalize_generated_timestamp(network.net_file)


def _write_tll(network: NetworkSpec, movements_path: Path) -> Path:
    """Synthesize ``{name}.tll.xml``: the real 8-phase program per TLS.

    States use the runtime semantics (``Intersection.green_state``): phase
    movements ``G``, free rights ``g`` in every phase, all else ``r``. Durations
    are placeholders - the env overrides states at runtime; netconvert only
    uses the states to derive right-of-way.
    """
    spec = yaml.safe_load(Path(movements_path).read_text(encoding="utf-8"))
    movements, phases = spec["movements"], spec["phases"]
    lines = ["<additional>"]
    for t in network.tls:
        binding = assert_wiring(
            movements,
            observed_links(t.tls_id, network.net_file),
            in_edge=t.in_edge,
            out_edge=t.out_edge,
        )
        n_links = 1 + max(i for idxs in binding.values() for i in idxs)
        free = {
            i for mid, m in movements.items() if not m["controlled"] for i in binding[mid]
        }
        lines.append(f'    <tlLogic id="{t.tls_id}" type="static" programID="0" offset="0">')
        for p in sorted(phases, key=lambda k: int(k)):
            green = {i for mid in phases[p]["green"] for i in binding[mid]}
            state = "".join(
                "G" if i in green else "g" if i in free else "r" for i in range(n_links)
            )
            lines.append(f'        <phase duration="10" state="{state}"/>')
        lines.append("    </tlLogic>")
    lines.append("</additional>")
    path = _NET_DIR / f"{network.name}.tll.xml"
    path.write_text(
        "<!-- Generated by scripts/build_network.py - do not edit by hand. -->\n"
        + "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )
    return path


def assert_free_links_yield(network: NetworkSpec, movements: dict, bindings: dict) -> None:
    """Assert every free (right-turn) link's ``response`` covers ALL its foes.

    Free rights are ``g`` in every phase, so anything they conflict with must
    outrank them. A free link whose response bit is 0 for a foe holds silent
    priority over it - the defect class behind the merge collisions.
    """
    root = ET.parse(network.net_file).getroot()
    errors: list[str] = []
    for t in network.tls:
        free = {
            i
            for mid, m in movements.items()
            if not m["controlled"]
            for i in bindings[t.tls_id][mid]
        }
        junction = root.find(f".//junction[@id='{t.tls_id}']")
        for req in junction.findall("request"):
            idx = int(req.get("index"))
            if idx not in free:
                continue
            foes, resp = reversed(req.get("foes")), reversed(req.get("response"))
            for k, (f, r) in enumerate(zip(foes, resp)):
                if f == "1" and r != "1":
                    errors.append(f"{t.tls_id}: free link {idx} does not yield to foe link {k}")
    if errors:
        raise AssertionError(
            "free rights hold right-of-way over foes:\n  - " + "\n  - ".join(errors)
        )


def _normalize_generated_timestamp(net_file: Path) -> None:
    """Replace the netconvert ``generated on <timestamp>`` date with a fixed token."""
    text = net_file.read_text(encoding="utf-8")
    normalized = re.sub(
        r"(<!-- generated on ).*?( by Eclipse SUMO netconvert)",
        r"\1(date omitted for reproducibility)\2",
        text,
        count=1,
    )
    if normalized != text:
        net_file.write_text(normalized, encoding="utf-8")


def assert_net_offset(net_file: Path = _NET_FILE) -> None:
    """Assert the compiled net has ``netOffset == (0, 0)``.

    Raises
    ------
    AssertionError
        If the ``<location>`` element's ``netOffset`` is not the origin.
    """
    location = ET.parse(net_file).getroot().find("location")
    if location is None:
        raise AssertionError("net file has no <location> element")
    offset = tuple(float(v) for v in location.attrib["netOffset"].split(","))
    if offset != (0.0, 0.0):
        raise AssertionError(f"netOffset is {offset}, expected (0.0, 0.0)")


def observed_links(
    tls_id: str = TLS_ID, net_file: Path = _NET_FILE
) -> list[tuple[int, str, str]]:
    """Return one TLS's controlled links as ``(link_index, in_lane, out_edge)``.

    Starts SUMO headless on the net alone (no routes needed) and reads
    ``getControlledLinks``.
    """
    traci.start([checkBinary("sumo"), "-n", str(net_file), "--no-step-log", "true"])
    try:
        rows: list[tuple[int, str, str]] = []
        for idx, conns in enumerate(traci.trafficlight.getControlledLinks(tls_id)):
            for in_lane, out_lane, _via in conns:
                rows.append((idx, in_lane, out_lane.rsplit("_", 1)[0]))
        return rows
    finally:
        traci.close()


def assert_wiring(
    movements: dict,
    links: list[tuple[int, str, str]],
    *,
    in_edge: dict[str, str] = IN_EDGE,
    out_edge: dict[str, str] = OUT_EDGE,
) -> dict[str, list[int]]:
    """Assert every movement is wired per ``movements.yaml`` and return the
    resolved ``{movement_id: [link_index, ...]}`` binding.

    Parameters
    ----------
    movements : dict
        The ``movements`` mapping loaded from ``movements.yaml``.
    links : list of tuple
        ``observed_links()`` output for ONE TLS.
    in_edge, out_edge : dict, optional
        The TLS's approach->edge maps (default: the legacy single-intersection
        maps for TLS ``C``); pass a :class:`TLSSpec`'s maps for other junctions.

    Raises
    ------
    AssertionError
        With ALL mismatches collected, if any movement is mis-wired.
    """
    binding: dict[str, list[int]] = {}
    errors: list[str] = []

    for mid, spec in movements.items():
        approach, turn, controlled = spec["approach"], spec["turn"], spec["controlled"]
        edge_prefix = f"{in_edge[approach]}_"
        want_out = out_edge[GEOMETRY[(approach, turn)]]

        # A movement owns EVERY link from its approach edge to its geometric exit,
        # regardless of which lane carries it. The shared rightmost lane emits TWO
        # links (right + through); its through link belongs to the THROUGH movement.
        # Binding it to the free right made through-on-red permanently permitted for
        # the shared lane in every phase - the deeper half of the gridlock artefact
        # (decisions.md 2026-08-28).
        hits = sorted(
            i for (i, lane, out) in links if lane.startswith(edge_prefix) and out == want_out
        )
        binding[mid] = hits

        if turn in ("left", "right"):
            # exactly one link, from the turn's designated lane
            want_lane = f"{in_edge[approach]}_{LANE_FOR_TURN[turn]}"
            if len(hits) != 1 or not any(
                lane == want_lane for (i, lane, _o) in links if i in hits
            ):
                errors.append(
                    f"{mid} ({approach} {turn}): expected exactly 1 link "
                    f"{want_lane}->{want_out}, found {hits}"
                )
        else:  # through: the dedicated middle lane, plus the shared rightmost lane
            lanes = {lane for (i, lane, _o) in links if i in hits}
            want_lanes = {
                f"{in_edge[approach]}_{LANE_FOR_TURN['through']}",
                f"{in_edge[approach]}_{LANE_FOR_TURN['right']}",
            }
            if lanes != want_lanes:
                errors.append(
                    f"{mid} ({approach} {turn}): expected links from {sorted(want_lanes)} "
                    f"->{want_out}, found lanes {sorted(lanes)} (links {hits})"
                )
        if not controlled:
            # lane-allocation check: the shared rightmost lane must reach BOTH exits
            shared = f"{in_edge[approach]}_{LANE_FOR_TURN['right']}"
            outs = {out for (_i, lane, out) in links if lane == shared}
            want_through = out_edge[GEOMETRY[(approach, "through")]]
            if not {want_through, want_out} <= outs:
                errors.append(
                    f"{mid} ({approach} {turn}, free): lane {shared} should reach "
                    f"{{{want_through}, {want_out}}}, reaches {outs}"
                )

    # The 12 movements must partition the TLS's links: every link claimed once.
    claimed = sorted(i for idxs in binding.values() for i in idxs)
    all_links = sorted({i for (i, _l, _o) in links})
    if claimed != all_links:
        errors.append(
            f"binding does not partition the links: claimed {claimed}, net has {all_links}"
        )

    # Every incoming lane must exist (12 movements -> 12 distinct incoming lanes).
    seen_lanes = {lane for (_i, lane, _o) in links}
    for mid, spec in movements.items():
        lane = f"{in_edge[spec['approach']]}_{LANE_FOR_TURN[spec['turn']]}"
        if lane not in seen_lanes:
            errors.append(f"{mid}: incoming lane {lane} not present in the net")

    if errors:
        raise AssertionError("network wiring does not match movements.yaml:\n  - " + "\n  - ".join(errors))
    return binding


def resolve_bindings(
    network: NetworkSpec, movements: dict
) -> dict[str, dict[str, list[int]]]:
    """Assert wiring at every TLS of ``network`` and return per-TLS bindings.

    Returns
    -------
    dict
        ``{tls_id: {movement_id: [link_index, ...]}}`` in TLS order.
    """
    return {
        t.tls_id: assert_wiring(
            movements,
            observed_links(t.tls_id, network.net_file),
            in_edge=t.in_edge,
            out_edge=t.out_edge,
        )
        for t in network.tls
    }


def write_binding_file(
    network: NetworkSpec, bindings: dict[str, dict[str, list[int]]]
) -> Path:
    """Write ``network``'s resolved binding yaml and return its path.

    Single-TLS networks keep the legacy flat schema
    (``{tls_id: C, link_indices: {...}}`` - byte-stable with the pre-arterial
    artifact); multi-TLS networks use the per-TLS schema
    (``{tls_id: {link_indices: {...}}, ...}``).
    """
    if len(network.tls) == 1:
        tls_id = network.tls[0].tls_id
        payload: dict = {"tls_id": tls_id, "link_indices": bindings[tls_id]}
    else:
        payload = {t.tls_id: {"link_indices": bindings[t.tls_id]} for t in network.tls}
    network.binding_file.write_text(
        "# Generated by scripts/build_network.py (T-01-02) - do not edit by hand.\n"
        "# Resolves the link_index: TBD flag in specs/movements.yaml (open-flag B1).\n"
        + yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return network.binding_file


def _build_and_bind(network: NetworkSpec, movements: dict) -> None:
    """Build one network, run all assertions, write its binding, print the table."""
    print(f"[net] building {network.net_file.name} via netconvert ...")
    build_net(network)
    assert_net_offset(network.net_file)
    print("[net] netOffset == (0,0) OK")

    bindings = resolve_bindings(network, movements)
    print(
        f"[net] all {len(movements)} movements wired correctly vs movements.yaml "
        f"at {', '.join(t.tls_id for t in network.tls)} OK"
    )
    assert_free_links_yield(network, movements, bindings)
    print("[net] free rights yield to all foes (response matrix) OK")

    binding_file = write_binding_file(network, bindings)
    print(f"[net] wrote binding -> {binding_file.relative_to(_REPO_ROOT)}")

    for t in network.tls:
        print(f"\n  [{t.tls_id}] movement  approach turn      controlled  link_index(es)")
        for mid, spec_m in movements.items():
            print(
                f"  {mid:<8s}  {spec_m['approach']:<8s} {spec_m['turn']:<9s} "
                f"{str(spec_m['controlled']):<10s}  {bindings[t.tls_id][mid]}"
            )


def main() -> None:
    """Build the net(s), run all assertions, write the bindings, exit 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--movements", type=Path, default=_VAULT_MOVEMENTS,
        help="path to the authoritative movements.yaml (default: vault SSOT)",
    )
    parser.add_argument(
        "--network", choices=[*NETWORKS, "all"], default="all",
        help="which network to build (default: all)",
    )
    args = parser.parse_args()

    spec = yaml.safe_load(args.movements.read_text(encoding="utf-8"))
    movements = spec["movements"]

    targets = list(NETWORKS.values()) if args.network == "all" else [NETWORKS[args.network]]
    for network in targets:
        _build_and_bind(network, movements)

    print("\n[net] OK - T-01-02 network(s) built and asserted.")
    sys.exit(0)


if __name__ == "__main__":
    main()
