"""T-01-02 - Build the real 4-way intersection network and assert its wiring.

Deterministically compiles ``config/network/intersection.net.xml`` from the
``.nod``/``.edg``/``.con`` source via ``netconvert``, then verifies the result
against the authoritative movement spec (``specs/movements.yaml`` in the vault):

1. ``netOffset == (0, 0)`` (required for the SUMO->Unity coordinate transform;
   research-sumo.md R4-3).
2. The 12 movements M0-M11 are wired correctly - every controlled movement maps
   to exactly one TLS link with the right approach + turn, and every free
   (rightmost) lane carries its through + right links. This is the
   ``getControlledLinks`` assertion that catches a silent link-index permutation
   (research-sumo.md "the silent killer", gotcha 1).

On success it writes the resolved binding to
``config/network/link_index_binding.yaml`` (closing the ``link_index: TBD`` open
flag B1) and prints the M -> link-index table.

Run::

    python -m scripts.build_network

Spec authority: ``specs/movements.yaml`` (the SSOT). Note: ``03-simulation.md``
Sections 4.3-4.4 carry a STALE, contradictory movement order - do not use it
(open-items.md A2).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
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

TLS_ID = "C"

# --- Geometry of a '+' intersection (authoritative, derived from approach layout) ---
# For an incoming approach, which outgoing approach each turn leads to.
GEOMETRY: dict[tuple[str, str], str] = {
    ("N", "left"): "E", ("N", "through"): "S", ("N", "right"): "W",
    ("E", "left"): "S", ("E", "through"): "W", ("E", "right"): "N",
    ("S", "left"): "W", ("S", "through"): "N", ("S", "right"): "E",
    ("W", "left"): "N", ("W", "through"): "E", ("W", "right"): "S",
}
IN_EDGE: dict[str, str] = {"N": "n_t", "E": "e_t", "S": "s_t", "W": "w_t"}
OUT_EDGE: dict[str, str] = {"N": "t_n", "E": "t_e", "S": "t_s", "W": "t_w"}
# MPLight allocation in SUMO lane indices (0 = rightmost .. 2 = leftmost):
#   left turn  -> leftmost lane (2);  through -> middle (1);  right -> rightmost (0).
LANE_FOR_TURN: dict[str, int] = {"left": 2, "through": 1, "right": 0}


def build_net() -> None:
    """Compile the network with ``netconvert`` (normalization disabled so the
    centre node stays at the origin and ``netOffset`` is ``(0, 0)``)."""
    cmd = [
        checkBinary("netconvert"),
        "--node-files", str(_NET_DIR / "intersection.nod.xml"),
        "--edge-files", str(_NET_DIR / "intersection.edg.xml"),
        "--connection-files", str(_NET_DIR / "intersection.con.xml"),
        "--output-file", str(_NET_FILE),
        "--no-turnarounds", "true",
        "--tls.default-type", "static",
        "--offset.disable-normalization", "true",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def assert_net_offset() -> None:
    """Assert the compiled net has ``netOffset == (0, 0)``.

    Raises
    ------
    AssertionError
        If the ``<location>`` element's ``netOffset`` is not the origin.
    """
    location = ET.parse(_NET_FILE).getroot().find("location")
    if location is None:
        raise AssertionError("net file has no <location> element")
    offset = tuple(float(v) for v in location.attrib["netOffset"].split(","))
    if offset != (0.0, 0.0):
        raise AssertionError(f"netOffset is {offset}, expected (0.0, 0.0)")


def observed_links() -> list[tuple[int, str, str]]:
    """Return the TLS's controlled links as ``(link_index, in_lane, out_edge)``.

    Starts SUMO headless on the net alone (no routes needed) and reads
    ``getControlledLinks``.
    """
    traci.start([checkBinary("sumo"), "-n", str(_NET_FILE), "--no-step-log", "true"])
    try:
        rows: list[tuple[int, str, str]] = []
        for idx, conns in enumerate(traci.trafficlight.getControlledLinks(TLS_ID)):
            for in_lane, out_lane, _via in conns:
                rows.append((idx, in_lane, out_lane.rsplit("_", 1)[0]))
        return rows
    finally:
        traci.close()


def assert_wiring(movements: dict, links: list[tuple[int, str, str]]) -> dict[str, list[int]]:
    """Assert every movement is wired per ``movements.yaml`` and return the
    resolved ``{movement_id: [link_index, ...]}`` binding.

    Parameters
    ----------
    movements : dict
        The ``movements`` mapping loaded from ``movements.yaml``.
    links : list of tuple
        ``observed_links()`` output.

    Raises
    ------
    AssertionError
        With ALL mismatches collected, if any movement is mis-wired.
    """
    binding: dict[str, list[int]] = {}
    errors: list[str] = []

    for mid, spec in movements.items():
        approach, turn, controlled = spec["approach"], spec["turn"], spec["controlled"]
        in_lane = f"{IN_EDGE[approach]}_{LANE_FOR_TURN[turn]}"

        if controlled:
            # Exactly one link: this lane -> the turn's outgoing edge.
            want_out = OUT_EDGE[GEOMETRY[(approach, turn)]]
            hits = [i for (i, lane, out) in links if lane == in_lane and out == want_out]
            if len(hits) != 1:
                errors.append(
                    f"{mid} ({approach} {turn}): expected exactly 1 link "
                    f"{in_lane}->{want_out}, found {len(hits)} {hits}"
                )
            binding[mid] = hits
        else:
            # Free rightmost lane: must carry BOTH its through and its right link.
            want_through = OUT_EDGE[GEOMETRY[(approach, "through")]]
            want_right = OUT_EDGE[GEOMETRY[(approach, "right")]]
            hits = sorted(i for (i, lane, out) in links if lane == in_lane)
            outs = {out for (_i, lane, out) in links if lane == in_lane}
            if not {want_through, want_right} <= outs:
                errors.append(
                    f"{mid} ({approach} {turn}, free): lane {in_lane} should reach "
                    f"{{{want_through}, {want_right}}}, reaches {outs}"
                )
            binding[mid] = hits

    # Every incoming lane must exist (12 movements -> 12 distinct incoming lanes).
    seen_lanes = {lane for (_i, lane, _o) in links}
    for mid, spec in movements.items():
        lane = f"{IN_EDGE[spec['approach']]}_{LANE_FOR_TURN[spec['turn']]}"
        if lane not in seen_lanes:
            errors.append(f"{mid}: incoming lane {lane} not present in the net")

    if errors:
        raise AssertionError("network wiring does not match movements.yaml:\n  - " + "\n  - ".join(errors))
    return binding


def main() -> None:
    """Build the net, run all assertions, write the binding, exit 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--movements", type=Path, default=_VAULT_MOVEMENTS,
        help="path to the authoritative movements.yaml (default: vault SSOT)",
    )
    args = parser.parse_args()

    spec = yaml.safe_load(args.movements.read_text(encoding="utf-8"))
    movements = spec["movements"]

    print("[net] building intersection.net.xml via netconvert ...")
    build_net()
    assert_net_offset()
    print("[net] netOffset == (0,0) OK")

    binding = assert_wiring(movements, observed_links())
    print(f"[net] all {len(movements)} movements wired correctly vs movements.yaml OK")

    _BINDING_FILE.write_text(
        "# Generated by scripts/build_network.py (T-01-02) - do not edit by hand.\n"
        "# Resolves the link_index: TBD flag in specs/movements.yaml (open-flag B1).\n"
        + yaml.safe_dump({"tls_id": TLS_ID, "link_indices": binding}, sort_keys=False),
        encoding="utf-8",
    )
    print(f"[net] wrote binding -> {_BINDING_FILE.relative_to(_REPO_ROOT)}")

    print("\n  movement  approach turn      controlled  link_index(es)")
    for mid, spec_m in movements.items():
        print(
            f"  {mid:<8s}  {spec_m['approach']:<8s} {spec_m['turn']:<9s} "
            f"{str(spec_m['controlled']):<10s}  {binding[mid]}"
        )
    print("\n[net] OK - T-01-02 network built and asserted.")
    sys.exit(0)


if __name__ == "__main__":
    main()
