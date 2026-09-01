"""Regression tests for the Step-1 two-intersection arterial network.

These guard the arterial (``C1`` + ``C2``) wiring the same way
``test_network.py`` guards the legacy single intersection: an *independent*
hand-pinned oracle for the controlled links at each TLS, plus spec-conformance
checks against ``specs/movements.yaml`` when the vault is present.

The arterial-specific invariants covered here:

* the multi-TLS binding yaml carries BOTH TLS with 12 link-index entries each;
* :class:`~src.env.intersection.Intersection` resolves 12 movements / 8 phases
  for ``C1`` AND ``C2``, with two objects coexisting on one SUMO connection;
* ``netOffset`` stays ``(0, 0)`` (Unity client contract);
* the shared connecting edge pair is wired as the back-pressure coupling:
  C1's east out-edge IS C2's west in-edge (and vice versa).
"""

from __future__ import annotations

import pytest
import traci
import yaml
from sumolib import checkBinary

from scripts.build_network import (
    ARTERIAL,
    MOVEMENTS_SPEC,
    assert_net_offset,
    build_net,
    observed_links,
    resolve_bindings,
    write_binding_file,
)
from src.env.intersection import N_MOVEMENTS, N_PHASES, Intersection

# The exact (link_index, in_lane, out_edge) wiring each TLS MUST produce,
# verified by hand against movements.yaml geometry (approaches at C1:
# N=n1_c1, E=c2_c1 (shared), S=s1_c1, W=w1_c1; at C2: N=n2_c2, E=e2_c2,
# S=s2_c2, W=c1_c2 (shared)). Any drift here is a real regression.
EXPECTED_LINKS_C1: list[tuple[int, str, str]] = [
    (0, "n1_c1_0", "c1_w1"), (1, "n1_c1_0", "c1_s1"), (2, "n1_c1_1", "c1_s1"), (3, "n1_c1_2", "c1_c2"),
    (4, "c2_c1_0", "c1_n1"), (5, "c2_c1_0", "c1_w1"), (6, "c2_c1_1", "c1_w1"), (7, "c2_c1_2", "c1_s1"),
    (8, "s1_c1_0", "c1_c2"), (9, "s1_c1_0", "c1_n1"), (10, "s1_c1_1", "c1_n1"), (11, "s1_c1_2", "c1_w1"),
    (12, "w1_c1_0", "c1_s1"), (13, "w1_c1_0", "c1_c2"), (14, "w1_c1_1", "c1_c2"), (15, "w1_c1_2", "c1_n1"),
]
EXPECTED_LINKS_C2: list[tuple[int, str, str]] = [
    (0, "n2_c2_0", "c2_c1"), (1, "n2_c2_0", "c2_s2"), (2, "n2_c2_1", "c2_s2"), (3, "n2_c2_2", "c2_e2"),
    (4, "e2_c2_0", "c2_n2"), (5, "e2_c2_0", "c2_c1"), (6, "e2_c2_1", "c2_c1"), (7, "e2_c2_2", "c2_s2"),
    (8, "s2_c2_0", "c2_e2"), (9, "s2_c2_0", "c2_n2"), (10, "s2_c2_1", "c2_n2"), (11, "s2_c2_2", "c2_c1"),
    (12, "c1_c2_0", "c2_s2"), (13, "c1_c2_0", "c2_e2"), (14, "c1_c2_1", "c2_e2"), (15, "c1_c2_2", "c2_n2"),
]

# Was a skipif on the vault copy's existence. The spec now ships in the repo
# (config/movements.yaml), so its absence is a broken checkout, not an environment to tolerate -
# and this skip is exactly why nothing caught the runtime depending on a folder outside the repo.
_needs_vault = pytest.mark.skipif(False, reason="movements spec ships in-repo")


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    """Compile the arterial network once for the whole test module."""
    build_net(ARTERIAL)


@pytest.fixture(scope="module")
def binding() -> dict:
    """Regenerate the arterial binding yaml from the freshly built net."""
    movements = yaml.safe_load(MOVEMENTS_SPEC.read_text(encoding="utf-8"))["movements"]
    write_binding_file(ARTERIAL, resolve_bindings(ARTERIAL, movements))
    return yaml.safe_load(ARTERIAL.binding_file.read_text(encoding="utf-8"))


def test_net_offset_is_origin() -> None:
    """The arterial net must keep netOffset == (0,0) for the Unity transform."""
    assert_net_offset(ARTERIAL.net_file)  # raises AssertionError if not the origin


def test_controlled_links_match_expected_wiring() -> None:
    """Each TLS must control exactly its 16 hand-verified links, in order."""
    assert observed_links("C1", ARTERIAL.net_file) == EXPECTED_LINKS_C1
    assert observed_links("C2", ARTERIAL.net_file) == EXPECTED_LINKS_C2


def test_connecting_edge_is_shared() -> None:
    """Back-pressure coupling: C1's east out-edge IS C2's west in-edge (and mirror)."""
    c1, c2 = ARTERIAL.tls
    assert c1.out_edge["E"] == c2.in_edge["W"] == "c1_c2"
    assert c2.out_edge["W"] == c1.in_edge["E"] == "c2_c1"
    # and the compiled net actually wires them: C1 emits onto c1_c2, C2 reads from it
    assert {out for (_i, _lane, out) in observed_links("C1", ARTERIAL.net_file)} >= {"c1_c2"}
    assert {
        lane.rsplit("_", 1)[0]
        for (_i, lane, _out) in observed_links("C2", ARTERIAL.net_file)
    } >= {"c1_c2"}


@_needs_vault
def test_binding_yaml_has_both_tls(binding: dict) -> None:
    """The multi-TLS binding must carry C1 AND C2 with 12 link-index entries each."""
    assert set(binding) == {"C1", "C2"}
    for tls_id in ("C1", "C2"):
        indices = binding[tls_id]["link_indices"]
        assert len(indices) == 12
        assert set(indices) == {f"M{i}" for i in range(12)}
        # controlled movements bind 1 link; free right-turn lanes bind 2
        assert all(len(v) in (1, 2) for v in indices.values())


@_needs_vault
def test_wiring_conforms_to_movements_spec(binding: dict) -> None:
    """Both TLS must assert clean against the authoritative movements.yaml."""
    movements = yaml.safe_load(MOVEMENTS_SPEC.read_text(encoding="utf-8"))["movements"]
    bindings = resolve_bindings(ARTERIAL, movements)  # raises on any mis-wiring
    assert bindings == {t: binding[t]["link_indices"] for t in ("C1", "C2")}


@_needs_vault
def test_both_intersections_resolve_and_coexist(binding: dict) -> None:
    """Intersection.from_traci must resolve C1 AND C2 (12 movements / 8 phases)
    as two coexisting objects on one SUMO connection."""
    traci.start(
        [checkBinary("sumo"), "-n", str(ARTERIAL.net_file), "--no-step-log", "true"]
    )
    try:
        ix1 = Intersection.from_traci(traci, "C1", binding_path=ARTERIAL.binding_file)
        ix2 = Intersection.from_traci(traci, "C2", binding_path=ARTERIAL.binding_file)
        for ix in (ix1, ix2):
            assert len(ix.movement_ids) == N_MOVEMENTS
            greens = {ix.green_state(a) for a in range(N_PHASES)}
            assert len(greens) == N_PHASES  # 8 distinct phases
            assert all(len(g) == 16 for g in greens)  # 16 controlled links per TLS
            assert ix.pressures(traci).shape == (N_MOVEMENTS,)
        # coupling shows up in the lanes: C1's E-approach lanes live on c2_c1,
        # which is exactly where C2's westbound movements discharge to.
        # E through = shared-lane link + middle-lane link (decisions.md 2026-08-28)
        assert ix1._in_lanes["M4"] == ["c2_c1_0", "c2_c1_1"]  # E through into C1
        assert ix2._out_lanes["M4"] == ["c2_c1_0", "c2_c1_1"]  # E through out of C2 (westbound)
    finally:
        traci.close()
