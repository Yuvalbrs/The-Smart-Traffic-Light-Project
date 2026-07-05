"""Tests for the arterial corridor route generation (network-arterial-plan.md step 3).

Covers: multi-edge routes traverse both junctions via the coupling edges, per-junction
turn resampling, determinism (same seed -> byte-identical file), approximate turn-split
conformance, and byte-stability of the LEGACY single-intersection generator.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from scripts.build_routes import (
    _ARTERIAL_TURNS,
    CorridorTrip,
    generate_corridor_trips,
    generate_trips,
    _route_xml,
)
from src.scenarios.config import load_scenario

_ROOT = Path(__file__).resolve().parents[1]
_SCN_DIR = _ROOT / "config" / "scenarios"


@pytest.fixture(scope="module")
def arterial_scn():
    return load_scenario(_SCN_DIR / "scn_a2.yaml")  # constant rates -> stable stats


@pytest.fixture(scope="module")
def trips(arterial_scn) -> list[CorridorTrip]:
    return generate_corridor_trips(arterial_scn, seed=0)


def test_routes_traverse_both_junctions(trips) -> None:
    """Corridor-through vehicles produce 3-edge routes over a coupling edge."""
    coupled = [t for t in trips if "c1_c2" in t.edges or "c2_c1" in t.edges]
    assert coupled, "no vehicle crossed the coupling edges"
    three_edge = [t for t in coupled if len(t.edges) == 3]
    assert three_edge, "no 3-edge (entry -> coupling -> exit) route generated"
    # every coupled route ends at a real exit edge of the FAR junction
    for t in coupled:
        assert t.edges[-1] not in ("c1_c2", "c2_c1")


def test_every_route_is_wired(trips) -> None:
    """Every consecutive edge pair is a legal movement in the arterial turn table."""
    legal_pairs = set()
    for (_node, _head, _turn), (out_edge, _n, _h) in _ARTERIAL_TURNS.items():
        legal_pairs.add(out_edge)
    for t in trips:
        assert 2 <= len(t.edges) <= 3
        for edge in t.edges[1:]:
            assert edge in legal_pairs


def test_deterministic_generation(arterial_scn) -> None:
    """Same (scenario, seed) -> byte-identical rendered route file."""
    a = _route_xml(arterial_scn, 3, generate_corridor_trips(arterial_scn, seed=3))
    b = _route_xml(arterial_scn, 3, generate_corridor_trips(arterial_scn, seed=3))
    assert a == b
    c = _route_xml(arterial_scn, 4, generate_corridor_trips(arterial_scn, seed=4))
    assert a != c  # different seed actually changes the draw


def test_turn_split_approximately_respected(trips) -> None:
    """First-junction turn frequencies approximate the 0.2/0.6/0.2 split."""
    counts = Counter(t.first_turn for t in trips)
    total = sum(counts.values())
    assert total > 500  # enough samples for a loose check
    assert counts["through"] / total == pytest.approx(0.6, abs=0.05)
    assert counts["left"] / total == pytest.approx(0.2, abs=0.05)
    assert counts["right"] / total == pytest.approx(0.2, abs=0.05)


def test_legacy_generator_byte_stable() -> None:
    """The legacy single-intersection generator is untouched: regenerating
    SCN-01 seed 0 must byte-match the committed route file."""
    scn = load_scenario(_SCN_DIR / "scn_01.yaml")
    rendered = _route_xml(scn, 0, generate_trips(scn, 0))
    committed = (_ROOT / "config" / "routes" / "scn_01_seed_00.rou.xml").read_text(
        encoding="utf-8"
    )
    assert rendered == committed
