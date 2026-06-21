"""Regression tests for the T-01-02 intersection network.

These guard the network wiring so a future edit to the ``.nod``/``.edg``/``.con``
sources (or a SUMO upgrade) cannot silently permute the link indices that the
whole pressure/phase/state pipeline depends on.

The core test (`test_controlled_links_match_expected_wiring`) is an *independent*
oracle: it pins the exact 16 controlled links by hand, so it does not merely
re-run ``build_network``'s own derivation. A second, spec-conformance test runs
the real assertion against ``specs/movements.yaml`` when the vault is present.
"""

from __future__ import annotations

import pytest
import yaml

from scripts.build_network import (
    _VAULT_MOVEMENTS,
    assert_net_offset,
    assert_wiring,
    build_net,
    observed_links,
)

# The exact (link_index, in_lane, out_edge) wiring the net MUST produce, verified
# by hand against movements.yaml geometry. Any drift here is a real regression.
EXPECTED_LINKS: list[tuple[int, str, str]] = [
    (0, "n_t_0", "t_w"), (1, "n_t_0", "t_s"), (2, "n_t_1", "t_s"), (3, "n_t_2", "t_e"),
    (4, "e_t_0", "t_n"), (5, "e_t_0", "t_w"), (6, "e_t_1", "t_w"), (7, "e_t_2", "t_s"),
    (8, "s_t_0", "t_e"), (9, "s_t_0", "t_n"), (10, "s_t_1", "t_n"), (11, "s_t_2", "t_w"),
    (12, "w_t_0", "t_s"), (13, "w_t_0", "t_e"), (14, "w_t_1", "t_e"), (15, "w_t_2", "t_n"),
]


@pytest.fixture(scope="module", autouse=True)
def _built_net() -> None:
    """Compile the network once for the whole test module."""
    build_net()


def test_net_offset_is_origin() -> None:
    """The compiled net must have netOffset == (0,0) for the Unity transform."""
    assert_net_offset()  # raises AssertionError if not the origin


def test_controlled_links_match_expected_wiring() -> None:
    """SUMO must control exactly the 16 hand-verified links, in order."""
    assert observed_links() == EXPECTED_LINKS


def test_twelve_incoming_lanes_present() -> None:
    """All 12 incoming movement lanes must appear in the controlled links."""
    lanes = {lane for (_idx, lane, _out) in observed_links()}
    assert lanes == {f"{e}_{i}" for e in ("n_t", "e_t", "s_t", "w_t") for i in range(3)}


@pytest.mark.skipif(
    not _VAULT_MOVEMENTS.exists(), reason="vault movements.yaml absent (e.g. CI box)"
)
def test_wiring_conforms_to_movements_spec() -> None:
    """The net wiring must assert clean against the authoritative movements.yaml."""
    movements = yaml.safe_load(_VAULT_MOVEMENTS.read_text(encoding="utf-8"))["movements"]
    binding = assert_wiring(movements, observed_links())
    assert len(binding) == 12
    assert binding["M0"] == [3]      # N left
    assert binding["M7"] == [10]     # S through
    assert binding["M2"] == [0, 1]   # N right lane (free, through + right)
