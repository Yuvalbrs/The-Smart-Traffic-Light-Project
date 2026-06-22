"""Tests for the SCN-01..05 scenario loader/validator (T-01-01).

Covers the DoD ("5 YAML files exist; each loads cleanly via the config loader")
plus the demand-profile math and the fail-loud validation paths.
"""

from __future__ import annotations

import pytest
import yaml

from src.scenarios.config import (
    SCENARIO_DIR,
    AxisDemand,
    ScenarioError,
    load_all,
    load_scenario,
)

EXPECTED_IDS = ["SCN-01", "SCN-02", "SCN-03", "SCN-04", "SCN-05"]


# --- DoD: the five files exist and load cleanly ---

def test_five_scenarios_load() -> None:
    scenarios = load_all()
    assert [s.id for s in scenarios] == EXPECTED_IDS


def test_each_scenario_well_formed() -> None:
    for s in load_all():
        assert s.duration_s > 0
        assert len(s.seeds) == 10
        assert pytest.approx(sum(s.turn_split.values())) == 1.0
        assert s.ns.profile in {"constant", "ramp", "sinusoidal"}
        assert s.ew.profile in {"constant", "ramp", "sinusoidal"}


# --- demand-profile math ---

def test_constant_rate_is_flat() -> None:
    d = AxisDemand("constant", {"vph": 200.0})
    assert d.rate_at(0) == 200.0
    assert d.rate_at(3600) == 200.0


def test_ramp_interpolates_then_holds() -> None:
    d = AxisDemand("ramp", {"vph_start": 800.0, "vph_end": 300.0, "ramp_s": 1800.0})
    assert d.rate_at(0) == 800.0
    assert d.rate_at(900) == pytest.approx(550.0)   # halfway down
    assert d.rate_at(1800) == 300.0
    assert d.rate_at(3600) == 300.0                 # held past the ramp


def test_sinusoidal_bounds_and_offset() -> None:
    base = AxisDemand("sinusoidal",
                      {"vph_min": 200.0, "vph_max": 600.0, "period_s": 3600.0, "phase_offset_deg": 0.0})
    assert base.rate_at(0) == pytest.approx(400.0)        # midline at t=0
    assert base.rate_at(900) == pytest.approx(600.0)      # quarter period -> peak
    # A 90deg offset axis starts at its peak instead of the midline.
    offset = AxisDemand("sinusoidal",
                        {"vph_min": 200.0, "vph_max": 600.0, "period_s": 3600.0, "phase_offset_deg": 90.0})
    assert offset.rate_at(0) == pytest.approx(600.0)


def test_scn03_axes_cross() -> None:
    """Rush hour: N/S starts high and decays; E/W starts low and rises."""
    scn03 = next(s for s in load_all() if s.id == "SCN-03")
    assert scn03.ns.rate_at(0) > scn03.ew.rate_at(0)
    assert scn03.ns.rate_at(1800) < scn03.ew.rate_at(1800)


# --- fail-loud validation ---

def test_missing_file_raises() -> None:
    with pytest.raises(ScenarioError, match="not found"):
        load_scenario(SCENARIO_DIR / "scn_99.yaml")


def test_bad_turn_split_rejected(tmp_path) -> None:
    bad = _valid_dict()
    bad["turn_split"] = {"left": 0.5, "through": 0.6, "right": 0.2}  # sums to 1.3
    p = tmp_path / "scn_bad.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ScenarioError, match="sum to 1.0"):
        load_scenario(p)


def test_unknown_profile_rejected(tmp_path) -> None:
    bad = _valid_dict()
    bad["demand"]["ns"] = {"profile": "exponential", "vph": 200}
    p = tmp_path / "scn_bad.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ScenarioError, match="profile"):
        load_scenario(p)


def test_missing_profile_param_rejected(tmp_path) -> None:
    bad = _valid_dict()
    bad["demand"]["ns"] = {"profile": "ramp", "vph_start": 800}  # missing vph_end, ramp_s
    p = tmp_path / "scn_bad.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ScenarioError, match="missing"):
        load_scenario(p)


def _valid_dict() -> dict:
    """A minimal valid scenario as a plain dict, for mutation in negative tests."""
    return {
        "id": "SCN-XX",
        "name": "test",
        "duration_s": 3600,
        "seeds": [0, 1, 2],
        "turn_split": {"left": 0.2, "through": 0.6, "right": 0.2},
        "vehicle": {"type": "passenger", "heavy_fraction": 0.0},
        "demand": {
            "ns": {"profile": "constant", "vph": 200},
            "ew": {"profile": "constant", "vph": 200},
        },
    }
