"""Tests for the schema v1.1.0 envelope validator (T-01-09 / open-item E7).

The load-bearing guarantee from the audit: a 1.0 payload - i.e. one missing
``movement_id`` - must be REJECTED, because every per-movement fairness KPI
depends on that field being present from the first recorded episode. The tests
below pin both rejection paths (old version stamp, and a missing field at the
current version) plus the happy paths.
"""

from __future__ import annotations

import pytest

from src.schema.validate import SCHEMA_VERSION, SchemaError, validate_envelope


def _sim_frame() -> dict:
    """A minimal well-formed ``sim_frame`` at the current schema version."""
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "sim_frame",
        "sim_time": 123.0,
        "seq": 123,
        "transition": False,
        "episode_id": 7,
        "payload": {
            "vehicles": [
                {"id": "veh0", "x": 12.3, "y": -4.5, "angle": 90.0, "speed": 8.1,
                 "lane": "n_t_1", "type": "car", "movement_id": "M1"},
            ],
            "signal": {
                "phase_index": 3,
                "signal_colors": {"M0": "red", "M1": "green"},
                "sumo_state": "GrGr",
                "phase_remaining_s": 6.0,
            },
        },
    }


def _kpi_frame() -> dict:
    """A minimal well-formed ``kpi_frame`` at the current schema version."""
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "kpi_frame",
        "sim_time": 123.0,
        "seq": 12,
        "payload": {
            "signal": {
                "phase_index": 0,
                "signal_colors": {"M0": "green"},
                "sumo_state": "G",
                "phase_remaining_s": 4.0,
            },
            "queue_lengths": [0] * 12,
            "pressures": [0.0] * 12,
            "running_kpis": {},
        },
    }


# --- happy paths ---

def test_valid_sim_frame_passes() -> None:
    validate_envelope(_sim_frame())  # must not raise


def test_valid_kpi_frame_passes() -> None:
    validate_envelope(_kpi_frame())


def test_kpi_frame_with_forecast_passes() -> None:
    frame = _kpi_frame()
    frame["payload"]["forecast_next_30s"] = [0.0] * 36
    validate_envelope(frame)


def test_null_movement_id_allowed() -> None:
    """A vehicle off the approaches (mid-junction) may have movement_id == null."""
    frame = _sim_frame()
    frame["payload"]["vehicles"][0]["movement_id"] = None
    validate_envelope(frame)


# --- the core DoD: 1.0 payloads / missing movement_id are rejected ---

def test_integer_schema_version_rejected() -> None:
    """The old int schema_version (1) is a 1.0-era payload -> rejected."""
    frame = _sim_frame()
    frame["schema_version"] = 1
    with pytest.raises(SchemaError, match="schema_version"):
        validate_envelope(frame)


def test_old_semver_rejected() -> None:
    frame = _sim_frame()
    frame["schema_version"] = "1.0"
    with pytest.raises(SchemaError, match="schema_version"):
        validate_envelope(frame)


def test_vehicle_without_movement_id_rejected() -> None:
    """The headline guard: a vehicle lacking movement_id is rejected."""
    frame = _sim_frame()
    del frame["payload"]["vehicles"][0]["movement_id"]
    with pytest.raises(SchemaError, match="movement_id"):
        validate_envelope(frame)


def test_bad_movement_id_value_rejected() -> None:
    frame = _sim_frame()
    frame["payload"]["vehicles"][0]["movement_id"] = "M99"
    with pytest.raises(SchemaError, match="M0..M11"):
        validate_envelope(frame)


def test_bad_signal_color_rejected() -> None:
    frame = _sim_frame()
    frame["payload"]["signal"]["signal_colors"]["M0"] = "purple"
    with pytest.raises(SchemaError):
        validate_envelope(frame)


def test_missing_envelope_key_rejected() -> None:
    frame = _sim_frame()
    del frame["seq"]
    with pytest.raises(SchemaError, match="missing required keys"):
        validate_envelope(frame)


def test_non_dict_rejected() -> None:
    with pytest.raises(SchemaError):
        validate_envelope([1, 2, 3])
