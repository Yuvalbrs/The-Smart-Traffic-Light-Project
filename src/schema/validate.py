"""T-01-09 - Validate data envelopes against schema v1.1.0 (open-item E7).

``specs/data-schema.md`` (in the vault) is the SSOT for the wire/JSONL envelope.
This module is the runtime guard that enforces it. Every persisted JSONL record
and every live WebSocket message is an envelope
``{schema_version, type, sim_time, seq, payload}``, and at **schema 1.1.0** every
vehicle in a ``sim_frame`` MUST carry a ``movement_id`` so that per-movement
fairness KPIs are computable.

The whole point of the 1.0 -> 1.1.0 bump is that a 1.0 payload - one lacking
``movement_id`` - is *rejected* rather than silently recorded and only later
found useless (which would force re-running every eval episode).

Pure standard library (no SQLAlchemy / pydantic) so it can guard the tracer
before the DB models (T-01-03) exist. Every violation raises ``SchemaError``.
"""

from __future__ import annotations

from typing import Any

#: The schema version this consumer understands. Payloads stamped otherwise are rejected.
SCHEMA_VERSION = "1.1.0"

#: The 12 movements M0..M11 (specs/movements.yaml). A vehicle's movement_id is one
#: of these, or ``None`` when it is not currently on an approach lane.
MOVEMENT_IDS = frozenset(f"M{i}" for i in range(12))

#: Valid per-movement signal colors.
SIGNAL_COLORS = frozenset({"red", "yellow", "green"})

_ENVELOPE_KEYS = ("schema_version", "type", "sim_time", "seq", "payload")
_VEHICLE_KEYS = ("id", "x", "y", "angle", "speed", "lane", "type", "movement_id")
_SIGNAL_KEYS = ("phase_index", "signal_colors", "sumo_state", "phase_remaining_s")


class SchemaError(ValueError):
    """Raised when an envelope does not conform to schema v1.1.0."""


def validate_envelope(msg: Any) -> None:
    """Validate one envelope (``sim_frame`` / ``kpi_frame``) against schema v1.1.0.

    Parameters
    ----------
    msg : Any
        A decoded JSON object (one JSONL line or one WS message).

    Raises
    ------
    SchemaError
        On the first structural violation, with a human-readable reason.
    """
    if not isinstance(msg, dict):
        raise SchemaError(f"envelope must be a dict, got {type(msg).__name__}")

    missing = [k for k in _ENVELOPE_KEYS if k not in msg]
    if missing:
        raise SchemaError(f"envelope missing required keys: {missing}")

    version = msg["schema_version"]
    if version != SCHEMA_VERSION:
        raise SchemaError(
            f"unsupported schema_version {version!r}; this consumer requires "
            f"{SCHEMA_VERSION!r} (a 1.0 payload lacks movement_id and is rejected)"
        )

    payload = msg["payload"]
    if not isinstance(payload, dict):
        raise SchemaError(f"payload must be a dict, got {type(payload).__name__}")

    frame_type = msg["type"]
    if frame_type == "sim_frame":
        _validate_sim_frame(payload)
    elif frame_type == "kpi_frame":
        _validate_kpi_frame(payload)
    # Event frame types are not pinned yet; only the envelope is enforced for them.


def _validate_sim_frame(payload: dict) -> None:
    """Enforce the ``sim_frame`` body: a vehicles list (each with movement_id) + signal."""
    vehicles = payload.get("vehicles")
    if not isinstance(vehicles, list):
        raise SchemaError("sim_frame.payload.vehicles must be a list")

    for i, veh in enumerate(vehicles):
        if not isinstance(veh, dict):
            raise SchemaError(f"vehicle[{i}] must be a dict, got {type(veh).__name__}")
        missing = [k for k in _VEHICLE_KEYS if k not in veh]
        if missing:
            raise SchemaError(
                f"vehicle[{i}] (id={veh.get('id')!r}) missing required keys {missing} - "
                f"'movement_id' is required at schema {SCHEMA_VERSION}"
            )
        mid = veh["movement_id"]
        if mid is not None and mid not in MOVEMENT_IDS:
            raise SchemaError(f"vehicle[{i}] movement_id {mid!r} not in M0..M11 (or null)")

    _validate_signal(payload.get("signal"))


def _validate_kpi_frame(payload: dict) -> None:
    """Enforce the ``kpi_frame`` body: signal + the fixed-width KPI arrays."""
    _validate_signal(payload.get("signal"))
    for field, n in (("queue_lengths", 12), ("pressures", 12)):
        val = payload.get(field)
        if not isinstance(val, list) or len(val) != n:
            raise SchemaError(f"kpi_frame.payload.{field} must be a length-{n} list")
    forecast = payload.get("forecast_next_30s")  # optional: omitted when no LSTM is loaded
    if forecast is not None and (not isinstance(forecast, list) or len(forecast) != 36):
        raise SchemaError(
            "kpi_frame.payload.forecast_next_30s must be a length-36 list when present"
        )


def _validate_signal(signal: Any) -> None:
    """Enforce the shared ``signal`` block (present in both frame types)."""
    if not isinstance(signal, dict):
        raise SchemaError("signal block must be a dict")
    missing = [k for k in _SIGNAL_KEYS if k not in signal]
    if missing:
        raise SchemaError(f"signal block missing keys: {missing}")

    phase = signal["phase_index"]
    # bool is a subclass of int in Python; exclude it explicitly.
    if isinstance(phase, bool) or not isinstance(phase, int) or not 0 <= phase <= 7:
        raise SchemaError(f"signal.phase_index must be int 0..7, got {phase!r}")

    colors = signal["signal_colors"]
    if not isinstance(colors, dict):
        raise SchemaError("signal.signal_colors must be a dict {Mk: color}")
    for mid, color in colors.items():
        if mid not in MOVEMENT_IDS:
            raise SchemaError(f"signal_colors key {mid!r} not in M0..M11")
        if color not in SIGNAL_COLORS:
            raise SchemaError(
                f"signal_colors[{mid}] = {color!r} not in {sorted(SIGNAL_COLORS)}"
            )
