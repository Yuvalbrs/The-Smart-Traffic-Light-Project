"""T-05-01 - Wire formats for the two 1 Hz WebSocket channels.

Two channels, both at 1 Hz (F10 amendment, ``system-architecture-overview.md`` §API contracts);
clients interpolate between frames for smoothness.

* **Unity channel** - the ``sim_frame`` envelope exactly as the tracer writes it (schema 1.1.0):
  every vehicle with position, kinematics and ``movement_id``, plus the ``signal`` block. It is
  passed through untouched, so Unity and the JSONL replay consume byte-identical frames.
* **Dashboard channel** - the derived per-second summary defined in
  ``system-architecture-overview.md``: ``sim_time``, ``current_phase``, ``queue_lengths`` (12),
  ``pressures`` (12), ``last_action``, ``running_kpis`` and ``forecast_next_30s``.

The vault's example dashboard object predates the "schema version on every wire message" rule
(``decisions.md`` 2026-06-18, architecture item 6), so a ``schema_version`` field is added here.
That is the only departure from the documented example.

**The running KPIs are running ESTIMATES, not the project's KPIs.** The confirmatory KPIs come
from :mod:`src.metrics.kpi_extractor` over SUMO's trip-info after the episode ends (one
implementation, no drift - ``project-rules.md``). These live values exist to make a dashboard
move; they are named ``*_so_far`` and must never be reported as results.
"""

from __future__ import annotations

from typing import Any, Sequence

SCHEMA_VERSION = "1.1.0"
DASHBOARD_MSG = "dashboard_frame"


def dashboard_frame(
    *,
    sim_time: float,
    current_phase: int,
    last_action: int,
    queue_lengths: Sequence[float],
    pressures: Sequence[float],
    avg_wait_so_far: float,
    throughput_so_far: float,
    forecast_next_30s: Sequence[float] | None = None,
    seq: int = 0,
    episode_id: int = 0,
) -> dict[str, Any]:
    """Build one dashboard-channel frame.

    Parameters
    ----------
    sim_time : float
        SUMO time of this frame, in seconds.
    current_phase : int
        NEMA phase (0-7) currently displayed.
    last_action : int
        The agent's most recent action. Equal to ``current_phase`` except during a
        yellow/all-red transition, when the lights still show the outgoing phase.
    queue_lengths : sequence of float
        Per-movement halting counts, M0-M11 (12 values).
    pressures : sequence of float
        Per-movement pressure, M0-M11 (12 values), unnormalized.
    avg_wait_so_far, throughput_so_far : float
        Running estimates only - see the module docstring.
    forecast_next_30s : sequence of float, optional
        The frozen LSTM's 36-dim forecast (3 horizons x 12 movements) when a forecast-using
        variant is driving. ``None`` for every non-hybrid controller, which is most of them:
        the field is always present so clients need no special case, but it is explicitly null
        rather than zero-filled, because zeros would read as a confident "no traffic" forecast.
    seq : int
        Monotonic frame counter, so a client can detect frames dropped by backpressure.
    episode_id : int
        Episode this frame belongs to.

    Returns
    -------
    dict
        The dashboard wire message.
    """
    queues = [float(q) for q in queue_lengths]
    return {
        "schema_version": SCHEMA_VERSION,
        "type": DASHBOARD_MSG,
        "seq": int(seq),
        "episode_id": int(episode_id),
        "sim_time": float(sim_time),
        "current_phase": int(current_phase),
        "last_action": int(last_action),
        "queue_lengths": queues,
        "pressures": [float(p) for p in pressures],
        "running_kpis": {
            "avg_wait_so_far": float(avg_wait_so_far),
            "throughput_so_far": float(throughput_so_far),
            "current_queue_total": float(sum(queues)),
        },
        "forecast_next_30s": (
            None if forecast_next_30s is None else [float(f) for f in forecast_next_30s]
        ),
    }
