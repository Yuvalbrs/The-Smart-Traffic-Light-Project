"""T-01-04 - JSONL tracer for per-second vehicle snapshots.

Writes one ``sim_frame`` envelope per simulated second (1 Hz, the F10 lock) to a
JSONL file - the same envelope the live WebSocket stream uses, so replay and live
share one parse path (data-schema.md s4). Each line is validated against schema
v1.1.0 before it is written, so a frame missing ``movement_id`` can never reach
disk.

Public surface:

- :class:`src.trace.writer.JsonlWriter` - the line writer (pure; no SUMO).
- :class:`src.trace.sim_frame.MovementResolver` - lane -> movement / signal colors.
- :func:`src.trace.sim_frame.build_sim_frame` - assemble one envelope from TraCI.
"""

from __future__ import annotations

from src.trace.sim_frame import MovementResolver, build_sim_frame
from src.trace.writer import JsonlWriter

__all__ = ["JsonlWriter", "MovementResolver", "build_sim_frame"]
