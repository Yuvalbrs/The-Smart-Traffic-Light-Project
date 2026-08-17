"""T-05-01 - FastAPI presentation layer (hub, live bridge, replay).

Locked architecture (``decisions.md`` 2026-06-18, amended F10 2026-06-20): FastAPI hub, SUMO
via TraCI is the single source of truth and is **never blocked by clients**, WebSocket live at
**1 Hz on both channels** (clients interpolate), REST for replay, no auth, bounded per-client
queues with backpressure, and a schema version on every wire message.
"""

from src.api.hub import Hub, Subscriber
from src.api.wire import SCHEMA_VERSION, dashboard_frame

__all__ = ["Hub", "Subscriber", "SCHEMA_VERSION", "dashboard_frame"]
