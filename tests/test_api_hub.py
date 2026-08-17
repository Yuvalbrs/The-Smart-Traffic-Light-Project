"""T-05-01 - backpressure + wire-shape guards for the API layer.

No SUMO here: the hub and the wire format are pure, and the whole point of the DoD's
backpressure clause is that it must hold regardless of what the simulation is doing.
"""

from __future__ import annotations

import asyncio

import pytest

from src.api.hub import MAX_QUEUE, Hub, Subscriber
from src.api.wire import SCHEMA_VERSION, dashboard_frame


def _frame(i: int) -> dict:
    return {"seq": i}


@pytest.mark.asyncio
async def test_subscriber_keeps_the_newest_frames_when_full() -> None:
    """A slow client loses the STALEST frames, not the freshest - coalesce-to-latest."""
    sub = Subscriber()
    for i in range(MAX_QUEUE + 5):
        sub.offer(_frame(i))

    got = [sub.queue.get_nowait()["seq"] for _ in range(sub.queue.qsize())]
    assert len(got) == MAX_QUEUE
    assert got[-1] == MAX_QUEUE + 4, "the newest frame must always survive"
    assert got == sorted(got), "surviving frames stay in order"
    assert 0 not in got, "the oldest frames are the ones dropped"
    assert sub.dropped == 5


@pytest.mark.asyncio
async def test_publish_never_blocks_and_counts_drops() -> None:
    """The producer keeps running at full speed even with a client that never reads."""
    hub = Hub("dash")
    hub.bind_loop(asyncio.get_running_loop())
    slow = hub.subscribe("slow")

    for i in range(500):
        hub.publish(_frame(i))
    await asyncio.sleep(0)  # let the loop drain the call_soon_threadsafe callbacks

    assert hub.published == 500
    assert slow.queue.qsize() == MAX_QUEUE
    assert slow.dropped > 0
    assert hub.dropped_total == slow.dropped


@pytest.mark.asyncio
async def test_one_slow_client_does_not_starve_a_fast_one() -> None:
    """Independent mailboxes: the fast client sees every frame it asks for."""
    hub = Hub("dash")
    hub.bind_loop(asyncio.get_running_loop())
    fast = hub.subscribe("fast")
    _slow = hub.subscribe("slow")

    seen = []
    for i in range(MAX_QUEUE * 3):
        hub.publish(_frame(i))
        await asyncio.sleep(0)
        while not fast.queue.empty():
            seen.append(fast.queue.get_nowait()["seq"])

    assert seen == list(range(MAX_QUEUE * 3))
    assert fast.dropped == 0


def test_publish_without_a_running_loop_still_delivers() -> None:
    """Frames produced before the server's loop exists are not silently lost."""
    hub = Hub("unity")
    sub = hub.subscribe()
    hub.publish(_frame(1))
    assert sub.queue.qsize() == 1


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery() -> None:
    hub = Hub("dash")
    hub.bind_loop(asyncio.get_running_loop())
    sub = hub.subscribe()
    hub.unsubscribe(sub)
    hub.publish(_frame(1))
    await asyncio.sleep(0)
    assert sub.queue.empty()
    assert hub.subscriber_count == 0


def test_dashboard_frame_matches_the_locked_contract() -> None:
    """Fields are fixed by system-architecture-overview.md; clients are written against them."""
    frame = dashboard_frame(
        sim_time=245.0,
        current_phase=3,
        last_action=5,
        queue_lengths=[12, 8, 5, 14, 7, 9, 11, 6, 3, 8, 10, 4],
        pressures=[0.42] * 12,
        avg_wait_so_far=23.4,
        throughput_so_far=1248.0,
    )
    for field in (
        "sim_time",
        "current_phase",
        "queue_lengths",
        "pressures",
        "last_action",
        "running_kpis",
        "forecast_next_30s",
    ):
        assert field in frame, field
    assert frame["schema_version"] == SCHEMA_VERSION, "schema version on every wire message"
    assert len(frame["queue_lengths"]) == 12
    assert len(frame["pressures"]) == 12
    assert set(frame["running_kpis"]) == {
        "avg_wait_so_far",
        "throughput_so_far",
        "current_queue_total",
    }
    assert frame["running_kpis"]["current_queue_total"] == 97.0


def test_forecast_is_null_not_zeroes_for_non_hybrid_controllers() -> None:
    """Zero-filling would read as a confident 'no traffic' forecast; absence must look absent."""
    frame = dashboard_frame(
        sim_time=1.0,
        current_phase=0,
        last_action=0,
        queue_lengths=[0] * 12,
        pressures=[0] * 12,
        avg_wait_so_far=0.0,
        throughput_so_far=0.0,
    )
    assert frame["forecast_next_30s"] is None

    hybrid = dashboard_frame(
        sim_time=1.0,
        current_phase=0,
        last_action=0,
        queue_lengths=[0] * 12,
        pressures=[0] * 12,
        avg_wait_so_far=0.0,
        throughput_so_far=0.0,
        forecast_next_30s=[0.5] * 36,
    )
    assert len(hybrid["forecast_next_30s"]) == 36


def test_dashboard_frame_is_json_serializable() -> None:
    """numpy floats leaking into the payload would break every WebSocket send."""
    import json

    import numpy as np

    frame = dashboard_frame(
        sim_time=np.float32(12.0),
        current_phase=np.int64(2),
        last_action=np.int64(2),
        queue_lengths=np.arange(12, dtype=np.float32),
        pressures=np.arange(12, dtype=np.float64),
        avg_wait_so_far=np.float64(1.5),
        throughput_so_far=np.float32(900.0),
    )
    assert json.loads(json.dumps(frame))["current_phase"] == 2
