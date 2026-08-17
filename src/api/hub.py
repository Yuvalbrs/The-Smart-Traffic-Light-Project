"""T-05-01 - Fan-out hub with bounded per-client queues and coalesce-to-latest backpressure.

The locked rule is that SUMO is never blocked by a client (``decisions.md`` 2026-06-18). The
simulation loop calls :meth:`Hub.publish` from its own thread and must return immediately, so a
slow or stalled WebSocket client can only ever cost itself frames - never the simulation.

Backpressure policy (T-05-01 DoD): each subscriber owns an ``asyncio.Queue(maxsize=8)``. When it
is full the OLDEST frame is dropped to make room for the newest, because for a live view the
freshest state is the only interesting one; a client that falls behind should jump to *now*
rather than replay a stale backlog. Every drop is counted so ``/health`` can report it, and the
sequence numbers already carried by ``sim_frame`` let a client detect the gap itself.
"""

from __future__ import annotations

import asyncio
from typing import Any

MAX_QUEUE = 8  # per-client buffer depth (T-05-01 DoD)


class Subscriber:
    """One connected client's bounded mailbox.

    Attributes
    ----------
    queue : asyncio.Queue
        Bounded to :data:`MAX_QUEUE` frames.
    dropped : int
        Frames discarded for this client because it could not keep up.
    """

    __slots__ = ("queue", "dropped", "name")

    def __init__(self, name: str = "client") -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=MAX_QUEUE)
        self.dropped = 0
        self.name = name

    def offer(self, frame: dict[str, Any]) -> bool:
        """Enqueue ``frame``, dropping the oldest if the mailbox is full.

        Returns
        -------
        bool
            ``True`` if nothing had to be dropped.
        """
        try:
            self.queue.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()  # coalesce to latest: discard the stalest frame
                self.dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - drained concurrently
                pass
            try:
                self.queue.put_nowait(frame)
            except asyncio.QueueFull:  # pragma: no cover - refilled concurrently
                self.dropped += 1
            return False

    async def get(self) -> dict[str, Any]:
        """Await the next frame for this client."""
        return await self.queue.get()


class Hub:
    """Fan-out of one frame stream to every subscriber, without blocking the producer.

    Parameters
    ----------
    name : str
        Channel name, used in ``/health`` output (``"dash"`` / ``"unity"``).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._subs: set[Subscriber] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.published = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the event loop that owns the subscriber queues.

        :meth:`publish` is called from the simulation thread, so handing frames to asyncio
        queues has to be marshalled onto this loop rather than touched directly.
        """
        self._loop = loop

    def subscribe(self, name: str = "client") -> Subscriber:
        """Register and return a new subscriber mailbox."""
        sub = Subscriber(name)
        self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        """Remove a subscriber (idempotent)."""
        self._subs.discard(sub)

    @property
    def subscriber_count(self) -> int:
        """How many clients are currently attached."""
        return len(self._subs)

    @property
    def dropped_total(self) -> int:
        """Frames dropped across all current subscribers."""
        return sum(s.dropped for s in self._subs)

    def publish(self, frame: dict[str, Any]) -> None:
        """Offer ``frame`` to every subscriber. Safe to call from a non-async thread.

        Never raises and never waits: this runs inside the SUMO stepping loop.
        """
        self.published += 1
        if not self._subs:
            return
        loop, subs = self._loop, list(self._subs)
        if loop is None or not loop.is_running():
            for sub in subs:  # no loop yet (tests, or pre-startup): degrade to direct offer
                sub.offer(frame)
            return
        loop.call_soon_threadsafe(self._offer_all, subs, frame)

    @staticmethod
    def _offer_all(subs: list[Subscriber], frame: dict[str, Any]) -> None:
        for sub in subs:
            sub.offer(frame)
