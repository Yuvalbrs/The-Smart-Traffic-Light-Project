"""T-02-08 - The single KPI extractor: 7 locked metrics from JSONL + trip-info.

Computes the seven KPIs defined in ``kpis.md`` (the formula SSOT) for one episode,
from two artifacts produced during a run:

* the **JSONL trace** (``sim_frame`` per second, T-01-04) - per-vehicle ``speed``
  and ``movement_id`` (for the queue samples and the per-movement fairness split)
  and ``sim_time`` (for the warm-up window);
* the **SUMO trip-info XML** (``--tripinfo-output``) - per-vehicle ``waitingTime``
  and ``waitingCount`` (wait, stops, p95) and ``depart`` (warm-up filtering).

The seven metrics (kpis.md):

1. ``avg_waiting_time``      mean per-vehicle accumulated wait (s)             [down]
2. ``avg_queue_length``     mean halting count over decision steps (veh)      [down]
3. ``throughput``           arrived / episode-hours (veh/h)                   [up]
4. ``num_stops``            mean per-vehicle stop count (waitingCount)        [down]
5a. ``fairness_std``        std of per-movement mean wait across M0..M11 (s)  [down]
5b. ``per_movement_max_wait``  worst single wait per movement (s[12])        [down]
6. ``wait_p95``             95th-percentile per-vehicle wait (s)             [down]

Audit additions (backlog T-02-08 DoD): the warm-up cutoff (``[warmup_s,
episode_length_s]``, applied identically for every controller); ``movement_id``
consumed for 5a/5b; the gridlock-guard outputs surfaced (``departed/arrived/
insertion_backlog_fraction`` + ``gridlock_censored`` when backlog > threshold);
and ``per_movement_p95_wait`` alongside the absolute max (open-items E5).

Scope: this is the pure extractor + its synthetic-data tests. Emitting the real
JSONL/trip-info during an episode is the eval runner's job (T-04-01); this function
consumes those files wherever they come from.

Decisions (surfaced, not silent):

* Warm-up filters per-vehicle KPIs by ``depart in [warmup_s, episode_length_s)``
  and queue samples by ``sim_time in [warmup_s, episode_length_s]``. **Throughput**
  follows the kpis.md formula over the *full* episode (a flow rate; kpis.md is the
  declared SSOT and its KPI-3 formula carries no warm-up).
* ``fairness_std`` is the sample std (ddof=1, matching ``statistics.stdev``) over
  the movements that actually carried traffic.
* ``per_movement_max_wait`` is the DB ``episode_kpi`` column; ``per_movement_p95_wait``
  and the gridlock fields have no ``episode_kpi`` column (gridlock lives on the
  ``episode`` table) - they ride on the result object for the eval runner to route.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

N_MOVEMENTS = 12
_HALT_SPEED = 0.1  # m/s; SUMO default halting threshold (kpis.md KPI 1/2)
_WARMUP_S = 300
_EPISODE_LENGTH_S = 3600
_GRIDLOCK_BACKLOG_THRESHOLD = 0.10  # insertion backlog fraction above which we censor


@dataclass(frozen=True)
class _Trip:
    """One vehicle's trip-info row (only the fields the KPIs need)."""

    vid: str
    depart: float
    waiting_time: float
    waiting_count: int


@dataclass(frozen=True)
class EpisodeKPIs:
    """The 7 KPIs + audit outputs for one episode (kpis.md / T-02-08 DoD)."""

    avg_waiting_time: float
    avg_queue_length: float
    throughput: float
    num_stops: float
    fairness_std: float
    wait_p95: float
    per_movement_max_wait: list[float | None]  # s[12]; None where no traffic
    per_movement_p95_wait: list[float | None]  # s[12]; audit E5
    worst_movement_id: int | None
    worst_movement_max_wait: float | None
    # gridlock-guard / provenance (route to the `episode` table, not `episode_kpi`)
    loaded_count: int
    departed_count: int
    arrived_count: int
    insertion_backlog_fraction: float
    gridlock_censored: bool
    n_vehicles_after_warmup: int

    def to_episode_kpi_fields(self) -> dict[str, Any]:
        """Return only the columns the ``episode_kpi`` table stores (models.py)."""
        return {
            "avg_waiting_time": self.avg_waiting_time,
            "avg_queue_length": self.avg_queue_length,
            "throughput": self.throughput,
            "num_stops": self.num_stops,
            "wait_p95": self.wait_p95,
            "fairness_std": self.fairness_std,
            "per_movement_max_wait": self.per_movement_max_wait,
        }


def _parse_tripinfo(path: Path) -> list[_Trip]:
    """Parse a SUMO trip-info XML into the per-vehicle rows the KPIs need."""
    trips: list[_Trip] = []
    for el in ET.parse(path).getroot().iter("tripinfo"):
        trips.append(
            _Trip(
                vid=el.get("id", ""),
                depart=float(el.get("depart", "nan")),
                waiting_time=float(el.get("waitingTime", "0")),
                waiting_count=int(float(el.get("waitingCount", "0"))),
            )
        )
    return trips


def _scan_trace(
    path: Path, *, warmup_s: float, episode_length_s: float
) -> tuple[list[int], dict[str, int], float]:
    """One pass over the JSONL trace.

    Returns ``(queue_samples, vid_to_movement, last_sim_time)``:

    * ``queue_samples`` - halting count (speed < threshold on an approach) per frame
      inside the warm-up window;
    * ``vid_to_movement`` - each vehicle's approach movement index (first non-null
      ``movement_id`` seen), for the fairness split;
    * ``last_sim_time`` - the largest ``sim_time`` (episode end, for throughput).
    """
    queue_samples: list[int] = []
    vid_to_movement: dict[str, int] = {}
    last_sim_time = 0.0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            sim_time = float(frame["sim_time"])
            last_sim_time = max(last_sim_time, sim_time)
            vehicles = frame["payload"]["vehicles"]
            in_window = warmup_s <= sim_time <= episode_length_s
            halting = 0
            for v in vehicles:
                mid = v.get("movement_id")
                if mid is None:
                    continue
                if v["id"] not in vid_to_movement:
                    vid_to_movement[v["id"]] = int(mid[1:])  # "M7" -> 7
                if in_window and v["speed"] < _HALT_SPEED:
                    halting += 1
            if in_window:
                queue_samples.append(halting)
    return queue_samples, vid_to_movement, last_sim_time


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def extract_kpis(
    jsonl_path: str | Path,
    tripinfo_path: str | Path,
    *,
    episode_counters: dict[str, int] | None = None,
    warmup_s: float = _WARMUP_S,
    episode_length_s: float = _EPISODE_LENGTH_S,
    gridlock_threshold: float = _GRIDLOCK_BACKLOG_THRESHOLD,
) -> EpisodeKPIs:
    """Compute the 7 KPIs (+ audit outputs) for one episode.

    Parameters
    ----------
    jsonl_path, tripinfo_path : path
        The episode's JSONL trace and SUMO trip-info XML.
    episode_counters : dict, optional
        ``{"loaded_count", "departed_count", "arrived_count"}`` from the env
        (``info["episode"]``). ``loaded_count`` is the only way to know the
        insertion backlog (trip-info holds completed trips only); when omitted,
        backlog is 0 and ``arrived_count`` falls back to the trip-info row count.
    warmup_s, episode_length_s : float
        The KPI window ``[warmup_s, episode_length_s]`` (applied identically for
        every controller).
    gridlock_threshold : float
        ``gridlock_censored`` is set when ``insertion_backlog_fraction`` exceeds it.
    """
    jsonl_path, tripinfo_path = Path(jsonl_path), Path(tripinfo_path)
    trips = _parse_tripinfo(tripinfo_path)
    queue_samples, vid_to_movement, last_sim_time = _scan_trace(
        jsonl_path, warmup_s=warmup_s, episode_length_s=episode_length_s
    )

    # per-vehicle KPIs: warm-up filter by departure time
    in_window = [t for t in trips if warmup_s <= t.depart < episode_length_s]
    waits = [t.waiting_time for t in in_window]
    stops = [float(t.waiting_count) for t in in_window]

    # per-movement split (5a / 5b / E5)
    by_movement: dict[int, list[float]] = {m: [] for m in range(N_MOVEMENTS)}
    for t in in_window:
        m = vid_to_movement.get(t.vid)
        if m is not None:
            by_movement[m].append(t.waiting_time)

    per_movement_max: list[float | None] = [
        (max(ws) if ws else None) for ws in by_movement.values()
    ]
    per_movement_p95: list[float | None] = [
        (float(np.percentile(ws, 95)) if ws else None) for ws in by_movement.values()
    ]
    movement_means = [float(np.mean(ws)) for ws in by_movement.values() if ws]
    fairness_std = float(np.std(movement_means, ddof=1)) if len(movement_means) >= 2 else float("nan")

    worst_id, worst_max = None, None
    present_max = [(m, mx) for m, mx in enumerate(per_movement_max) if mx is not None]
    if present_max:
        worst_id, worst_max = max(present_max, key=lambda pair: pair[1])

    # throughput: kpis.md KPI-3, full episode (flow rate, no warm-up)
    counters = episode_counters or {}
    arrived_count = int(counters.get("arrived_count", len(trips)))
    loaded_count = int(counters.get("loaded_count", arrived_count))
    departed_count = int(counters.get("departed_count", arrived_count))
    duration_h = (last_sim_time or episode_length_s) / 3600.0
    throughput = arrived_count / duration_h if duration_h > 0 else float("nan")

    backlog = (loaded_count - departed_count) / loaded_count if loaded_count else 0.0

    return EpisodeKPIs(
        avg_waiting_time=_safe_mean(waits),
        avg_queue_length=_safe_mean([float(q) for q in queue_samples]),
        throughput=throughput,
        num_stops=_safe_mean(stops),
        fairness_std=fairness_std,
        wait_p95=float(np.percentile(waits, 95)) if waits else float("nan"),
        per_movement_max_wait=per_movement_max,
        per_movement_p95_wait=per_movement_p95,
        worst_movement_id=worst_id,
        worst_movement_max_wait=worst_max,
        loaded_count=loaded_count,
        departed_count=departed_count,
        arrived_count=arrived_count,
        insertion_backlog_fraction=backlog,
        gridlock_censored=backlog > gridlock_threshold,
        n_vehicles_after_warmup=len(in_window),
    )
