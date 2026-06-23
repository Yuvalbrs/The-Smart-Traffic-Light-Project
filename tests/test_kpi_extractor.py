"""T-02-08 - Tests for the KPI extractor on synthetic episodes (DoD).

Every KPI is checked against a hand-computed expected value on a tiny crafted
JSONL + trip-info pair, so a formula regression fails loudly. Also covers the
warm-up cutoff, the per-movement fairness split, and gridlock censoring.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.metrics.kpi_extractor import extract_kpis

# --- the crafted episode (window [10, 100); v_warm departs at 5 -> excluded) ---
#
# in-window vehicles (depart >= 10):
#   v1  M0  wait 10  stops 1
#   v2  M0  wait 20  stops 2
#   v3  M4  wait 30  stops 3
#   v4  M4  wait 50  stops 0
_TRIPINFO = """<tripinfos>
  <tripinfo id="v_warm" depart="5.00" arrival="8.00" waitingTime="999.00" waitingCount="9"/>
  <tripinfo id="v1" depart="20.00" arrival="35.00" waitingTime="10.00" waitingCount="1"/>
  <tripinfo id="v2" depart="30.00" arrival="55.00" waitingTime="20.00" waitingCount="2"/>
  <tripinfo id="v3" depart="40.00" arrival="80.00" waitingTime="30.00" waitingCount="3"/>
  <tripinfo id="v4" depart="50.00" arrival="95.00" waitingTime="50.00" waitingCount="0"/>
</tripinfos>
"""


def _v(vid: str, speed: float, movement: str | None) -> dict:
    return {"id": vid, "speed": speed, "movement_id": movement}


_FRAMES = [
    {"sim_time": 5, "payload": {"vehicles": [_v("v_warm", 0.0, "M0")]}},  # pre-warmup
    {"sim_time": 20, "payload": {"vehicles": [_v("v1", 0.0, "M0"), _v("v2", 5.0, "M0")]}},
    {"sim_time": 30, "payload": {"vehicles": [_v("v1", 0.0, "M0"), _v("v2", 0.0, "M0"), _v("v3", 0.0, "M4")]}},
    {"sim_time": 40, "payload": {"vehicles": [_v("v3", 0.0, "M4"), _v("v4", 5.0, "M4")]}},
]


def _write(tmp_path: Path, frames=_FRAMES, tripinfo=_TRIPINFO) -> tuple[Path, Path]:
    jsonl = tmp_path / "trace.jsonl"
    jsonl.write_text("\n".join(json.dumps(f) for f in frames) + "\n", encoding="utf-8")
    tinfo = tmp_path / "tripinfo.xml"
    tinfo.write_text(tripinfo, encoding="utf-8")
    return jsonl, tinfo


def _extract(tmp_path: Path, **kw):
    jsonl, tinfo = _write(tmp_path)
    counters = {"loaded_count": 6, "departed_count": 5, "arrived_count": 5}
    return extract_kpis(jsonl, tinfo, episode_counters=counters,
                        warmup_s=10, episode_length_s=100, **kw)


def test_per_vehicle_kpis(tmp_path) -> None:
    k = _extract(tmp_path)
    assert k.n_vehicles_after_warmup == 4  # v_warm excluded by the warm-up cutoff
    assert k.avg_waiting_time == pytest.approx(27.5)   # (10+20+30+50)/4
    assert k.num_stops == pytest.approx(1.5)           # (1+2+3+0)/4
    assert k.wait_p95 == pytest.approx(47.0)           # p95 of [10,20,30,50]


def test_queue_and_throughput(tmp_path) -> None:
    k = _extract(tmp_path)
    # halting-on-approach per in-window frame: [1, 3, 1]
    assert k.avg_queue_length == pytest.approx(5 / 3)
    # arrived 5 / (last_sim_time 40 s -> hours); early end normalizes by actual duration
    assert k.throughput == pytest.approx(5 / (40 / 3600.0))


def test_fairness_split_and_starvation(tmp_path) -> None:
    k = _extract(tmp_path)
    # M0 waits [10,20] mean 15; M4 waits [30,50] mean 40 -> sample std of [15,40]
    assert k.fairness_std == pytest.approx(math.sqrt(312.5))
    assert k.per_movement_max_wait[0] == pytest.approx(20.0)
    assert k.per_movement_max_wait[4] == pytest.approx(50.0)
    assert all(k.per_movement_max_wait[m] is None for m in range(12) if m not in (0, 4))
    assert k.per_movement_p95_wait[0] == pytest.approx(19.5)  # p95 of [10,20]
    assert k.per_movement_p95_wait[4] == pytest.approx(49.0)  # p95 of [30,50]
    assert k.worst_movement_id == 4 and k.worst_movement_max_wait == pytest.approx(50.0)


def test_gridlock_censoring(tmp_path) -> None:
    k = _extract(tmp_path)  # backlog (6-5)/6 = 0.1667 > 0.10
    assert k.insertion_backlog_fraction == pytest.approx(1 / 6)
    assert k.gridlock_censored is True
    # below threshold -> not censored
    jsonl, tinfo = _write(tmp_path)
    k2 = extract_kpis(jsonl, tinfo, warmup_s=10, episode_length_s=100,
                      episode_counters={"loaded_count": 100, "departed_count": 99, "arrived_count": 99})
    assert k2.gridlock_censored is False


def test_episode_kpi_fields_match_db_columns(tmp_path) -> None:
    k = _extract(tmp_path)
    fields = k.to_episode_kpi_fields()
    assert set(fields) == {
        "avg_waiting_time", "avg_queue_length", "throughput", "num_stops",
        "wait_p95", "fairness_std", "per_movement_max_wait",
    }
    assert len(fields["per_movement_max_wait"]) == 12


def test_empty_after_warmup_is_nan_not_crash(tmp_path) -> None:
    # all vehicles depart inside the warm-up -> nothing survives the cutoff
    jsonl, tinfo = _write(tmp_path)
    k = extract_kpis(jsonl, tinfo, warmup_s=1000, episode_length_s=2000,
                     episode_counters={"loaded_count": 5, "departed_count": 5, "arrived_count": 0})
    assert k.n_vehicles_after_warmup == 0
    assert math.isnan(k.avg_waiting_time)
    assert math.isnan(k.fairness_std)
    assert k.worst_movement_id is None
