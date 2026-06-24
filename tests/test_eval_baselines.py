"""T-04-05 - Tests for the baseline eval dry-run.

Measurable DoD parts: (1) the env now emits a non-empty trip-info XML when asked
(the wiring this task adds); (2) one episode through the pipeline yields all 7
KPIs; (3) the dry-run produces a table covering the 3 baselines. Short episodes +
warm-up keep the SUMO-backed suite fast; SCN-02 is heavy so it actually produces
trips inside a short window.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from scripts.build_actuated import build_actuated_add
from scripts.build_network import build_net
from scripts.build_routes import write_routes
from scripts.eval_baselines import CONTROLLERS, run_episode
from src.env.sumo_env import SUMOEnv
from src.metrics.kpi_extractor import EpisodeKPIs
from src.scenarios.config import load_all


@pytest.fixture(scope="module", autouse=True)
def _built() -> None:
    build_net()
    build_actuated_add()


def _scn(scenario_id: str = "SCN-02"):
    return next(s for s in load_all() if s.id == scenario_id)


def test_env_emits_nonempty_tripinfo(tmp_path) -> None:
    """The new tripinfo_path wiring makes SUMO write per-vehicle trip rows."""
    scn = _scn()
    route = write_routes(scn, 0)
    tripinfo = tmp_path / "trips.xml"
    env = SUMOEnv(route, episode_length_s=120, sumo_seed=0, tripinfo_path=tripinfo)
    try:
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
    finally:
        env.close()  # SUMO finalizes the XML on close

    assert tripinfo.exists()
    trips = list(ET.parse(tripinfo).getroot().iter("tripinfo"))
    assert trips, "heavy scenario should complete at least one trip in 120 s"
    # the two attributes the KPI extractor depends on are present
    assert trips[0].get("waitingTime") is not None
    assert trips[0].get("waitingCount") is not None


def test_tripinfo_off_by_default(tmp_path) -> None:
    scn = _scn()
    route = write_routes(scn, 0)
    env = SUMOEnv(route, episode_length_s=20)  # no tripinfo_path
    try:
        assert env._tripinfo_path is None
        assert "--tripinfo-output" not in env._sumo_args()
    finally:
        env.close()


def test_reset_reuse_without_fresh_output_path_raises(tmp_path) -> None:
    """H1: reusing one env across episodes with a FIXED output path would let SUMO's
    traci.load truncate the prior file - the guard must catch it, and a per-episode
    path via reset(options=...) must be allowed."""
    scn = _scn()
    route = write_routes(scn, 0)
    env = SUMOEnv(route, episode_length_s=60, sumo_seed=0, tripinfo_path=tmp_path / "a.xml")
    try:
        env.reset()
        with pytest.raises(RuntimeError, match="tripinfo_path"):
            env.reset()  # reuse with the same fixed path -> would overwrite -> guard
        env.reset(options={"tripinfo_path": str(tmp_path / "b.xml")})  # fresh path is OK
    finally:
        env.close()


def test_warmup_past_episode_length_raises(tmp_path) -> None:
    """M2: an empty KPI window must fail loudly, not return silent NaNs."""
    scn = _scn()
    with pytest.raises(ValueError, match="empty"):
        run_episode(scn, 0, "webster", work_dir=tmp_path, episode_length_s=120, warmup_s=300.0)


def test_run_episode_returns_seven_kpis(tmp_path) -> None:
    scn = _scn()
    kpis = run_episode(
        scn, 0, "webster", work_dir=tmp_path, episode_length_s=180, warmup_s=0.0
    )
    assert isinstance(kpis, EpisodeKPIs)
    # all 7 reported scalars are finite numbers on a real (non-warmed-out) episode
    for field in ("avg_waiting_time", "avg_queue_length", "throughput", "num_stops",
                  "wait_p95", "fairness_std"):
        assert not np.isnan(getattr(kpis, field)), f"{field} is NaN"
    assert kpis.arrived_count > 0


def test_dryrun_covers_all_three_baselines(tmp_path) -> None:
    """One seed through every baseline -> a paired 3-controller x 7-KPI table."""
    scn = _scn()
    table: dict[str, EpisodeKPIs] = {}
    for controller in CONTROLLERS:
        table[controller] = run_episode(
            scn, 0, controller, work_dir=tmp_path, episode_length_s=180, warmup_s=0.0
        )
    assert set(table) == set(CONTROLLERS)
    assert all(k.arrived_count > 0 for k in table.values())
