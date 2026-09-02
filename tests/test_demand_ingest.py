"""Measured demand: ingesting real counts, and driving a scenario from the database.

Until now the database was a pure SINK - results flowed in, nothing flowed back out into a
simulation - and every scenario's traffic came from a formula in a YAML file. These tests cover
the other direction: a published dataset is normalised into ``demand_count`` and a scenario reads
its arrival rates from there.

The arithmetic that matters most is the AVERAGE in :func:`load_measured_bins`.
``build_routes._APPROACH_AXIS`` gives every approach on an axis the full axis rate, so summing the
two measured approaches would silently double the traffic the dataset recorded. That is tested
explicitly, because it is invisible in any single number and would look like a plausible result.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy.orm import Session

from scripts.ingest_demand import approach_of, bin_counts, read_flow
from src.db.engine import create_db_engine, init_db
from src.db.models import DemandCount
from src.scenarios.config import AxisDemand, ScenarioError, load_measured_bins


# ------------------------------------------------------------------------------------------
# Reading the source format
# ------------------------------------------------------------------------------------------


def test_road_id_maps_to_the_approach_it_arrives_from() -> None:
    """A road pointing east is entered from the WEST - the inversion is easy to get backwards."""
    assert approach_of("road_0_1_0") == "W"   # points east  -> west approach
    assert approach_of("road_1_0_1") == "S"   # points north -> south approach
    assert approach_of("road_2_1_2") == "E"   # points west  -> east approach
    assert approach_of("road_1_2_3") == "N"   # points south -> north approach


def test_an_unreadable_road_id_is_refused() -> None:
    with pytest.raises(ValueError, match="direction"):
        approach_of("road_1_1_9")


def _flow_entry(start: float, road: str, end: float | None = None) -> dict:
    return {
        "vehicle": {"length": 5.0},
        "route": [road, "road_1_1_0"],
        "interval": 5,
        "startTime": start,
        "endTime": start if end is None else end,
    }


def test_one_entry_is_one_vehicle(tmp_path) -> None:
    path = tmp_path / "flow.json"
    path.write_text(json.dumps([_flow_entry(1, "road_0_1_0"), _flow_entry(2, "road_1_0_1")]))
    assert read_flow(path) == [(1.0, "W"), (2.0, "S")]


def test_a_spanning_entry_is_refused_rather_than_undercounted(tmp_path) -> None:
    """`interval` means "repeat every N seconds" when endTime > startTime.

    Every entry in the shipped benchmark is a single vehicle, so this reader assumes it - but it
    says so out loud instead of quietly counting a 300-second flow as one car.
    """
    path = tmp_path / "flow.json"
    path.write_text(json.dumps([_flow_entry(10, "road_0_1_0", end=310)]))
    with pytest.raises(SystemExit, match="one vehicle per entry"):
        read_flow(path)


# ------------------------------------------------------------------------------------------
# Binning
# ------------------------------------------------------------------------------------------


def test_empty_bins_are_emitted_not_skipped() -> None:
    """A missing row and a zero are different claims.

    A skipped quiet bin would leave the step-lookup holding the previous rate across it, i.e.
    inventing traffic that the measurement says was not there.
    """
    rows = bin_counts([(0.0, "N"), (700.0, "N")], bin_seconds=300.0)
    north = [r for r in rows if r[0] == "N"]
    assert [r[3] for r in north] == [1, 0, 1]
    # and every approach appears, even ones with no vehicles at all
    assert {r[0] for r in rows} == {"N", "E", "S", "W"}


def test_binning_conserves_the_vehicle_count() -> None:
    vehicles = [(float(t), "N" if t % 2 else "S") for t in range(0, 1000, 7)]
    rows = bin_counts(vehicles, bin_seconds=300.0)
    assert sum(r[3] for r in rows) == len(vehicles)


# ------------------------------------------------------------------------------------------
# Database -> demand profile
# ------------------------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    """A database holding one hour of counts: N busy, S quiet, so an average is distinguishable."""
    path = tmp_path / "t.db"
    engine = create_db_engine(path)
    init_db(engine)
    with Session(engine) as session:
        session.add_all([
            DemandCount(source="probe", approach="N", bin_start_s=0, bin_end_s=300, vehicles=100),
            DemandCount(source="probe", approach="S", bin_start_s=0, bin_end_s=300, vehicles=50),
            DemandCount(source="probe", approach="N", bin_start_s=300, bin_end_s=600, vehicles=20),
            DemandCount(source="probe", approach="S", bin_start_s=300, bin_end_s=600, vehicles=10),
        ])
        session.commit()
    return path


def test_axis_rate_is_the_MEAN_of_its_approaches_not_the_sum(db) -> None:
    """The load-bearing arithmetic.

    build_routes gives BOTH approaches on an axis the full axis rate, so summing here would
    double the measured traffic - and the result would still look entirely plausible.
    """
    bins = load_measured_bins("probe", ["N", "S"], db_path=db)
    assert len(bins) == 2
    # bin 0: mean(100, 50) = 75 vehicles per 300 s = 900 veh/h.  The sum would give 1800.
    assert bins[0] == (0.0, 300.0, 900.0)
    assert bins[1] == (300.0, 600.0, 180.0)


def test_bins_come_back_in_time_order(db) -> None:
    bins = load_measured_bins("probe", ["N", "S"], db_path=db)
    assert [b[0] for b in bins] == sorted(b[0] for b in bins)


def test_an_unknown_source_returns_nothing(db) -> None:
    assert load_measured_bins("no-such-dataset", ["N"], db_path=db) == ()


def test_a_missing_database_says_how_to_create_one(tmp_path) -> None:
    with pytest.raises(ScenarioError, match="ingest_demand"):
        load_measured_bins("probe", ["N"], db_path=tmp_path / "absent.db")


# ------------------------------------------------------------------------------------------
# The profile itself
# ------------------------------------------------------------------------------------------


def test_rate_steps_across_bins_and_never_interpolates(monkeypatch) -> None:
    """Each row is a COUNT over an interval, so the rate is constant across it by construction.

    Interpolating would invent a within-bin shape the measurement does not contain.
    """
    import src.scenarios.config as cfg

    monkeypatch.setattr(cfg, "_measured_bins",
                        lambda *_: ((0.0, 300.0, 900.0), (300.0, 600.0, 180.0)))
    demand = AxisDemand(profile="tabular", params={}, source="x", approaches=("N",))
    assert demand.rate_at(0) == 900.0
    assert demand.rate_at(299.9) == 900.0     # still the first bin's rate at its very end
    assert demand.rate_at(300) == 180.0       # steps, not ramps
    assert demand.rate_at(450) == 180.0


def test_past_the_end_of_the_record_the_last_rate_holds(monkeypatch) -> None:
    """An episode longer than the data should read as "more of the same", not an empty road.

    Dropping to zero would look exactly like a broken simulator to anyone watching.
    """
    import src.scenarios.config as cfg

    monkeypatch.setattr(cfg, "_measured_bins", lambda *_: ((0.0, 300.0, 900.0),))
    demand = AxisDemand(profile="tabular", params={}, source="x", approaches=("N",))
    assert demand.rate_at(10_000) == 900.0


def test_a_tabular_axis_with_no_data_refuses_to_invent_a_rate(monkeypatch) -> None:
    import src.scenarios.config as cfg

    monkeypatch.setattr(cfg, "_measured_bins", lambda *_: ())
    demand = AxisDemand(profile="tabular", params={}, source="ghost", approaches=("N",))
    with pytest.raises(ScenarioError, match="ingest_demand"):
        demand.rate_at(0)


# ------------------------------------------------------------------------------------------
# The real scenario, when the real data has been ingested
# ------------------------------------------------------------------------------------------


def test_listing_every_scenario_does_not_require_a_database() -> None:
    """load_all() globs every file, so an eager database read would break a fresh clone.

    This is the same failure the vault-path bug had: something the runtime needs, living
    somewhere a clone does not have. The measured series is therefore fetched on first USE.
    """
    from src.scenarios.config import load_all

    scenarios = load_all()
    assert len(scenarios) >= 10
    measured = [s for s in scenarios if s.id == "SCN-R1"]
    assert measured, "SCN-R1 should be discoverable without touching the database"
    assert measured[0].ns.profile == "tabular"
    assert measured[0].ns.source == "hangzhou_bc-tyc_18041607"


def test_the_measured_scenario_is_not_in_the_confirmatory_set() -> None:
    """Adding a scenario to the pre-registered set after the fact makes it post-hoc."""
    from scripts.eval_runner import CONFIRMATORY_SCENARIOS

    assert "SCN-R1" not in CONFIRMATORY_SCENARIOS
