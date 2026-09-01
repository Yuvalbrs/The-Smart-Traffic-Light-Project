"""T-05-04 / T-05-01 - REST replay + session-control endpoint tests.

Everything here runs against a throwaway SQLite file and throwaway JSONL traces, and no test
starts a simulation: the DoD asks for schema-shape and backpressure coverage without live SUMO
in the CI path, and a 3600 s episode has no business inside a unit test.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api.server import create_app
from src.db.engine import create_db_engine, init_db
from src.db.models import Episode, EpisodeKpi, ExperimentRun

RUN_ID = "11111111-2222-3333-4444-555555555555"
EMPTY_RUN_ID = "99999999-0000-0000-0000-000000000000"


@pytest.fixture()
def client(tmp_path):
    """A TestClient wired to a temp DB holding one run of two episodes, plus one trace file."""
    db_path = tmp_path / "traffic.db"
    engine = create_db_engine(db_path)
    init_db(engine)
    with Session(engine) as session:
        run = ExperimentRun(
            name="eval-webster",
            mode="eval",
            controller="webster",
            config=json.dumps({"eval_seeds": [7000, 7001]}),
            run_id=RUN_ID,
            git_sha="abc1234",
            sumo_version="Eclipse SUMO sumo 1.27.0",
            data_version="data-8eb28eecdefb",
            lstm_version="lstm-df67afd839d4",
        )
        empty = ExperimentRun(name="eval-empty", mode="eval", controller="actuated",
                              config="{}", run_id=EMPTY_RUN_ID)
        session.add_all([run, empty])
        session.flush()
        for i, (seed, censored) in enumerate([(7000, 0), (7001, 1)]):
            ep = Episode(
                run_id_fk=run.id,
                index_in_run=i,
                seed=seed,
                scenario="SCN-05",
                total_reward=-1943.0,
                insertion_backlog_fraction=0.0 if not censored else 0.73,
                gridlock_censored=censored,
            )
            session.add(ep)
            session.flush()
            session.add(
                EpisodeKpi(
                    episode_id_fk=ep.id,
                    avg_waiting_time=3.83 + i,
                    avg_queue_length=12.0,
                    throughput=1221.0,
                    num_stops=1.2,
                    wait_p95=40.0,
                    fairness_std=9.7,
                    worst_movement_max_wait=62.0,
                )
            )
        session.commit()

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace = trace_dir / ("SCN-05_seed7000_webster.jsonl")
    with trace.open("w", encoding="utf-8") as fh:
        for seq in range(5):
            fh.write(json.dumps({"schema_version": "1.1.0", "type": "sim_frame",
                                 "seq": seq, "sim_time": float(seq)}) + "\n")

    app = create_app(db_path=db_path, trace_dirs=[trace_dir])
    with TestClient(app) as c:
        yield c


def test_health_reports_channels_and_backpressure_settings(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["session"] is None
    # Exact set, not a superset: a channel silently disappearing from /health is how a dead
    # feed goes unnoticed. "training" joined dash/unity when in-app training landed.
    assert set(body["channels"]) == {"dash", "unity", "training"}
    for channel in body["channels"].values():
        assert channel["queue_maxsize"] == 8
        assert channel["dropped"] == 0


def test_controllers_lists_the_product_and_all_three_baselines(client) -> None:
    body = client.get("/controllers").json()
    assert "sel/plain" in body["controllers"]
    for baseline in ("webster", "max_pressure", "actuated"):
        assert baseline in body["controllers"], baseline
    assert "SCN-05" in body["scenarios"]


def test_list_runs_is_paginated_newest_first(client) -> None:
    """Newest-first with limit/offset; the two pages must not overlap."""
    page1 = client.get("/runs", params={"limit": 1}).json()
    page2 = client.get("/runs", params={"limit": 1, "offset": 1}).json()
    assert page1["total"] == page2["total"] == 2
    assert len(page1["runs"]) == len(page2["runs"]) == 1

    first, second = page1["runs"][0], page2["runs"][0]
    assert first["run_id"] != second["run_id"]
    assert first["run_id"] == EMPTY_RUN_ID, "newest run (highest id) comes first"

    # The provenanced run carries its chain; the bare one honestly reports nulls.
    provenanced = next(r for r in (first, second) if r["run_id"] == RUN_ID)
    assert provenanced["version_chain"]["git_sha"] == "abc1234"
    assert provenanced["controller"] == "webster"


def test_metadata_carries_the_full_version_chain_and_episodes(client) -> None:
    body = client.get("/runs/" + RUN_ID + "/metadata").json()
    assert body["controller"] == "webster"
    assert body["episode_count"] == 2
    assert body["scenarios"] == ["SCN-05"]
    assert body["seeds"] == [7000, 7001]
    chain = body["version_chain"]
    assert chain["data_version"] == "data-8eb28eecdefb"
    assert chain["lstm_version"] == "lstm-df67afd839d4"
    assert body["config"]["eval_seeds"] == [7000, 7001]


def test_kpis_table_exposes_the_censoring_flag(client) -> None:
    """A censored episode's KPIs are not comparable; a client must be able to say so."""
    body = client.get("/runs/" + RUN_ID + "/kpis").json()
    assert "gridlock_censored" in body["columns"]
    assert [r["gridlock_censored"] for r in body["rows"]] == [False, True]
    assert body["rows"][0]["worst_movement_max_wait"] == 62.0


def test_trace_streams_ndjson_frames(client) -> None:
    resp = client.get("/runs/" + RUN_ID + "/trace")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    lines = [ln for ln in resp.text.splitlines() if ln.strip()]
    assert len(lines) == 5
    assert json.loads(lines[0])["type"] == "sim_frame"


def test_trace_limit_truncates_the_stream(client) -> None:
    resp = client.get("/runs/" + RUN_ID + "/trace", params={"limit": 2})
    assert len([ln for ln in resp.text.splitlines() if ln.strip()]) == 2


def test_timeline_is_the_documented_alias_of_trace(client) -> None:
    a = client.get("/runs/" + RUN_ID + "/trace").text
    b = client.get("/runs/" + RUN_ID + "/timeline").text
    assert a == b


def test_missing_run_is_404(client) -> None:
    assert client.get("/runs/not-a-run/metadata").status_code == 404


def test_missing_trace_file_explains_where_it_looked(client) -> None:
    resp = client.get("/runs/" + EMPTY_RUN_ID + "/trace")
    assert resp.status_code == 404
    assert "JSONL" in resp.json()["detail"]


def test_start_session_rejects_unknown_controller_and_scenario(client) -> None:
    assert client.post("/sessions", json={"controller": "ppo"}).status_code == 422
    assert client.post(
        "/sessions", json={"controller": "webster", "scenario": "SCN-99"}
    ).status_code == 422


def test_no_session_yet_is_404(client) -> None:
    assert client.get("/sessions/current").status_code == 404
    assert client.delete("/sessions/current").status_code == 404
