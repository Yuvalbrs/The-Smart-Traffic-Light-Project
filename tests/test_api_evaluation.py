"""Tests for evaluating a user-trained model, and for what counts as a campaign.

The train -> evaluate -> compare loop only works if two things hold: a model trained in the app
can be evaluated into the results database, and doing so does not disturb the pre-registered
campaign already in there. The second is the subtle one and is the reason most of this file
exists - see :func:`test_a_user_evaluation_does_not_become_the_selected_campaign`.

No SUMO and no child process are started here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.evaluation import MAX_UI_SEEDS, EvaluationManager, EvaluationStatus, _EPISODE_RE, _label_for
from src.api.server import create_app


class _NullHub:
    def publish(self, frame) -> None:  # pragma: no cover - no job is started in these tests
        pass


# ------------------------------------------------------------------------------------------
# Progress parsing - the contract with eval_runner's stdout
# ------------------------------------------------------------------------------------------


def test_episode_regex_parses_a_real_eval_line() -> None:
    """Built with eval_runner's own format spec rather than typed by hand."""
    line = f"[eval] SCN-05 seed7000 {'ui:demo':<18} wait={23.9:6.1f} thru={1827.0:7.1f}"
    match = _EPISODE_RE.match(line)
    assert match is not None
    assert match["scenario"] == "SCN-05"
    assert match["seed"] == "7000"
    assert match["algo"] == "ui:demo"


def test_episode_regex_ignores_headers_and_noise() -> None:
    for line in ("[eval] === SCN-05 : 4 algos x 2 seeds ===",
                 "[eval] single user model: ui:demo <- runs\\ui_plain_seed42_t-1",
                 "[eval] partial run -> eval_results_partial_model-demo.csv",
                 "[eval] OK - 8 episodes -> ...",
                 "Traceback (most recent call last):"):
        assert _EPISODE_RE.match(line) is None


# ------------------------------------------------------------------------------------------
# Manager guards
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", ["../../etc/passwd", "a/b", "a\\b", "", ".", ".."])
def test_a_path_cannot_be_smuggled_through_model_id(model_id, tmp_path) -> None:
    """model_id arrives in an HTTP body and is spliced into a command line."""
    mgr = EvaluationManager(_NullHub(), runs_dir=tmp_path)
    with pytest.raises(ValueError, match="run-directory name"):
        mgr.start(model_id=model_id, scenario="SCN-05", seeds=1)


@pytest.mark.parametrize("seeds", [0, -1, MAX_UI_SEEDS + 1])
def test_seed_count_is_bounded(seeds, tmp_path) -> None:
    mgr = EvaluationManager(_NullHub(), runs_dir=tmp_path)
    (tmp_path / "m" / "checkpoints").mkdir(parents=True)
    with pytest.raises(ValueError, match="seeds must be between"):
        mgr.start(model_id="m", scenario="SCN-05", seeds=seeds)


def test_an_untrained_model_is_refused_with_a_useful_message(tmp_path) -> None:
    (tmp_path / "m").mkdir()
    mgr = EvaluationManager(_NullHub(), runs_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="train it before evaluating"):
        mgr.start(model_id="m", scenario="SCN-05", seeds=1)


def test_episodes_total_counts_the_baselines_too() -> None:
    """eval_runner evaluates the model AND the three baselines on shared seeds (paired design)."""
    status = EvaluationStatus(job_id="e-1", model_id="m", label="m", scenario="SCN-05", seeds=5)
    assert status.episodes_total == 20
    assert status.pct == 0.0
    status.episodes_done = 10
    assert status.pct == 50.0


def test_label_is_sanitized_for_the_command_line_and_sql() -> None:
    assert _label_for("ui_plain_seed42_t-1") == "plain_seed42_t-1"
    assert "/" not in _label_for("weird/name")
    assert " " not in _label_for("a b;drop table")
    assert _label_for("!!!") == "---"  # non-empty fallback, never an empty label


# ------------------------------------------------------------------------------------------
# What counts as a campaign
# ------------------------------------------------------------------------------------------


def _seed_db(engine, *, sha: str, controllers: list[str], scenario: str, when: str, n: int = 2):
    """Insert n episodes per controller under one git_sha, with KPIs."""
    from sqlalchemy import text as _t

    with engine.begin() as conn:
        for controller in controllers:
            run = conn.execute(_t(
                "INSERT INTO experiment_run (schema_version, name, mode, controller, config, "
                "git_sha, created_at) VALUES ('1.1.0', :n, 'eval', :c, '{}', :s, :w)"
            ), {"n": f"eval-{controller}", "c": controller, "s": sha, "w": when}).lastrowid
            for i in range(n):
                ep = conn.execute(_t(
                    "INSERT INTO episode (schema_version, run_id_fk, index_in_run, seed, scenario, "
                    "total_reward, gridlock_censored) VALUES ('1.1.0', :r, :i, :sd, :sc, 0, 0)"
                ), {"r": run, "i": i, "sd": 7000 + i, "sc": scenario}).lastrowid
                conn.execute(_t(
                    "INSERT INTO episode_kpi (schema_version, episode_id_fk, avg_waiting_time, "
                    "throughput) VALUES ('1.1.0', :e, :w, 1000.0)"
                ), {"e": ep, "w": 10.0})


@pytest.fixture()
def app_with_db(tmp_path):
    app = create_app(db_path=tmp_path / "t.db")
    from src.db.engine import init_db

    init_db(app.state.engine)
    return app


def test_a_user_evaluation_does_not_become_the_selected_campaign(app_with_db) -> None:
    """The regression this file exists for.

    Evaluating one user model also writes baseline rows, so its git_sha is the NEWEST in the
    database. Selecting the campaign by recency alone therefore replaced a full campaign with a
    handful of episodes from a single UI click - silently, and with the comparison still looking
    perfectly plausible.
    """
    engine = app_with_db.state.engine
    _seed_db(engine, sha="campaign1", controllers=["plain", "hybrid", "random-lstm", "webster"],
             scenario="SCN-05", when="2026-09-01 10:00:00", n=15)
    _seed_db(engine, sha="userclick", controllers=["ui:mine", "webster"],
             scenario="SCN-05", when="2026-09-01 23:00:00", n=2)

    body = TestClient(app_with_db).get("/comparison", params={"scenario": "SCN-05"}).json()
    assert body["provenance"]["git_sha"] == "campaign1", "a UI click was selected as the campaign"
    controllers = {r["controller"] for r in body["rows"]}
    assert {"plain", "hybrid", "random-lstm"} <= controllers  # the campaign's arms are all present
    assert "ui:mine" in controllers                            # and the user model rides alongside
    assert [r for r in body["rows"] if r["controller"] == "plain"][0]["n_episodes"] == 15


def test_user_model_rows_are_flagged_and_their_code_version_is_reported(app_with_db) -> None:
    engine = app_with_db.state.engine
    _seed_db(engine, sha="campaign1", controllers=["plain", "hybrid", "random-lstm"],
             scenario="SCN-05", when="2026-09-01 10:00:00", n=15)
    _seed_db(engine, sha="userclick", controllers=["ui:mine"],
             scenario="SCN-05", when="2026-09-01 23:00:00", n=2)

    body = TestClient(app_with_db).get("/comparison", params={"scenario": "SCN-05"}).json()
    by_controller = {r["controller"]: r for r in body["rows"]}
    assert by_controller["ui:mine"]["is_user_model"] is True
    assert by_controller["plain"]["is_user_model"] is False
    # The reader must be able to see that this row came from different code than the campaign.
    assert body["provenance"]["user_model_shas"] == ["userclick"]
    assert by_controller["ui:mine"]["label"] == "mine"  # the ui: prefix is not shown raw


def test_campaign_listing_excludes_ad_hoc_evaluations(app_with_db) -> None:
    engine = app_with_db.state.engine
    _seed_db(engine, sha="campaign1", controllers=["plain", "hybrid", "random-lstm"],
             scenario="SCN-05", when="2026-09-01 10:00:00", n=15)
    _seed_db(engine, sha="userclick", controllers=["ui:mine", "webster"],
             scenario="SCN-05", when="2026-09-01 23:00:00", n=2)

    body = TestClient(app_with_db).get("/comparison", params={"scenario": "SCN-05"}).json()
    shas = [c["git_sha"] for c in body["provenance"]["available_campaigns"]]
    assert shas == ["campaign1"], f"a UI evaluation was listed as a campaign: {shas}"


def test_an_explicit_campaign_can_still_be_requested(app_with_db) -> None:
    engine = app_with_db.state.engine
    _seed_db(engine, sha="old", controllers=["plain", "hybrid", "random-lstm"],
             scenario="SCN-05", when="2026-06-01 10:00:00", n=5)
    _seed_db(engine, sha="new", controllers=["plain", "hybrid", "random-lstm"],
             scenario="SCN-05", when="2026-09-01 10:00:00", n=15)
    client = TestClient(app_with_db)
    assert client.get("/comparison", params={"scenario": "SCN-05"}).json()["provenance"]["git_sha"] == "new"
    older = client.get("/comparison", params={"scenario": "SCN-05", "git_sha": "old"}).json()
    assert older["provenance"]["git_sha"] == "old"
    assert [r for r in older["rows"] if r["controller"] == "plain"][0]["n_episodes"] == 5


# ------------------------------------------------------------------------------------------
# HTTP surface
# ------------------------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(db_path=tmp_path / "e.db"))


def test_evaluation_current_404s_before_any_job(client) -> None:
    assert client.get("/evaluation/current").status_code == 404


def test_stopping_nothing_404s(client) -> None:
    assert client.delete("/evaluation/current").status_code == 404


def test_unknown_scenario_is_refused(client) -> None:
    r = client.post("/evaluation", json={"model_id": "whatever", "scenario": "SCN-99", "seeds": 2})
    assert r.status_code == 422


def test_unknown_model_is_a_404_not_a_500(client) -> None:
    r = client.post("/evaluation", json={"model_id": "does_not_exist", "scenario": "SCN-05", "seeds": 2})
    assert r.status_code == 404
    assert "train it" in r.json()["detail"]


def test_path_traversal_over_http_is_a_422(client) -> None:
    r = client.post("/evaluation", json={"model_id": "../../secrets", "scenario": "SCN-05", "seeds": 1})
    assert r.status_code == 422
    assert client.get("/evaluation/current").status_code == 404  # nothing was launched


def test_health_reports_the_evaluation_channel(client) -> None:
    assert "evaluation" in client.get("/health").json()["channels"]
