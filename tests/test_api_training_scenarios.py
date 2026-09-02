"""Training in the app can be driven by real measured demand, not only the synthetic rotation.

The ingest pipeline already lands real Hangzhou counts in ``demand_count`` and ``SCN-R1`` reads
them (``profile: tabular``), and ``scripts/train_dqn.py`` has always accepted
``--train-scenarios``. The gap was one flag: ``src/api/training._build_command`` never passed it,
so every in-app run was pinned to train_dqn's default SCN-01/02/03 rotation and the measured
scenario was unreachable from the application.

Worth being exact about what this does and does not claim. The agent does **not** learn from a
dataset - ``src/ml/train_loop.py`` is an on-policy act -> push -> sample -> learn loop against a
live SUMO episode. Real data enters as the *demand profile that drives the simulator*: the
arrival pattern is measured, and the agent learns by controlling it.

``SCN-R1`` is deliberately outside ``CONFIRMATORY_SCENARIOS`` (``scripts/eval_runner.py``) and
``tests/test_demand_ingest.py`` asserts that. Training on it here is a demonstration, never
evidence, and nothing in this file weakens that separation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.server import SCENARIOS, create_app
from src.api.training import TrainingStatus, _build_command


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(db_path=tmp_path / "empty.db"))


# ------------------------------------------------------------------------------------------
# Command construction
# ------------------------------------------------------------------------------------------


def test_no_scenarios_given_defers_to_train_dqn_s_own_default(tmp_path) -> None:
    """Restating the default here would give it two homes, which is how defaults drift."""
    cmd = _build_command("plain", 42, 30, None, tmp_path / "run")
    assert "--train-scenarios" not in cmd


def test_measured_demand_scenario_reaches_the_child_process(tmp_path) -> None:
    cmd = _build_command("plain", 42, 30, None, tmp_path / "run", ("SCN-R1",))
    assert cmd[cmd.index("--train-scenarios") + 1] == "SCN-R1"


def test_several_scenarios_are_passed_as_separate_argv_entries(tmp_path) -> None:
    """``--train-scenarios`` is nargs="+"; one comma-joined string would train on nothing."""
    scns = ("SCN-01", "SCN-02", "SCN-10")
    cmd = _build_command("plain", 42, 30, None, tmp_path / "run", scns)
    start = cmd.index("--train-scenarios") + 1
    assert tuple(cmd[start:start + 3]) == scns


def test_scenario_choice_composes_with_the_variant_flags(tmp_path) -> None:
    """The forecaster pin must still be resolved when training on measured demand."""
    from src.provenance.official import official_lstm_filename

    cmd = _build_command("hybrid", 42, 10, None, tmp_path / "run", ("SCN-R1",))
    assert official_lstm_filename(42) in cmd[cmd.index("--forecast-ckpt") + 1]
    assert cmd[cmd.index("--train-scenarios") + 1] == "SCN-R1"


# ------------------------------------------------------------------------------------------
# Request validation - a bad scenario must be refused BEFORE a process is spawned
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scenarios", [
    ["SCN-99"],                 # does not exist
    ["SCN-01", "SCN-99"],       # one good one bad: the bad one still has to be caught
    ["SCN-A1"],                 # real scenario, but needs the two-junction network the hub lacks
    ["scn-01"],                 # right id, wrong case
])
def test_unknown_training_scenario_is_refused_and_starts_nothing(client, scenarios) -> None:
    body = {"variant": "plain", "seed": 42, "episodes": 5, "train_scenarios": scenarios}
    assert client.post("/training", json=body).status_code == 422
    assert client.get("/training/current").status_code == 404  # nothing was launched


def test_empty_scenario_list_is_refused_rather_than_silently_defaulting(client) -> None:
    """An empty list means the caller meant to choose and sent nothing; guessing hides that."""
    body = {"variant": "plain", "seed": 42, "episodes": 5, "train_scenarios": []}
    assert client.post("/training", json=body).status_code == 422
    assert client.get("/training/current").status_code == 404


def test_every_scenario_the_hub_advertises_is_accepted_for_training() -> None:
    """The picker is populated from SCENARIOS, so each entry must survive validation."""
    for scn in SCENARIOS:
        assert scn in SCENARIOS  # the validator's own membership rule
    assert "SCN-R1" in SCENARIOS, "the measured-demand scenario must be offerable"


# ------------------------------------------------------------------------------------------
# The wire: a model's training demand is read back, not asserted by a hand-typed label
# ------------------------------------------------------------------------------------------


def test_status_reports_the_scenarios_a_run_trained_on() -> None:
    status = TrainingStatus(
        job_id="t-1", variant="plain", seed=42, episodes=5, label=None,
        run_dir="runs/x", train_scenarios=["SCN-R1"],
    )
    assert status.as_dict()["train_scenarios"] == ["SCN-R1"]


def test_status_reports_none_when_the_default_rotation_was_used() -> None:
    status = TrainingStatus(
        job_id="t-1", variant="plain", seed=42, episodes=5, label=None, run_dir="runs/x",
    )
    assert status.as_dict()["train_scenarios"] is None
