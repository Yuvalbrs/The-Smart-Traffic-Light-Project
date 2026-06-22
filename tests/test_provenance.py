"""T-01-06 - Tests for the provenance chain.

DoD: ``data_version`` is deterministic (same inputs -> same string); all three
layers are covered; and the documented ``run_id`` caveat holds - it is a UUID
that *records* inputs rather than reproducing them (identical inputs -> different
run_ids).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from src.db.engine import create_db_engine, init_db
from src.db.models import ExperimentRun
from src.provenance.records import record_experiment_run, record_model_artifact
from src.provenance.versions import (
    checkpoint_filename,
    config_hash,
    data_version,
    git_sha,
    hash_files,
    lstm_version,
    new_run_id,
    torch_versions,
)

_DV_INPUTS = dict(
    scenario_configs_hash="abc123",
    generator_git_sha="deadbeef",
    generation_seed=42,
    sumo_version="SUMO 1.27.0",
)


# --- data_version determinism (the headline DoD) ---

def test_data_version_is_deterministic() -> None:
    assert data_version(**_DV_INPUTS) == data_version(**_DV_INPUTS)
    assert data_version(**_DV_INPUTS).startswith("data-")


@pytest.mark.parametrize("field,new", [
    ("scenario_configs_hash", "xyz789"),
    ("generator_git_sha", "cafe0000"),
    ("generation_seed", 43),
    ("sumo_version", "SUMO 1.28.0"),
])
def test_data_version_changes_with_any_input(field, new) -> None:
    changed = {**_DV_INPUTS, field: new}
    assert data_version(**changed) != data_version(**_DV_INPUTS)


# --- lstm_version depends on data_version ---

def test_lstm_version_deterministic_and_chained() -> None:
    dv = data_version(**_DV_INPUTS)
    args = dict(data_version=dv, lstm_config_hash="cfg1", training_code_git_sha="sha1", training_seed=7)
    assert lstm_version(**args) == lstm_version(**args)
    assert lstm_version(**args).startswith("lstm-")
    # changing the upstream data_version changes the lstm_version
    other = lstm_version(**{**args, "data_version": "data-different"})
    assert other != lstm_version(**args)


# --- config_hash / hash_files are stable + order-independent ---

def test_config_hash_is_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_hash_files_is_order_independent_and_content_sensitive(tmp_path) -> None:
    f1 = tmp_path / "scn_01.yaml"
    f2 = tmp_path / "scn_02.yaml"
    f1.write_text("one", encoding="utf-8")
    f2.write_text("two", encoding="utf-8")
    before = hash_files([f1, f2])
    assert before == hash_files([f2, f1])  # order-independent
    f2.write_text("two-changed", encoding="utf-8")
    assert hash_files([f1, f2]) != before  # content change -> different hash


# --- run_id: a UUID that RECORDS inputs, not a reproducible hash of them ---

def test_run_id_is_a_unique_uuid() -> None:
    a, b = new_run_id(), new_run_id()
    assert a != b  # not derived from inputs - a fresh UUID each time
    uuid.UUID(a)  # parses as a valid UUID or raises
    uuid.UUID(b)


# --- checkpoint filename embeds the version chain ---

def test_checkpoint_filename_embeds_versions() -> None:
    dv, lv = "data-aaaa1111bbbb", "lstm-cccc2222dddd"
    name = checkpoint_filename("dqn", data_version=dv, lstm_version=lv, step=50000)
    assert dv in name and lv in name and "step50000" in name and name.endswith(".pt")
    # an LSTM checkpoint has no lstm_version segment
    lstm_name = checkpoint_filename("lstm", data_version=dv, step=20000)
    assert dv in lstm_name and "lstm-" not in lstm_name.replace("lstm__", "")


# --- environment collectors are best-effort and never raise ---

def test_env_collectors_dont_raise() -> None:
    sha = git_sha()
    assert sha is None or all(c in "0123456789abcdef" for c in sha)
    tv = torch_versions()
    assert set(tv) == {"torch", "cuda", "cudnn_deterministic"}  # torch likely absent now


# --- SQLite writers persist the full provenance tuple (records, not reproduces) ---

def test_record_run_persists_provenance(tmp_path) -> None:
    engine = create_db_engine(tmp_path / "p.db")
    init_db(engine)
    rid = new_run_id()
    dv = data_version(**_DV_INPUTS)
    with Session(engine) as s:
        run = record_experiment_run(
            s, name="r0", mode="training", controller="dqn", config={"lr": 1e-4},
            run_id=rid, data_version=dv, lstm_version="lstm-x", git_sha="sha0", sumo_version="SUMO 1.27.0",
        )
        record_model_artifact(
            s, run=run, kind="dqn",
            path=checkpoint_filename("dqn", data_version=dv, lstm_version="lstm-x", step=1000),
            step=1000, metrics={"val": 0.1},
        )
        s.commit()
        run_pk = run.id

    with Session(engine) as s:
        loaded = s.get(ExperimentRun, run_pk)
        assert loaded.run_id == rid
        assert loaded.data_version == dv
        assert loaded.lstm_version == "lstm-x"
        assert loaded.git_sha == "sha0"
        assert len(loaded.artifacts) == 1
        assert dv in loaded.artifacts[0].path


def test_identical_inputs_get_different_run_ids(tmp_path) -> None:
    """run_id RECORDS provenance; two runs with identical inputs still differ."""
    engine = create_db_engine(tmp_path / "p2.db")
    init_db(engine)
    dv = data_version(**_DV_INPUTS)
    with Session(engine) as s:
        r1 = record_experiment_run(
            s, name="a", mode="training", controller="dqn", config={}, run_id=new_run_id(), data_version=dv
        )
        r2 = record_experiment_run(
            s, name="b", mode="training", controller="dqn", config={}, run_id=new_run_id(), data_version=dv
        )
        s.commit()
        assert r1.data_version == r2.data_version  # same provenance
        assert r1.run_id != r2.run_id  # but distinct run identity
