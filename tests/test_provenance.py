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


# ---------------------------------------------------------------------------
# 2026-09-01: the LSTM-corpus provenance chain. Three linked defects, found when
# a regenerated manifest showed 100 files carrying only 10 data_versions.
# ---------------------------------------------------------------------------


def test_data_version_distinguishes_datasets_sharing_every_other_input():
    """The corpus defect: ten scenarios at one seed shared all four original inputs.

    scenario_configs_hash is a hash of ALL scenario YAMLs, so it is constant across scenarios;
    git sha, SUMO version and seed are too. Ten genuinely different CSVs were therefore minted
    with ONE data_version, and an id that cannot tell SCN-01 from SCN-10 cannot carry the
    provenance chain preregistration s9 requires.
    """
    a = data_version(**_DV_INPUTS, dataset_key="SCN-01")
    b = data_version(**_DV_INPUTS, dataset_key="SCN-10")
    assert a != b
    assert a.startswith("data-") and b.startswith("data-")


def test_data_version_without_a_dataset_key_is_unchanged():
    """Backward compatibility: ids minted before the parameter existed must not move."""
    assert data_version(**_DV_INPUTS, dataset_key="") == data_version(**_DV_INPUTS)


def test_data_version_dataset_key_is_deterministic():
    assert (data_version(**_DV_INPUTS, dataset_key="SCN-04")
            == data_version(**_DV_INPUTS, dataset_key="SCN-04"))


def test_dataset_version_follows_the_bytes_not_just_the_config(tmp_path):
    """The checkpoint-level defect.

    _dataset_data_version aggregated per-file data_versions only. Those are derived from
    configuration, so no change in the CSV bytes could ever move the dataset id - and a corpus
    of 10x SCN-01 produced the same id as the real 10-scenario corpus. It now aggregates the
    per-file content digests, so the id names the data it was actually trained on.
    """
    import json as _json

    from scripts.train_lstm import _dataset_data_version

    def _write(shas):
        (tmp_path / "manifest.json").write_text(
            _json.dumps([{"file": f"f{i}.csv", "data_version": "data-same", "file_sha256": s}
                         for i, s in enumerate(shas)]), encoding="utf-8")

    _write(["aa" * 32, "bb" * 32])
    v1 = _dataset_data_version(tmp_path)
    _write(["aa" * 32, "cc" * 32])          # same config, one file's CONTENT changed
    v2 = _dataset_data_version(tmp_path)
    assert v1 != v2, "a changed corpus must change the dataset version"
    _write(["bb" * 32, "aa" * 32])          # order must not matter
    assert _dataset_data_version(tmp_path) == v1


def test_dataset_version_falls_back_for_pre_amendment_manifests(tmp_path):
    """Manifests written before file_sha256 existed still resolve, via the old aggregation."""
    import json as _json

    from scripts.train_lstm import _dataset_data_version

    (tmp_path / "manifest.json").write_text(
        _json.dumps([{"file": "a.csv", "data_version": "data-1"},
                     {"file": "b.csv", "data_version": "data-2"}]), encoding="utf-8")
    assert _dataset_data_version(tmp_path).startswith("data-")


def test_official_pin_staleness_guard_fires_on_a_regenerated_corpus(tmp_path):
    """The single-pin design fixed forecaster drift; it never caught CORPUS drift.

    The pin can stay valid-looking while `data/lstm/` is regenerated underneath it, so every run
    keeps loading a real checkpoint that no longer matches its own data_version. That is silent,
    and it poisons the provenance chain preregistration s9 promises. The guard makes it loud.
    """
    import json

    import pytest as _pytest

    from src.provenance.official import assert_official_matches_corpus, official_data_version

    (tmp_path / "manifest.json").write_text(
        json.dumps([{"file": "a.csv", "data_version": "data-x", "file_sha256": "ab" * 32}]),
        encoding="utf-8")
    with _pytest.raises(RuntimeError, match="stale"):
        assert_official_matches_corpus(tmp_path)
    # and it names both sides so the fix is obvious from the message alone
    try:
        assert_official_matches_corpus(tmp_path)
    except RuntimeError as exc:
        assert official_data_version() in str(exc)
