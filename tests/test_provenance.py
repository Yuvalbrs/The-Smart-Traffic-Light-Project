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


# ------------------------------------------------------------------------------------------
# A6.4 - the pin is per DQN training seed
# ------------------------------------------------------------------------------------------


def test_every_training_seed_gets_its_own_forecaster():
    """No two DQN training seeds may share a forecaster.

    This is the whole point of A6.4. Under the single pin, three "independent" hybrid runs
    carried ONE forecaster, so forecaster-training variance was replicated zero times - while
    the gate verdict is known to flip sign across forecaster seeds on identical data. If a
    future edit collapses two seeds onto one checkpoint, the runs silently stop being
    independent and nothing else in the suite would notice.
    """
    from src.provenance.official import OFFICIAL_LSTM_BY_SEED, OFFICIAL_TRAIN_SEEDS

    assert set(OFFICIAL_TRAIN_SEEDS) == {42, 123, 2024}
    filenames = [OFFICIAL_LSTM_BY_SEED[s] for s in OFFICIAL_TRAIN_SEEDS]
    assert len(set(filenames)) == len(filenames), f"seeds share a forecaster: {filenames}"
    assert "PENDING" not in filenames


def test_no_seed_agnostic_accessor_exists():
    """A call site that cannot name its seed was silently mixing three experiments.

    Asserted behaviourally - each accessor is CALLED with no argument and must reject it -
    rather than by inspecting signatures, so re-adding a seed-agnostic default (the exact
    regression this guards) fails the test instead of passing a signature check.
    """
    import pytest as _pytest

    from src.provenance import official

    for name in ("official_lstm_filename", "official_lstm_path",
                 "official_lstm_version", "official_lstm_checked"):
        with _pytest.raises(TypeError):
            getattr(official, name)()


def test_unknown_seed_is_refused_and_names_the_pinned_seeds():
    """Borrowing another seed's forecaster must be impossible, not merely discouraged."""
    import pytest as _pytest

    from src.provenance.official import OFFICIAL_TRAIN_SEEDS, official_lstm_filename

    with _pytest.raises(KeyError) as exc:
        official_lstm_filename(7)
    for seed in OFFICIAL_TRAIN_SEEDS:
        assert str(seed) in str(exc.value)


def test_all_pins_agree_on_their_corpus():
    """Three seeds of ONE training process over ONE corpus; a disagreement means piecemeal pins."""
    from src.provenance.official import (
        OFFICIAL_LSTM_BY_SEED, OFFICIAL_TRAIN_SEEDS, official_data_version,
    )

    assert official_data_version().startswith("data-")
    tags = {f for s in OFFICIAL_TRAIN_SEEDS
            for f in OFFICIAL_LSTM_BY_SEED[s].split("__") if f.startswith("data-")}
    assert len(tags) == 1, f"pins disagree about the corpus: {tags}"


def test_corpus_guard_checks_every_seed_not_only_the_one_being_loaded(monkeypatch, tmp_path):
    """A guard that validated only the arm you happen to run would pass a half-stale matrix.

    Seed 42's pin is left matching the fabricated corpus and seed 2024's is made stale, and both
    checkpoint files are made to exist - so the ONLY thing that can fail is 2024's staleness. A
    guard that stopped at the first seed, or checked only the seed being loaded, returns clean
    here and fails this test.
    """
    import json

    import pytest as _pytest

    from scripts.train_lstm import _dataset_data_version
    from src.provenance import official

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(
        json.dumps([{"file": "a.csv", "data_version": "data-x", "file_sha256": "cd" * 32}]),
        encoding="utf-8")
    on_disk = _dataset_data_version(corpus)

    ckpts = tmp_path / "ckpts"
    ckpts.mkdir()
    pins = {
        42: f"lstm__{on_disk}__lstm-aaaaaaaaaaaa.pt",        # matches the fabricated corpus
        2024: "lstm__data-stalestale__lstm-bbbbbbbbbbbb.pt",  # does not
    }
    for name in pins.values():
        (ckpts / name).write_bytes(b"")  # existence only; the guard never opens them
    monkeypatch.setattr(official, "CKPT_DIR", ckpts)
    monkeypatch.setattr(official, "OFFICIAL_LSTM_BY_SEED", pins)
    monkeypatch.setattr(official, "OFFICIAL_TRAIN_SEEDS", (42, 2024))

    with _pytest.raises(RuntimeError, match="stale") as exc:
        official.assert_official_matches_corpus(corpus)
    assert "2024" in str(exc.value)


def test_split_guard_fires_when_the_pinned_checkpoints_trained_on_another_split(monkeypatch):
    """The corpus hash is structurally blind to the split - this is the guard that is not.

    ``data_version`` hashes all 100 CSVs whichever subset is read, so changing SPLITS leaves it
    identical and the corpus guard passes a checkpoint trained on different data. A6.4 put the
    split inside the checkpoint for exactly this; the guard reads it back.
    """
    import pytest as _pytest

    from src.ml import lstm_data
    from src.provenance.official import assert_official_matches_split

    assert_official_matches_split()  # the real pins match the real split
    monkeypatch.setattr(lstm_data, "SPLITS", {
        "train": ("SCN-01",), "val": ("SCN-04",), "test": ("SCN-05",),
    })
    with _pytest.raises(RuntimeError, match="DIFFERENT split"):
        assert_official_matches_split()
