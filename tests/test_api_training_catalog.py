"""Tests for the in-app training jobs and the model/comparison catalogue.

The training manager drives ``scripts/train_dqn.py`` as a child process and learns how far it has
got by PARSING ITS STDOUT. That is a contract between two modules with nothing but a log format
holding it together, so the regex is tested against a line produced by the real formatter rather
than a hand-typed approximation - a format change must fail here, not silently flat-line the UI's
progress bar at 0%.

No SUMO and no child process are started: the parsing, the command construction and the catalogue
are all pure functions over strings and directories.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.catalog import CONTROLLER_META, KPI_SPEC, _describe_run_dir
from src.api.server import create_app
from src.api.training import (
    MAX_UI_EPISODES,
    VARIANTS,
    _EPISODE_RE,
    TrainingManager,
    TrainingStatus,
    _build_command,
)


# ------------------------------------------------------------------------------------------
# Progress parsing - the cross-module contract
# ------------------------------------------------------------------------------------------


def _real_progress_line(ep: int, total: int, scenario: str, reward: float, eps: float, buf: int) -> str:
    """Reproduce ``src/ml/train_loop.py``'s episode line with ITS format spec, not a guess."""
    return (f"  ep {ep:3d}/{total}  {scenario}  "
            f"reward={reward:12,.1f}  eps={eps:.3f}  buffer={buf}")


def test_episode_regex_parses_the_real_log_line() -> None:
    match = _EPISODE_RE.match(_real_progress_line(12, 300, "SCN-01", -38.9, 0.994, 4320))
    assert match is not None
    assert int(match["ep"]) == 12
    assert int(match["total"]) == 300
    assert float(match["reward"].replace(",", "")) == pytest.approx(-38.9)
    assert float(match["eps"]) == pytest.approx(0.994)


def test_episode_regex_survives_the_thousands_separator() -> None:
    """``{:12,.1f}`` inserts commas past 999, which naive float() would choke on."""
    line = _real_progress_line(7, 300, "SCN-02", -12345.6, 0.05, 90000)
    match = _EPISODE_RE.match(line)
    assert match is not None
    assert "," in match["reward"]  # guard: this test is worthless if the format stops grouping
    assert float(match["reward"].replace(",", "")) == pytest.approx(-12345.6)


def test_episode_regex_ignores_other_output() -> None:
    for line in ("[matrix] === cell 1/9: plain seed=42 ===",
                 "Using libsumo as traci as requested by environment variable.",
                 "Traceback (most recent call last):",
                 ""):
        assert _EPISODE_RE.match(line) is None


# ------------------------------------------------------------------------------------------
# Command construction - each variant must reach train_dqn's matching branch
# ------------------------------------------------------------------------------------------


def test_plain_variant_passes_no_forecast_flags(tmp_path) -> None:
    cmd = _build_command("plain", 42, 30, None, tmp_path / "run")
    assert "--random-lstm" not in cmd and "--forecast-ckpt" not in cmd
    assert cmd[cmd.index("--variant") + 1] == "plain"
    assert cmd[cmd.index("--episodes") + 1] == "30"


def test_hybrid_variant_loads_that_seed_s_pinned_forecaster(tmp_path) -> None:
    """A6.4: the hybrid arm must resolve the forecaster through the per-seed guarded accessor."""
    from src.provenance.official import official_lstm_filename

    for seed in (42, 123, 2024):
        cmd = _build_command("hybrid", seed, 10, None, tmp_path / "run")
        ckpt = cmd[cmd.index("--forecast-ckpt") + 1]
        assert official_lstm_filename(seed) in ckpt, f"seed {seed} got another seed's forecaster"


def test_random_lstm_variant_uses_the_control_flag(tmp_path) -> None:
    cmd = _build_command("random-lstm", 42, 10, None, tmp_path / "run")
    assert "--random-lstm" in cmd
    assert "--forecast-ckpt" not in cmd  # train_dqn rejects both together


def test_short_runs_still_checkpoint_and_validate(tmp_path) -> None:
    """A 30-episode demo run must not inherit the 300-episode cadence and produce nothing."""
    cmd = _build_command("plain", 42, 30, None, tmp_path / "run")
    assert int(cmd[cmd.index("--checkpoint-every") + 1]) <= 30
    assert int(cmd[cmd.index("--validation-every") + 1]) <= 30


# ------------------------------------------------------------------------------------------
# Manager guards
# ------------------------------------------------------------------------------------------


class _NullHub:
    def publish(self, frame) -> None:  # pragma: no cover - never reached in these tests
        pass


@pytest.mark.parametrize("variant", ["", "iqn", "hybird", "PLAIN"])
def test_unknown_variant_is_refused(variant, tmp_path) -> None:
    mgr = TrainingManager(_NullHub(), runs_dir=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown variant"):
        mgr.start(variant=variant, seed=42, episodes=5)


@pytest.mark.parametrize("episodes", [0, -1, MAX_UI_EPISODES + 1])
def test_episode_count_is_bounded(episodes, tmp_path) -> None:
    """An unbounded box on a web form is how a two-hour job starts during a demo."""
    mgr = TrainingManager(_NullHub(), runs_dir=tmp_path)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="episodes must be between"):
        mgr.start(variant="plain", seed=42, episodes=episodes)


def test_status_pct_never_divides_by_zero() -> None:
    status = TrainingStatus(job_id="t-0", variant="plain", seed=1, episodes=0, label=None, run_dir="x")
    assert status.pct == 0.0
    assert status.as_dict()["pct"] == 0.0


def test_every_variant_is_buildable(tmp_path) -> None:
    """VARIANTS is what the API validates against; a name in it that cannot build is a 500."""
    for variant in VARIANTS:
        assert _build_command(variant, 42, 5, 300, tmp_path / "run")[0]


# ------------------------------------------------------------------------------------------
# Catalogue
# ------------------------------------------------------------------------------------------


def _make_run(root, name, *, episodes=300, last_ep=299, variant="plain", seed=42):
    run = root / name
    (run / "checkpoints").mkdir(parents=True)
    for ep in {0, last_ep}:
        (run / "checkpoints" / f"ep{ep}.pt").write_bytes(b"")
    (run / "config.yaml").write_text(
        f"variant: {variant}\nseed: {seed}\nn_episodes: {episodes}\nobs_dim: 20\ngit_sha: abc1234\n",
        encoding="utf-8",
    )
    return run


def test_describe_run_dir_reports_completeness(tmp_path) -> None:
    done = _describe_run_dir(_make_run(tmp_path, "plain_seed42"))
    assert done is not None and done["has_final"] is True and done["episodes_trained"] == 300
    partial = _describe_run_dir(_make_run(tmp_path, "plain_seed7", last_ep=50))
    assert partial is not None and partial["has_final"] is False


def test_describe_run_dir_ignores_directories_without_checkpoints(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    assert _describe_run_dir(tmp_path / "empty") is None


def test_describe_run_dir_survives_a_half_written_config(tmp_path) -> None:
    """config.yaml is written by a live run; a torn read must not break the whole listing."""
    run = _make_run(tmp_path, "plain_seed9")
    (run / "config.yaml").write_text("variant: [unclosed\n", encoding="utf-8")
    entry = _describe_run_dir(run)
    assert entry is not None and entry["id"] == "plain_seed9"


def test_models_endpoint_excludes_quarantined_directories(tmp_path) -> None:
    """Leading underscore is this project's quarantine marker for a superseded world."""
    import src.api.catalog as catalog

    _make_run(tmp_path, "plain_seed42")
    _make_run(tmp_path, "_pre_a6_2026-09-01")
    original = catalog._RUNS_DIR
    catalog._RUNS_DIR = tmp_path
    try:
        ids = [m["id"] for m in catalog.list_models()["models"]]
    finally:
        catalog._RUNS_DIR = original
    assert ids == ["plain_seed42"]


def test_ui_trained_models_are_marked_and_sorted_first(tmp_path) -> None:
    import src.api.catalog as catalog

    _make_run(tmp_path, "plain_seed42")
    _make_run(tmp_path, "ui_plain_seed42_t-1", episodes=30, last_ep=29)
    original = catalog._RUNS_DIR
    catalog._RUNS_DIR = tmp_path
    try:
        models = catalog.list_models()["models"]
    finally:
        catalog._RUNS_DIR = original
    assert models[0]["source"] == "user"
    assert models[1]["source"] == "matrix"


def test_kpi_spec_and_controller_meta_are_coherent() -> None:
    """The client marks best cells from lower_is_better alone; a missing flag silently inverts one."""
    assert all({"key", "label", "lower_is_better"} <= set(spec) for spec in KPI_SPEC)
    assert len({spec["key"] for spec in KPI_SPEC}) == len(KPI_SPEC)
    assert KPI_SPEC[1]["key"] == "throughput" and KPI_SPEC[1]["lower_is_better"] is False
    # Exactly the arms this project authored may be flagged as ours; a baseline never may.
    for name in ("webster", "max_pressure", "actuated"):
        assert CONTROLLER_META[name]["is_ours"] is False


# ------------------------------------------------------------------------------------------
# HTTP surface
# ------------------------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(db_path=tmp_path / "empty.db"))


def test_comparison_404s_when_the_scenario_has_no_episodes(client) -> None:
    response = client.get("/comparison", params={"scenario": "SCN-05"})
    assert response.status_code == 404
    assert "eval_runner" in response.json()["detail"]  # tells the reader what to run


def test_training_current_404s_before_any_job(client) -> None:
    assert client.get("/training/current").status_code == 404


def test_stopping_nothing_404s(client) -> None:
    assert client.delete("/training/current").status_code == 404


@pytest.mark.parametrize("body,expected", [
    ({"variant": "nope", "seed": 42, "episodes": 5}, 422),
    ({"variant": "plain", "seed": 42, "episodes": MAX_UI_EPISODES + 1}, 422),
    ({"variant": "plain", "seed": 42, "episodes": 0}, 422),
])
def test_training_rejects_bad_requests_without_starting_anything(client, body, expected) -> None:
    assert client.post("/training", json=body).status_code == expected
    assert client.get("/training/current").status_code == 404  # nothing was launched


def test_health_reports_the_training_channel(client) -> None:
    channels = client.get("/health").json()["channels"]
    assert "training" in channels
    assert set(channels["training"]) >= {"subscribers", "published", "dropped"}


def test_models_endpoint_is_json_serializable(client) -> None:
    """Path objects leaking into a response 500 at serialization time, not at construction."""
    response = client.get("/models")
    assert response.status_code == 200
    json.dumps(response.json())  # raises if anything non-serializable slipped in
