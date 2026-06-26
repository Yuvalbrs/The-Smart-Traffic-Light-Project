"""T-03-08 - Tests for the sanity-gate verdict logic (pure, no SUMO).

Exercises ``_analyze`` against synthetic episode/step CSVs so the GO/NO-GO thresholds
(reward trend up, Q bounded, mask active) are pinned independently of a real run.
"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.sanity_gate import _analyze


def _write_run(tmp: Path, rewards, legal_means, q_maxes) -> None:
    with (tmp / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "ep_reward", "mask_legal_mean"])
        for i, (r, m) in enumerate(zip(rewards, legal_means)):
            w.writerow([i, r, m])
    with (tmp / "steps.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["total_step", "q_mean", "q_max"])
        for i, q in enumerate(q_maxes):
            w.writerow([i, q / 2.0, q])


def test_analyze_go_when_all_checks_pass(tmp_path):
    rewards = list(range(-100, 0, 5))  # strictly improving (20 episodes)
    legal = [5.0] * 20  # mask constrains (< 8)
    q = [10.0, 20.0, 15.0, 50.0]  # bounded, finite
    _write_run(tmp_path, rewards, legal, q)

    go, reasons, metrics = _analyze(tmp_path)
    assert go is True
    assert metrics["reward_slope_per_episode"] > 0
    assert 0.0 < metrics["mask_fire_rate"] < 1.0
    assert all("PASS" in r for r in reasons)


def test_analyze_nogo_on_exploding_q(tmp_path):
    rewards = list(range(-100, 0, 5))
    legal = [5.0] * 20
    q = [10.0, 1.0e6]  # explodes past the guard
    _write_run(tmp_path, rewards, legal, q)

    go, reasons, _ = _analyze(tmp_path)
    assert go is False
    assert any("FAIL" in r and "Q bounded" in r for r in reasons)


def test_analyze_nogo_on_dead_mask(tmp_path):
    rewards = list(range(-100, 0, 5))
    legal = [8.0] * 20  # never constrained -> mask is dead
    q = [10.0, 20.0]
    _write_run(tmp_path, rewards, legal, q)

    go, reasons, metrics = _analyze(tmp_path)
    assert go is False
    assert metrics["mask_fire_rate"] == 0.0
    assert any("FAIL" in r and "mask" in r for r in reasons)


def test_analyze_nogo_on_declining_reward(tmp_path):
    rewards = list(range(0, -100, -5))  # strictly worsening
    legal = [5.0] * 20
    q = [10.0, 20.0]
    _write_run(tmp_path, rewards, legal, q)

    go, reasons, _ = _analyze(tmp_path)
    assert go is False
    assert any("FAIL" in r and "reward trend" in r for r in reasons)
