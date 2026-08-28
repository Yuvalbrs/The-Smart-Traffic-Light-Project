"""T-04-02 - Tests for the analysis stats helpers (the conclusions depend on these being right)."""

from __future__ import annotations

import math

import numpy as np

from scripts.analyze_eval import _holm, _wilcoxon


def test_holm_bonferroni_known_example():
    # p = [0.01, 0.04, 0.03], m=3. Holm: sort asc, step-down (m-rank)*p, enforce monotone.
    adj = _holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03  # 3 * 0.01
    assert adj[2] == 0.06  # 2 * 0.03
    assert adj[1] == 0.06  # max(1 * 0.04, 0.06) monotone
    assert adj[1] >= adj[2] >= adj[0]  # non-decreasing along the sorted order


def test_holm_ignores_nan_in_family_size():
    # one NaN -> family size m=2, not 3.
    adj = _holm([0.01, float("nan"), 0.02])
    assert math.isnan(adj[1])
    assert adj[0] == 0.02  # 2 * 0.01


def test_holm_caps_at_one():
    assert all(p <= 1.0 for p in _holm([0.6, 0.7, 0.8]))


def test_wilcoxon_clear_shift_is_significant():
    a = np.array([10.0, 11, 12, 13, 14, 15, 16, 17])
    b = a - 3.0  # constant positive paired difference
    p, med, lo, hi, n = _wilcoxon(a, b)
    assert n == 8
    assert med == 3.0
    assert p < 0.05


def test_wilcoxon_all_zero_diff_is_non_rejection_not_untestable():
    """Identical samples give p = 1.0, not NaN (decisions.md 2026-08-28).

    NaN dropped the test out of its Holm family, shrinking m and thereby making
    every SURVIVING test in that family easier to pass. No difference at all is an
    unambiguous NON-rejection, which is exactly what p = 1.0 states.
    """
    a = np.array([1.0, 2, 3])
    p, med, lo, hi, n = _wilcoxon(a, a.copy())
    assert not math.isnan(p) and p == 1.0
    assert med == 0.0 and lo == 0.0 and hi == 0.0
    assert n == 3
