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


def test_wilcoxon_all_zero_diff_returns_nan_p():
    a = np.array([1.0, 2, 3])
    p, med, _lo, _hi, n = _wilcoxon(a, a.copy())
    assert math.isnan(p)  # no difference -> undefined, not a spurious significance
    assert med == 0.0
