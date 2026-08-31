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


# ---------------------------------------------------------------------------
# Preregistration amendments A1/A2/A3 (2026-09-01). These tests pin the two
# statistical biases that were live before the amendment, so neither can come
# back silently.
# ---------------------------------------------------------------------------

import pytest

from scripts.analyze_eval import _arm_value, _index, _series, min_testable_n

_TRAIN_SEEDS = ("42", "123", "2024")


def _dqn_row(variant, ts, es, wait, censored=0):
    return {"variant": variant, "train_seed": ts, "scenario": "SCN-05", "eval_seed": es,
            "algo": variant, "gridlock_censored": str(censored), "avg_waiting_time": str(wait)}


def _base_row(algo, es, wait, censored=0):
    return {"variant": "", "train_seed": "", "scenario": "SCN-05", "eval_seed": es,
            "algo": algo, "gridlock_censored": str(censored), "avg_waiting_time": str(wait)}


def _fixture(n_eval_seeds=5, dqn_wait=10.0, base_wait=20.0, dqn_censor=(), base_censor=()):
    """One hybrid variant (3 train seeds) + one baseline, over n eval seeds."""
    rows = []
    for i in range(n_eval_seeds):
        es = str(7000 + i)
        for ts in _TRAIN_SEEDS:
            rows.append(_dqn_row("hybrid", ts, es, dqn_wait + i, 1 if (es, ts) in dqn_censor else 0))
        rows.append(_base_row("webster", es, base_wait + i, 1 if es in base_censor else 0))
    return rows


def _seeds(n):
    return [str(7000 + i) for i in range(n)]


# --- A1.3 / A3: the arithmetic floor on n -----------------------------------

def test_min_testable_n_matches_the_holm_floors_in_the_amendment():
    """A1.3's table. These three numbers are why eval seeds went 5 -> 15."""
    assert min_testable_n(21) == 10   # C1 / H1 family
    assert min_testable_n(7) == 9     # C2 / H2 and C3 / H3 families
    assert min_testable_n(1) == 6     # uncorrected alpha=0.05


def test_n_five_cannot_reject_which_is_the_whole_reason_for_the_amendment():
    """The floor is above 5 even with NO multiplicity correction at all.

    Aggregating train seeds (A1.2) over the frozen 5 eval seeds gives n=5, and the two-sided
    signed-rank test's minimum attainable p is 2/2**5 = 0.0625 > 0.05. The design would have
    been incapable of rejecting anything, for any data. Hence 7000-7014.
    """
    assert min_testable_n(1) > 5
    assert 2 / 2 ** 5 > 0.05
    assert 2 / 2 ** 15 < 0.05 / 21


# --- A1.2: the pseudo-replication fix ---------------------------------------

def test_baseline_pairing_no_longer_repeats_the_same_number_three_times():
    """THE BIAS THIS AMENDMENT EXISTS FOR (A1.1).

    The superseded rule paired over (train_seed x eval_seed), so with 5 eval seeds it produced
    15 'independent' differences from only 5 independent baseline observations. n is now the
    number of eval seeds, full stop.
    """
    rows = _fixture(n_eval_seeds=5)
    dqn, base = _index(rows)
    a, b, meta = _series(dqn, base, _seeds(5), "SCN-05", "avg_waiting_time", "lower",
                         "hybrid", "webster", False, "primary")
    assert len(a) == len(b) == 5, "n must be the eval-seed count, not train x eval"
    assert meta["dropped"] == 0
    # each baseline value appears exactly once
    assert len(set(b.tolist())) == 5


def test_dqn_value_is_the_mean_across_train_seeds_and_dispersion_is_reported():
    rows = [_dqn_row("hybrid", "42", "7000", 10.0),
            _dqn_row("hybrid", "123", "7000", 20.0),
            _dqn_row("hybrid", "2024", "7000", 30.0),
            _base_row("webster", "7000", 99.0)]
    dqn, base = _index(rows)
    val, censored, sd, present = _arm_value(dqn, base, "hybrid", True, "SCN-05", "7000", "avg_waiting_time")
    assert present and not censored
    assert val == pytest.approx(20.0)          # the mean, not three samples
    assert sd == pytest.approx(10.0)            # ddof=1 sd, reported not spent


def test_dqn_vs_dqn_also_uses_the_eval_seed_as_the_unit():
    """A1.2 applies the SAME rule to every family, so no comparison is advantaged by pairing."""
    rows = _fixture(n_eval_seeds=5)
    for i in range(5):
        for ts in _TRAIN_SEEDS:
            rows.append(_dqn_row("plain", ts, str(7000 + i), 15.0 + i))
    dqn, base = _index(rows)
    a, b, _m = _series(dqn, base, _seeds(5), "SCN-05", "avg_waiting_time", "lower",
                       "hybrid", "plain", True, "primary")
    assert len(a) == len(b) == 5


# --- A2: censoring is informative, not missing ------------------------------

def test_pair_where_only_the_comparator_gridlocked_is_no_longer_a_free_win():
    """THE SECOND BIAS (A2.1).

    Under the superseded rule a pair was dropped ONLY if BOTH arms were censored, so a pair in
    which only the BASELINE gridlocked was kept at its face value -- a win the treatment earned
    by not failing, laundered into a wait-time win. Now the censored arm takes the worst rank
    (primary), and the complete-case sensitivity analysis drops the pair entirely.
    """
    rows = _fixture(n_eval_seeds=5, base_censor={"7002"})
    dqn, base = _index(rows)

    a_p, b_p, m_p = _series(dqn, base, _seeds(5), "SCN-05", "avg_waiting_time", "lower",
                            "hybrid", "webster", False, "primary")
    assert len(a_p) == 5, "primary drops no pair for censoring"
    assert m_p["imputed"] == 1 and m_p["cens_b"] == 1
    # A4: the guarantee is on the DIFFERENCE, because that is what the signed-rank test ranks.
    d = a_p - b_p
    assert abs(d[2]) > max(abs(x) for i, x in enumerate(d) if i != 2), "imputed pair must rank top"
    assert d[2] < 0, "the baseline failed, so the difference must favour hybrid"

    a_c, b_c, m_c = _series(dqn, base, _seeds(5), "SCN-05", "avg_waiting_time", "lower",
                            "hybrid", "webster", False, "complete")
    assert len(a_c) == 4 and m_c["dropped_cens"] == 1, "complete-case drops it"


def test_both_censored_is_a_tie_not_a_drop():
    """A2.2: mutual failure scores as the tie it is; Pratt then handles the zero difference."""
    rows = _fixture(n_eval_seeds=5, base_censor={"7003"},
                    dqn_censor={("7003", "42"), ("7003", "123"), ("7003", "2024")})
    dqn, base = _index(rows)
    a, b, meta = _series(dqn, base, _seeds(5), "SCN-05", "avg_waiting_time", "lower",
                         "hybrid", "webster", False, "primary")
    assert len(a) == 5
    assert a[3] == b[3], "both censored -> both worst rank -> paired difference exactly 0"
    assert meta["imputed"] == 2


def test_worst_rank_respects_the_better_direction():
    """Throughput is higher-is-better, so its sentinel must sit BELOW every observed value."""
    rows = _fixture(n_eval_seeds=5, base_censor={"7001"})
    for r in rows:
        r["throughput"] = str(2000.0 + float(r["avg_waiting_time"]))
    dqn, base = _index(rows)
    a, b, _m = _series(dqn, base, _seeds(5), "SCN-05", "throughput", "higher",
                       "hybrid", "webster", False, "primary")
    d = a - b
    assert abs(d[1]) > max(abs(x) for i, x in enumerate(d) if i != 1), "imputed pair must rank top"
    assert d[1] > 0, "higher-is-better: a failed baseline must make hybrid-minus-baseline POSITIVE"


def test_a_variant_censored_on_one_train_seed_is_censored_for_that_episode():
    """A2.4: averaging a gridlock away behind two clean seeds would hide the failure."""
    rows = _fixture(n_eval_seeds=5, dqn_censor={("7004", "123")})
    dqn, base = _index(rows)
    _v, censored, _sd, present = _arm_value(dqn, base, "hybrid", True, "SCN-05", "7004", "avg_waiting_time")
    assert present and censored, "ANY censored train seed censors the aggregate"


def test_missing_row_is_a_drop_not_a_censoring():
    """s8's missing-pair-member rule is unchanged and stays separate from censoring."""
    rows = [r for r in _fixture(n_eval_seeds=5)
            if not (r["variant"] == "hybrid" and r["eval_seed"] == "7001")]
    dqn, base = _index(rows)
    a, _b, meta = _series(dqn, base, _seeds(5), "SCN-05", "avg_waiting_time", "lower",
                          "hybrid", "webster", False, "primary")
    assert len(a) == 4
    assert meta["missing"] == 1 and meta["dropped_cens"] == 0


# --- A4 (2026-09-01): the correction to A2.2's imputation scale --------------


@pytest.mark.parametrize("direction,fail_arm,expect_positive", [
    ("lower", "b", False),   # lower-is-better, comparator failed -> hybrid-minus-other negative
    ("lower", "a", True),    # lower-is-better, treatment failed  -> positive
    ("higher", "b", True),   # higher-is-better, comparator failed -> positive
    ("higher", "a", False),  # higher-is-better, treatment failed  -> negative
])
def test_a4_imputed_pair_always_takes_the_top_signed_rank(direction, fail_arm, expect_positive):
    """A2.2 promised the imputed pair would take the largest rank; A4 makes that TRUE.

    The original rule imputed an extreme VALUE. The Wilcoxon signed-rank test ranks |d_i|, not
    values, so when the surviving arm sat near the imputed extreme the pair could rank LAST-but-one
    or even first - measured at rank 1 of 5 on a worked example, meaning an episode where the
    comparator gridlocked contributed the LEAST evidence. Imputing on the difference scale makes
    the guarantee structural instead of hoped-for.
    """
    rows = []
    pairs = [(10.0, 20.0), (19.0, 19.5), (11.0, 20.0), (12.0, 20.0), (13.0, 20.0)]
    if direction == "higher":
        pairs = [(b, a) for a, b in pairs]
    hit = 1  # the near-tie pair: the case the value-scale imputation got wrong
    for i, (av, bv) in enumerate(pairs):
        es = str(7000 + i)
        for ts in _TRAIN_SEEDS:
            rows.append(_dqn_row("hybrid", ts, es, av, 1 if (i == hit and fail_arm == "a") else 0))
        rows.append(_base_row("webster", es, bv, 1 if (i == hit and fail_arm == "b") else 0))
    dqn, base = _index(rows)
    a, b, meta = _series(dqn, base, _seeds(5), "SCN-05", "avg_waiting_time", direction,
                         "hybrid", "webster", False, "primary")
    d = a - b
    assert meta["imputed"] == 1 and len(d) == 5
    assert abs(d[hit]) > max(abs(x) for i, x in enumerate(d) if i != hit), (
        "the censored pair must carry the LARGEST |difference|, i.e. the top signed rank")
    assert bool(d[hit] > 0) is expect_positive, "the imputed sign must point away from the arm that failed"


def test_a4_mutual_failure_is_still_an_exact_tie():
    rows = _fixture(n_eval_seeds=5, base_censor={"7003"},
                    dqn_censor={("7003", ts) for ts in _TRAIN_SEEDS})
    dqn, base = _index(rows)
    a, b, meta = _series(dqn, base, _seeds(5), "SCN-05", "avg_waiting_time", "lower",
                         "hybrid", "webster", False, "primary")
    assert (a - b)[3] == 0.0 and meta["imputed"] == 2


def test_a4_holm_family_size_is_not_shrunk_by_undecidable_tests():
    """A3 reduced m by the undecidable count; A4 retires that.

    n depends on the data (censoring, missingness), so an m that follows it is the data-dependent
    m that ``_holm``'s own docstring warns against - it weakens the correction exactly when the
    evidence is already degraded. Pinning m is the conservative error, which is the safe one.
    """
    import inspect

    from scripts.analyze_eval import _family

    src = inspect.getsource(_family)
    assert "m_used = prereg_m" in src, "m must stay pinned at the pre-registered family size"
    assert "max(1, prereg_m - n_und)" not in src, "the A3 m-reduction must be gone"
