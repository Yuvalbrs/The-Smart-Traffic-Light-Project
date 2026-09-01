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


def _full_kpi_rows(n_eval_seeds, *, blank_kpi=None, blank_after=None):
    """Fixture carrying ALL 7 locked KPIs, so a single KPI can be made undecidable in isolation."""
    from scripts.analyze_eval import KPIS

    rows = []
    for i in range(n_eval_seeds):
        es = str(7000 + i)
        for ts in _TRAIN_SEEDS:
            r = {"variant": "hybrid", "train_seed": ts, "scenario": "SCN-05", "eval_seed": es,
                 "algo": "hybrid", "gridlock_censored": "0"}
            for col, _l, _d in KPIS:
                r[col] = "" if (col == blank_kpi and i >= blank_after) else str(10.0 + i)
            rows.append(r)
        b = {"variant": "", "train_seed": "", "scenario": "SCN-05", "eval_seed": es,
             "algo": "webster", "gridlock_censored": "0"}
        for col, _l, _d in KPIS:
            b[col] = str(20.0 + i)
        rows.append(b)
    return rows


def _holm_for(rows, kpi_label):
    """Run the real C2-shaped family and return (p_holm, n) for one KPI."""
    from scripts.analyze_eval import _family

    dqn, base = _index(rows)
    seeds = sorted({r["eval_seed"] for r in rows}, key=int)
    out = _family(dqn, base, seeds, "SCN-05", "hybrid",
                  [("webster", False, "webster")], "T", [])
    row = next(r for r in out if r["klabel"] == kpi_label)
    return row["p_holm"], row["n"], row["undecidable"]


def test_a4_an_undecidable_test_does_not_make_its_siblings_easier_to_pass():
    """A3 shrank m by the undecidable count; A4 retires that, and this checks the BEHAVIOUR.

    n is a function of the DATA (censoring, missing cells), so an m that follows it is exactly the
    data-dependent m ``_holm``'s docstring warns against: the correction gets weaker precisely when
    the evidence has already been degraded. The observable consequence is what is asserted here -
    knocking one KPI down below its decidability floor must leave every OTHER KPI's Holm-adjusted
    p-value untouched.
    """
    n_seeds = 12  # comfortably above the m=7 floor of 9
    baseline = _full_kpi_rows(n_seeds)
    # make avg_queue undecidable ALONE by dropping its value on the last 5 eval seeds (n 12 -> 7)
    degraded = _full_kpi_rows(n_seeds, blank_kpi="avg_queue_length", blank_after=7)

    p_before, n_before, und_before = _holm_for(baseline, "avg_wait")
    p_after, n_after, und_after = _holm_for(degraded, "avg_wait")
    _p, n_q, und_q = _holm_for(degraded, "avg_queue")

    assert und_q is True and n_q == 7, "the degraded KPI must actually fall below the floor"
    assert und_before is False and und_after is False, "avg_wait must stay decidable in both"
    assert n_before == n_after == 12, "avg_wait's own n must be unaffected"
    assert p_after == p_before, (
        "a sibling becoming UNDECIDABLE must NOT change this test's Holm-adjusted p - "
        f"m shrank ({p_before} -> {p_after})")


def test_a4_holm_uses_the_preregistered_family_size_not_the_testable_count():
    """The pinned m is observable: the smallest raw p is multiplied by the FULL family size."""
    from scripts.analyze_eval import _family

    rows = _full_kpi_rows(12)
    dqn, base = _index(rows)
    seeds = sorted({r["eval_seed"] for r in rows}, key=int)
    out = _family(dqn, base, seeds, "SCN-05", "hybrid", [("webster", False, "webster")], "T", [])
    decidable = [r for r in out if not r["undecidable"] and not math.isnan(r["p"])]
    assert decidable, "fixture must produce decidable tests"
    smallest = min(decidable, key=lambda r: r["p"])
    assert smallest["p_holm"] == pytest.approx(min(1.0, 7 * smallest["p"])), (
        "the most significant test must be corrected by the pre-registered m=7")


# ---------------------------------------------------------------------------
# The three critical defects found by adversarial review of A1-A4 (2026-09-01).
# ---------------------------------------------------------------------------


def test_undecidability_uses_effective_n_not_the_raw_pair_count():
    """DEFECT 2: A4.2's mutual-failure ties switched off A3's own guard.

    A3 compared the RAW pair count against the attainable-n floor. A4.2 scores a both-censored
    pair as an exact zero difference, and Pratt zeros carry no signed-rank evidence - they raise
    the attainable minimum p well above 2/2**n. So a family could hold 9 mutual failures plus 6
    unanimous observed pairs, report n=15, be incapable of rejecting under any data, and still
    consume a Holm slot that taxes its 20 siblings.
    """
    from scripts.analyze_eval import _effective_n, min_testable_n

    n_seeds = 15
    mutual = {str(7000 + i) for i in range(9)}  # 9 pairs where BOTH arms gridlock
    rows = _fixture(n_eval_seeds=n_seeds, base_censor=mutual,
                    dqn_censor={(es, ts) for es in mutual for ts in _TRAIN_SEEDS})
    dqn, base = _index(rows)
    a, b, meta = _series(dqn, base, _seeds(n_seeds), "SCN-05", "avg_waiting_time", "lower",
                         "hybrid", "webster", False, "primary")
    assert len(a) == 15, "A2 still drops no pair for censoring"
    assert _effective_n(a, b) == 6, "only the 6 fully-observed pairs carry evidence"
    assert _effective_n(a, b) < min_testable_n(21), (
        "6 informative pairs cannot clear the C1 Holm floor -> the test must be UNDECIDABLE, "
        "which the raw count of 15 would have hidden")


def test_fully_censored_comparison_is_flagged_not_published_as_parity():
    """DEFECT 4: total data loss was published as measured parity with a zero-width CI.

    With every pair censored, `_series` returned all-zero arrays, `_wilcoxon` hit its
    `allclose(d, 0)` branch and returned (p=1.0, median=0.0, CI=[0,0]) - which the supporting
    table printed as "no effect detected", dropped=0. The allclose reasoning is right for
    MEASURED ties and wrong for IMPUTED ones, and `_wilcoxon` cannot tell them apart because the
    provenance was thrown away before it was called. `meta` now carries it.
    """
    n = 5
    every = {str(7000 + i) for i in range(n)}
    rows = _fixture(n_eval_seeds=n, base_censor=every,
                    dqn_censor={(es, ts) for es in every for ts in _TRAIN_SEEDS})
    dqn, base = _index(rows)
    a, b, meta = _series(dqn, base, _seeds(n), "SCN-05", "avg_waiting_time", "lower",
                         "hybrid", "webster", False, "primary")
    assert meta["fully_imputed"] is True and meta["n_observed"] == 0
    assert (a - b).tolist() == [0.0] * n, "mutual failure is still scored as a tie (A2.2)"
    # the complete-case sensitivity analysis has literally nothing left
    _ac, _bc, meta_c = _series(dqn, base, _seeds(n), "SCN-05", "avg_waiting_time", "lower",
                               "hybrid", "webster", False, "complete")
    assert meta_c["dropped_cens"] == n


def test_a_partially_censored_comparison_still_carries_its_own_kpi_scale():
    """DEFECT 3: the imputed magnitude fell back to an ABSOLUTE 1.0 with no observed pair.

    `_worst_rank_delta` scaled off fully-observed DIFFERENCES only, so with none it returned the
    constant 1.0 - applied identically to num_stops (~0.04) and throughput (~1000). Every imputed
    difference became +-1.0, the KPI stopped contributing, and all seven KPIs returned the SAME
    p-value: a sign test on the censoring pattern, printed as seven per-KPI rows.

    The magnitude must track the KPI's own scale. (When NO pair is observed the comparison carries
    no KPI information at all and identical p-values are correct - it is then flagged
    `fully_imputed`; see the test above.)
    """
    from scripts.analyze_eval import _worst_rank_delta

    small = [{"va": 0.040, "vb": 0.041, "ca": True, "cb": False},
             {"va": 0.043, "vb": 0.044, "ca": True, "cb": False}]
    large = [{"va": 900.0, "vb": 910.0, "ca": True, "cb": False},
             {"va": 930.0, "vb": 940.0, "ca": True, "cb": False}]
    d_small, d_large = _worst_rank_delta(small), _worst_rank_delta(large)
    assert d_small != 1.0 and d_large != 1.0, "the absolute-1.0 fallback must be gone"
    assert d_large > d_small * 100, (
        f"the imputed magnitude must track the KPI scale, got {d_small} vs {d_large}")


def test_train_seed_provenance_is_carried_so_a_lone_seed_cannot_read_as_robust():
    """DEFECT 8: a single surviving train seed reported sd_ts = 0.00, i.e. perfect robustness.

    `_arm_value` treats a train seed as present if ANY row exists, so deleting two of three rows
    yielded (value, censored=False, sd=0.0, present=True) - rendered in the results table as
    `0.00`. `meta` now reports how many train-seed values actually entered the mean.
    """
    rows = [r for r in _fixture(n_eval_seeds=3)
            if not (r["variant"] == "hybrid" and r["eval_seed"] == "7001"
                    and r["train_seed"] in ("123", "2024"))]
    dqn, base = _index(rows)
    _a, _b, meta = _series(dqn, base, _seeds(3), "SCN-05", "avg_waiting_time", "lower",
                           "hybrid", "webster", False, "primary")
    assert "n_train_seed_values" in meta
