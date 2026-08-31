"""T-04-02 - Tests for the build_analysis pure functions (table assembly, censoring bookkeeping,
family sizes). Small synthetic fixtures, no SUMO; one smoke test touches the real CSV.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scripts.build_analysis import (
    ALGO_LABELS,
    ROW_ORDER,
    _algo_key,
    _df_to_markdown,
    _md_escape,
    build_confirmatory_family,
    build_family_c1,
    build_family_c2,
    build_family_c3,
    build_honest_findings,
    build_sel_plain_table,
    build_t1_main_results,
    build_t2_ablation_results,
    kpi_stats,
    load_rows,
    write_table,
)


# A distinct, deterministic per-algo offset so EVERY KPI (not just avg_wait/throughput) differs
# between algorithms - a flat constant would give a zero paired diff -> NaN Wilcoxon p (Pratt
# zero-method on an all-zero difference is undefined) and silently shrink the Holm family size.
_ALGO_OFFSET = {"webster": 0.0, "hybrid": 1.0, "plain": 2.0, "random-lstm": 3.0}


def _row(algo, variant, train_seed, scenario, eval_seed, avg_wait, throughput, censored=0, worst=100.0):
    """Build one synthetic eval_results.csv row (dict), matching the real column set."""
    off = _ALGO_OFFSET.get(variant or algo, 0.0)
    return {
        "algo": algo,
        "variant": variant,
        "train_seed": train_seed,
        "scenario": scenario,
        "eval_seed": eval_seed,
        "total_reward": "-100.0",
        "avg_waiting_time": "" if math.isnan(avg_wait) else str(avg_wait),
        "avg_queue_length": str(5.0 + off),
        "throughput": str(throughput),
        "num_stops": str(0.5 + off),
        "wait_p95": str(20.0 + off),
        "fairness_std": str(2.0 + off),
        "worst_movement_max_wait": "" if math.isnan(worst) else str(worst + off),
        "gridlock_censored": str(censored),
    }


def _make_fixture_rows() -> list[dict]:
    """A small SCN-05 fixture: 1 baseline (webster) x 5 eval seeds, 1 DQN variant (hybrid) x 3
    train seeds x 5 eval seeds = n=15 pairable samples against the baseline (repeated per seed)."""
    rows = []
    eval_seeds = ["7000", "7001", "7002", "7003", "7004"]
    for es in eval_seeds:
        rows.append(_row("webster", "", "", "SCN-05", es, avg_wait=10.0, throughput=500.0))
    for ts in ("42", "123", "2024"):
        for es in eval_seeds:
            # hybrid consistently 2s faster than webster -> a clear, detectable paired shift
            rows.append(_row("dqn-hybrid", "hybrid", ts, "SCN-05", es, avg_wait=8.0, throughput=520.0))
            rows.append(_row("dqn-plain", "plain", ts, "SCN-05", es, avg_wait=9.0, throughput=510.0))
    return rows


# --------------------------------------------------------------------------------------------
# kpi_stats - descriptive stats + censoring bookkeeping
# --------------------------------------------------------------------------------------------


def test_kpi_stats_mean_std_and_censoring_count():
    rows = _make_fixture_rows()
    s = kpi_stats(rows, "SCN-05", "hybrid", "avg_waiting_time")
    assert s["n_valid"] == 15
    assert s["n_total"] == 15
    assert s["n_censored"] == 0
    assert s["mean"] == pytest.approx(8.0)
    assert s["std"] == pytest.approx(0.0)


def test_kpi_stats_excludes_nan_but_counts_censored_separately():
    rows = _make_fixture_rows()
    # censor + NaN-out 2 of the hybrid rows' avg_waiting_time (a gridlock episode)
    hybrid_rows = [r for r in rows if r["variant"] == "hybrid"]
    hybrid_rows[0]["avg_waiting_time"] = ""
    hybrid_rows[0]["gridlock_censored"] = "1"
    hybrid_rows[1]["gridlock_censored"] = "1"  # censored but still has a valid KPI value
    s = kpi_stats(rows, "SCN-05", "hybrid", "avg_waiting_time")
    assert s["n_total"] == 15
    assert s["n_censored"] == 2  # both censored rows counted...
    assert s["n_valid"] == 14  # ...but only the NaN one is dropped from the mean


def test_algo_key_uses_variant_for_dqn_rows_and_algo_for_baselines():
    dqn_row = _row("dqn-hybrid", "hybrid", "42", "SCN-05", "7000", 8.0, 500.0)
    base_row = _row("webster", "", "", "SCN-05", "7000", 10.0, 500.0)
    assert _algo_key(dqn_row) == "hybrid"
    assert _algo_key(base_row) == "webster"


# --------------------------------------------------------------------------------------------
# Confirmatory families - reuses the locked _series/_wilcoxon/_holm; check family bookkeeping.
# Amended 2026-09-01 (prereg A1/A2/A3): n is the EVAL-SEED count, and censoring no longer drops
# pairs in the primary analysis.
# --------------------------------------------------------------------------------------------


def test_confirmatory_family_c2_detects_clear_shift_and_reports_family_size_7():
    """Amendment A1.2: n is the number of EVAL SEEDS, not train_seed x eval_seed.

    The fixture has 3 train seeds x 5 eval seeds. The superseded pairing reported n=15 from 5
    independent traffic realizations; the three train seeds are now averaged into each eval
    seed's value and their spread is reported as ``sd_train_seed`` instead of being counted.
    """
    rows = _make_fixture_rows()
    df = build_confirmatory_family(rows, "hybrid", [("plain", True, "DQN-plain")], scenario="SCN-05")
    assert len(df) == 7  # 7 KPIs
    assert (df["family_size"] == 7).all()
    avg_wait_row = df[df["kpi_col"] == "avg_waiting_time"].iloc[0]
    assert avg_wait_row["n"] == 5, "n counts eval seeds (A1.2), not train x eval pairs"
    assert avg_wait_row["dropped"] == 0
    assert avg_wait_row["median_diff"] == pytest.approx(-1.0)  # hybrid(8) - plain(9)
    assert avg_wait_row["effect_source"] == "primary"  # nothing censored -> nothing imputed
    # A3: with m=7 a test needs n>=9 to be capable of rejecting, so this 5-seed fixture is
    # correctly flagged undecidable rather than being reported as a non-rejection.
    assert avg_wait_row["min_testable_n"] == 9
    assert bool(avg_wait_row["undecidable"]) is True


def test_confirmatory_family_scores_a_mutual_gridlock_as_a_tie_not_a_drop():
    """Amendment A2.2, replacing the superseded both-censored DROP.

    Both arms failing on the same traffic is a tie, and the primary analysis keeps it as one:
    both take the worst rank, the paired difference is exactly 0, and Pratt handles the zero.
    The complete-case sensitivity analysis still drops it, and both numbers are carried in the
    same frame so neither can be picked after the fact.
    """
    rows = _make_fixture_rows()
    for r in rows:
        if r["eval_seed"] == "7000" and r["variant"] in ("hybrid", "plain"):
            r["gridlock_censored"] = "1"
            r["avg_waiting_time"] = ""
    df = build_confirmatory_family(rows, "hybrid", [("plain", True, "DQN-plain")], scenario="SCN-05")
    avg_wait_row = df[df["kpi_col"] == "avg_waiting_time"].iloc[0]
    assert avg_wait_row["n"] == 5, "primary drops NO pair for censoring"
    assert avg_wait_row["censored_a"] == 1 and avg_wait_row["censored_b"] == 1
    assert avg_wait_row["imputed"] == 2, "both arms take the worst rank"
    assert avg_wait_row["n_cc"] == 4 and avg_wait_row["dropped_cc"] == 1, "complete-case drops it"
    # A2.4: once anything is imputed the effect size comes from the measured episodes only.
    assert avg_wait_row["effect_source"] == "complete-case"


def test_a_lone_censored_comparator_no_longer_hands_the_treatment_a_free_win():
    """Amendment A2.1 - the selection bias, pinned.

    The superseded rule dropped a pair only when BOTH arms were censored, so a pair in which
    only the COMPARATOR gridlocked survived at face value: a win earned by not failing, counted
    as a wait-time win. Now the failed arm is ranked worst (primary) and the pair is dropped
    outright by the sensitivity analysis.
    """
    rows = _make_fixture_rows()
    for r in rows:
        if r["eval_seed"] == "7000" and r["variant"] == "plain":
            r["gridlock_censored"] = "1"
            r["avg_waiting_time"] = ""
    df = build_confirmatory_family(rows, "hybrid", [("plain", True, "DQN-plain")], scenario="SCN-05")
    avg_wait_row = df[df["kpi_col"] == "avg_waiting_time"].iloc[0]
    assert avg_wait_row["censored_a"] == 0 and avg_wait_row["censored_b"] == 1
    assert avg_wait_row["n"] == 5 and avg_wait_row["imputed"] == 1
    assert avg_wait_row["n_cc"] == 4 and avg_wait_row["dropped_cc"] == 1
    assert avg_wait_row["dropped"] == 0, "the primary analysis drops nothing for censoring"


def test_holm_family_c1_c2_c3_shapes_on_real_csv():
    """Smoke test against the real, committed eval_results.csv - the only place this suite
    touches the filesystem beyond the synthetic fixtures."""
    rows = load_rows()
    assert len(rows) == 300
    c1 = build_family_c1(rows)
    c2 = build_family_c2(rows)
    c3 = build_family_c3(rows)
    assert len(c1) == 21  # 3 baselines x 7 KPIs
    assert len(c2) == 7
    assert len(c3) == 7
    # every realized p_holm is a valid probability (or NaN if the family excluded it)
    for df in (c1, c2, c3):
        finite = df["p_holm"].dropna()
        assert ((finite >= 0.0) & (finite <= 1.0)).all()


# --------------------------------------------------------------------------------------------
# T1 / T2 table shape
# --------------------------------------------------------------------------------------------


def test_t1_has_one_table_per_scenario_and_all_row_order_algorithms():
    rows = _make_fixture_rows()
    tables = build_t1_main_results(rows)
    assert set(tables.keys()) == {"SCN-05"}
    df = tables["SCN-05"]
    assert list(df["algorithm"]) == [ALGO_LABELS[a] for a in ROW_ORDER]
    assert "n_censored/n_total" in df.columns


def test_t2_rows_restricted_to_dqn_variants_only():
    rows = _make_fixture_rows()
    tables = build_t2_ablation_results(rows)
    df = tables["SCN-05"]
    assert list(df["algorithm"]) == ["DQN-hybrid", "DQN-plain", "DQN-random-lstm"]


def test_honest_findings_scn01_hybrid_worse_than_plain_on_real_csv():
    """Regression-locks the pre-registered honest finding: hybrid is WORSE than plain on SCN-01
    avg wait among non-censored episodes (~2.71s vs ~1.36s) - this must never get quietly fixed
    by a future refactor of the censoring filter."""
    rows = load_rows()
    honest = build_honest_findings(rows)
    hyb = honest[(honest["scenario"] == "SCN-01") & (honest["algorithm"] == "DQN-hybrid")
                 & (honest["kpi"] == "avg_waiting_time")].iloc[0]
    pln = honest[(honest["scenario"] == "SCN-01") & (honest["algorithm"] == "DQN-plain")
                 & (honest["kpi"] == "avg_waiting_time")].iloc[0]
    assert hyb["mean_noncensored"] == pytest.approx(2.71, abs=0.02)
    assert pln["mean_noncensored"] == pytest.approx(1.36, abs=0.02)
    assert hyb["mean_noncensored"] > pln["mean_noncensored"]


def test_honest_findings_fully_excludes_censored_rows_even_when_kpi_is_non_nan():
    rows = _make_fixture_rows()
    # SCN-04 IS one of build_honest_findings' hardcoded scenarios; add a censored-but-valid row
    rows.append(_row("dqn-hybrid", "hybrid", "42", "SCN-04", "9000", avg_wait=100.0, throughput=1.0,
                      censored=1))  # valid (non-NaN) avg_waiting_time, but censored
    honest = build_honest_findings(rows)
    hyb = honest[(honest["scenario"] == "SCN-04") & (honest["algorithm"] == "DQN-hybrid")
                 & (honest["kpi"] == "avg_waiting_time")].iloc[0]
    # the extreme censored value (100.0) must be excluded, not just NaN-filtered
    assert hyb["n_noncensored"] == 0
    assert math.isnan(hyb["mean_noncensored"])


# --------------------------------------------------------------------------------------------
# sel/plain log parsing
# --------------------------------------------------------------------------------------------


def test_sel_plain_log_parsing_includes_every_scenario_and_nan(tmp_path):
    log_text = """Using libsumo as traci as requested by environment variable.

=== SCN-04 ===   (wait s | throughput | %gridlock)
  webster         wait= 11.03 | thru= 1412.8 | grid=  20%
  plain-s42       wait=  1.86 | thru= 1526.4 | grid=  20%

=== SCN-06 ===   (wait s | throughput | %gridlock)
  webster         wait=  1.46 | thru=  956.6 | grid=   0%

=== SCN-05 ===   (wait s | throughput | %gridlock)
  plain-s123      wait=   nan | thru=  429.0 | grid= 100%
"""
    log_path = tmp_path / "compare_sel_plain.log"
    log_path.write_text(log_text, encoding="utf-8")
    df = build_sel_plain_table(log_path)
    assert set(df["scenario"]) == {"SCN-04", "SCN-06", "SCN-05"}
    assert len(df) == 4
    nan_row = df[(df["scenario"] == "SCN-05") & (df["condition"] == "plain-s123")].iloc[0]
    assert math.isnan(nan_row["avg_wait_s"])
    assert nan_row["pct_gridlock"] == 100.0


# --------------------------------------------------------------------------------------------
# markdown rendering - no stray unmatched `**`/`|` that would corrupt table structure
# --------------------------------------------------------------------------------------------


def test_md_escape_neutralizes_bold_and_pipe_chars():
    assert _md_escape("p<0.01**") == "p<0.01\\*\\*"
    assert _md_escape("a|b") == "a\\|b"


def test_df_to_markdown_escapes_asterisks_in_cells():
    df = pd.DataFrame({"a": ["**", "plain"], "b": [1, 2]})
    md = _df_to_markdown(df)
    assert "| \\*\\* | 1 |" in md
    # no bare, unescaped double-asterisk survives into the rendered table
    assert "| ** |" not in md


def test_write_table_writes_both_md_and_csv_for_a_dict_of_scenarios(tmp_path):
    df = pd.DataFrame({"algorithm": ["Webster"], "avg_wait": ["1.00 ± 0.10"]})
    write_table({"SCN-05": df}, "unit_test_table", tmp_path, "title")
    assert (tmp_path / "unit_test_table.md").exists()
    assert (tmp_path / "unit_test_table.csv").exists()
    csv_df = pd.read_csv(tmp_path / "unit_test_table.csv")
    assert "scenario" in csv_df.columns
    assert csv_df.iloc[0]["scenario"] == "SCN-05"
