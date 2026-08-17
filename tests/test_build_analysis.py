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
# Confirmatory families - reuses the locked _pairs/_wilcoxon/_holm; check family bookkeeping
# --------------------------------------------------------------------------------------------


def test_confirmatory_family_c2_detects_clear_shift_and_reports_family_size_7():
    rows = _make_fixture_rows()
    df = build_confirmatory_family(rows, "hybrid", [("plain", True, "DQN-plain")], scenario="SCN-05")
    assert len(df) == 7  # 7 KPIs
    assert (df["family_size"] == 7).all()
    avg_wait_row = df[df["kpi_col"] == "avg_waiting_time"].iloc[0]
    assert avg_wait_row["n"] == 15
    assert avg_wait_row["dropped"] == 0
    assert avg_wait_row["median_diff"] == pytest.approx(-1.0)  # hybrid(8) - plain(9)


def test_confirmatory_family_drops_pairs_both_censored():
    rows = _make_fixture_rows()
    # censor the SAME eval seed for both hybrid and plain across all 3 train seeds -> 3 pairs dropped
    for r in rows:
        if r["eval_seed"] == "7000" and r["variant"] in ("hybrid", "plain"):
            r["gridlock_censored"] = "1"
            r["avg_waiting_time"] = ""
    df = build_confirmatory_family(rows, "hybrid", [("plain", True, "DQN-plain")], scenario="SCN-05")
    avg_wait_row = df[df["kpi_col"] == "avg_waiting_time"].iloc[0]
    assert avg_wait_row["n"] == 12
    assert avg_wait_row["dropped"] == 3


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
