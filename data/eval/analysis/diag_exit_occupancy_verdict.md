# Phase-0 diagnostic — exit-edge spillback vs insertion backlog (SCN-05, plain-DQN ep299)

**VERDICT: DEAD**

100% of gridlocked episodes never reach mean exit occupancy 0.70 (median sustained 0.023, worst-edge median 0.046) while insertion backlog grows in 100% of them. The exits are NOT full: an exit-occupancy mask would never bind -> non-binding constraint = no-op (scar, struck x2). Candidate DEAD; sel/plain ships unchanged and this measurement becomes report evidence that gridlock is intrinsic.

- Episodes: 15 (14 gridlocked at backlog > 0.1).
- Pre-registered thresholds (fixed in code before the first run, finish-plan.md §Phase 0): dead < 0.7, attempt >= 0.85, sustained over 60 s; onset = 5+ vehicles pending for 60 s.
- Raw per-step data: `data\eval\analysis\diag_exit_occupancy.csv` · plot: `data\eval\analysis\P6_exit_occupancy_vs_backlog.png`

| train_seed | eval_seed | gridlock | backlog_frac | onset_step | exit_sustained_pre_onset | exit_worst_edge_sustained | exit_peak | approach_peak | pending_peak |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 7000 | YES | 0.732 | 60 | 0.022 | 0.041 | 0.061 | 0.658 | 1157 |
| 42 | 7001 | YES | 0.739 | 70 | 0.022 | 0.039 | 0.061 | 0.655 | 1186 |
| 42 | 7002 | no | -0.001 | None | 0.028 | 0.058 | 0.073 | 0.087 | 1 |
| 42 | 7003 | YES | 0.636 | 95 | 0.024 | 0.043 | 0.061 | 0.643 | 1014 |
| 42 | 7004 | YES | 0.642 | 89 | 0.023 | 0.047 | 0.073 | 0.658 | 1020 |
| 123 | 7000 | YES | 0.410 | 73 | 0.022 | 0.047 | 0.073 | 0.559 | 647 |
| 123 | 7001 | YES | 0.765 | 62 | 0.023 | 0.045 | 0.061 | 0.656 | 1228 |
| 123 | 7002 | YES | 0.594 | 105 | 0.024 | 0.047 | 0.073 | 0.659 | 919 |
| 123 | 7003 | YES | 0.511 | 139 | 0.027 | 0.054 | 0.098 | 0.651 | 814 |
| 123 | 7004 | YES | 0.686 | 79 | 0.024 | 0.051 | 0.086 | 0.660 | 1091 |
| 2024 | 7000 | YES | 0.736 | 60 | 0.022 | 0.041 | 0.061 | 0.649 | 1163 |
| 2024 | 7001 | YES | 0.602 | 88 | 0.029 | 0.055 | 0.061 | 0.652 | 966 |
| 2024 | 7002 | YES | 0.559 | 120 | 0.024 | 0.049 | 0.068 | 0.656 | 864 |
| 2024 | 7003 | YES | 0.432 | 134 | 0.025 | 0.045 | 0.061 | 0.650 | 687 |
| 2024 | 7004 | YES | 0.729 | 72 | 0.022 | 0.043 | 0.073 | 0.661 | 1160 |
