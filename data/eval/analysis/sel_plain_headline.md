# sel/plain: EpisodeLevelSelector(plain-DQN, Webster) vs plain DQN, per scenario x seed

Parsed verbatim from runs/compare_sel_plain.log (scripts/compare_selector_plain.py). Every row present in the log is included, SCN-06 included.

| scenario | condition | avg_wait_s | throughput | pct_gridlock |
|---|---|---|---|---|
| SCN-04 | webster | 11.03 | 1412.8 | 20.0 |
| SCN-04 | plain-s42 | 1.86 | 1526.4 | 20.0 |
| SCN-04 | sel/plain-s42 | 1.93 | 1178.2 | 40.0 |
| SCN-04 | plain-s123 | 2.02 | 1442.6 | 20.0 |
| SCN-04 | sel/plain-s123 | 1.9 | 1514.2 | 20.0 |
| SCN-04 | plain-s2024 | 2.32 | 1230.2 | 40.0 |
| SCN-04 | sel/plain-s2024 | 2.27 | 1401.4 | 40.0 |
| SCN-06 | webster | 1.46 | 956.6 | 0.0 |
| SCN-06 | plain-s42 | 1.57 | 972.0 | 0.0 |
| SCN-06 | sel/plain-s42 | 1.53 | 971.6 | 0.0 |
| SCN-06 | plain-s123 | 1.46 | 971.0 | 0.0 |
| SCN-06 | sel/plain-s123 | 1.49 | 971.0 | 0.0 |
| SCN-06 | plain-s2024 | 1.69 | 971.2 | 0.0 |
| SCN-06 | sel/plain-s2024 | 1.73 | 847.8 | 20.0 |
| SCN-05 | webster | 3.83 | 1221.4 | 20.0 |
| SCN-05 | plain-s42 | 3.06 | 529.0 | 80.0 |
| SCN-05 | sel/plain-s42 | 4.03 | 1027.0 | 40.0 |
| SCN-05 | plain-s123 | nan | 429.0 | 100.0 |
| SCN-05 | sel/plain-s123 | 4.03 | 1070.2 | 40.0 |
| SCN-05 | plain-s2024 | nan | 392.0 | 100.0 |
| SCN-05 | sel/plain-s2024 | 4.02 | 1253.0 | 20.0 |

