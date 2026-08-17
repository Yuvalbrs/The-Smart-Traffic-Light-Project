# Honest findings, gridlock-censored episodes FULLY excluded (stricter than T1/T2)

SCN-01: DQN-hybrid is worse than DQN-plain on avg wait among non-censored episodes. SCN-04: the DQN-beats-Webster story. Both are reported here, not softened.

| scenario | algorithm | kpi | n_noncensored | mean_noncensored | std_noncensored |
|---|---|---|---|---|---|
| SCN-01 | DQN-hybrid | avg_waiting_time | 13 | 2.7121247975922427 | 1.019709340615767 |
| SCN-01 | DQN-hybrid | throughput | 13 | 781.0 | 44.81815108487483 |
| SCN-01 | DQN-hybrid | worst_movement_max_wait | 13 | 105.46153846153847 | 71.83153367963982 |
| SCN-01 | DQN-plain | avg_waiting_time | 12 | 1.3567181822541625 | 0.15461519598172127 |
| SCN-01 | DQN-plain | throughput | 12 | 791.6666666666666 | 40.207039938374734 |
| SCN-01 | DQN-plain | worst_movement_max_wait | 12 | 37.166666666666664 | 14.011899704655804 |
| SCN-01 | Webster | avg_waiting_time | 4 | 1.1914937372747243 | 0.15120573602942308 |
| SCN-01 | Webster | throughput | 4 | 799.25 | 48.6372628067548 |
| SCN-01 | Webster | worst_movement_max_wait | 4 | 25.25 | 5.188127472091127 |
| SCN-04 | DQN-hybrid | avg_waiting_time | 10 | 2.1001614594933886 | 0.3052061862061863 |
| SCN-04 | DQN-hybrid | throughput | 10 | 1627.3 | 72.81796176463301 |
| SCN-04 | DQN-hybrid | worst_movement_max_wait | 10 | 74.7 | 20.90215299915298 |
| SCN-04 | DQN-plain | avg_waiting_time | 11 | 2.042482839629662 | 0.23866318744913215 |
| SCN-04 | DQN-plain | throughput | 11 | 1676.1818181818182 | 56.591197516607096 |
| SCN-04 | DQN-plain | worst_movement_max_wait | 11 | 49.36363636363637 | 8.453079051715148 |
| SCN-04 | Webster | avg_waiting_time | 4 | 11.033747694979525 | 2.671929976556793 |
| SCN-04 | Webster | throughput | 4 | 1660.5 | 61.93814118403404 |
| SCN-04 | Webster | worst_movement_max_wait | 4 | 180.75 | 54.248655897327694 |

