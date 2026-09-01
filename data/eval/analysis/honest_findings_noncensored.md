# Honest findings, gridlock-censored episodes FULLY excluded (stricter than T1/T2)

SCN-01: DQN-hybrid is worse than DQN-plain on avg wait among non-censored episodes. SCN-04: the DQN-beats-Webster story. Both are reported here, not softened.

| scenario | algorithm | kpi | n_noncensored | mean_noncensored | std_noncensored |
|---|---|---|---|---|---|
| SCN-01 | DQN-hybrid | avg_waiting_time | 45 | 18.017671865966967 | 0.8847189486286576 |
| SCN-01 | DQN-hybrid | throughput | 45 | 794.0444444444445 | 32.4744343890527 |
| SCN-01 | DQN-hybrid | worst_movement_max_wait | 45 | 100.57777777777778 | 4.938203983824341 |
| SCN-01 | DQN-plain | avg_waiting_time | 45 | 17.379728115828236 | 0.9633096640878827 |
| SCN-01 | DQN-plain | throughput | 45 | 794.8444444444444 | 31.769736568252522 |
| SCN-01 | DQN-plain | worst_movement_max_wait | 45 | 99.97777777777777 | 3.6586689636886507 |
| SCN-01 | Webster | avg_waiting_time | 15 | 33.06286013777348 | 1.2354470247484965 |
| SCN-01 | Webster | throughput | 15 | 790.4666666666667 | 32.8717565845268 |
| SCN-01 | Webster | worst_movement_max_wait | 15 | 102.13333333333334 | 0.6399404734221844 |
| SCN-04 | DQN-hybrid | avg_waiting_time | 45 | 18.924084392787886 | 0.9150650816364483 |
| SCN-04 | DQN-hybrid | throughput | 45 | 1670.8666666666666 | 37.75796123155534 |
| SCN-04 | DQN-hybrid | worst_movement_max_wait | 45 | 102.8 | 3.852979956534234 |
| SCN-04 | DQN-plain | avg_waiting_time | 45 | 18.03102606862543 | 0.9266801421791688 |
| SCN-04 | DQN-plain | throughput | 45 | 1671.2444444444445 | 38.45017765389598 |
| SCN-04 | DQN-plain | worst_movement_max_wait | 45 | 102.24444444444444 | 2.2578706335793024 |
| SCN-04 | Webster | avg_waiting_time | 15 | 17.790341214653882 | 0.8716128614458153 |
| SCN-04 | Webster | throughput | 15 | 1672.8666666666666 | 38.9521073456452 |
| SCN-04 | Webster | worst_movement_max_wait | 15 | 96.0 | 13.96424004376894 |

