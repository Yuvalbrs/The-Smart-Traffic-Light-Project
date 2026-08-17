# Supporting/exploratory: hybrid vs plain / random-lstm, headline KPIs, all 5 scenarios, raw p

| scenario | vs | kpi | direction | median_diff_hybrid_minus_other | ci_lo | ci_hi | n | dropped | p_raw | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| SCN-01 | DQN-plain | avg_wait | lower | 1.5461944801894387 | 0.4797808123958326 | 2.028775054874579 | 14 | 1 | 0.0040283203125 | hybrid worse |
| SCN-01 | DQN-plain | throughput | higher | -0.5 | -2.0 | 1.0 | 14 | 1 | 0.4663581710854011 | no effect detected at n=15 |
| SCN-01 | DQN-plain | worst_max(5b) | lower | 49.0 | 36.0 | 99.5 | 14 | 1 | 0.001708984375 | hybrid worse |
| SCN-01 | DQN-random-lstm | avg_wait | lower | 1.4151205335675008 | -0.8350055055157746 | 2.2147651006711406 | 15 | 0 | 0.0301513671875 | hybrid worse |
| SCN-01 | DQN-random-lstm | throughput | higher | -1.0 | -5.0 | 0.0 | 15 | 0 | 0.009420718750162885 | hybrid worse |
| SCN-01 | DQN-random-lstm | worst_max(5b) | lower | 50.0 | -4.0 | 75.0 | 15 | 0 | 0.03300625766123251 | hybrid worse |
| SCN-02 | DQN-plain | avg_wait | lower | nan | nan | nan | 0 | 15 | nan | no effect detected at n=15 |
| SCN-02 | DQN-plain | throughput | higher | nan | nan | nan | 0 | 15 | nan | no effect detected at n=15 |
| SCN-02 | DQN-plain | worst_max(5b) | lower | nan | nan | nan | 0 | 15 | nan | no effect detected at n=15 |
| SCN-02 | DQN-random-lstm | avg_wait | lower | nan | nan | nan | 0 | 15 | nan | no effect detected at n=15 |
| SCN-02 | DQN-random-lstm | throughput | higher | nan | nan | nan | 0 | 15 | nan | no effect detected at n=15 |
| SCN-02 | DQN-random-lstm | worst_max(5b) | lower | nan | nan | nan | 0 | 15 | nan | no effect detected at n=15 |
| SCN-03 | DQN-plain | avg_wait | lower | 0.54127094238006 | -1.1262659916981597 | 2.0852266259374543 | 6 | 9 | 0.4375 | no effect detected at n=15 |
| SCN-03 | DQN-plain | throughput | higher | 115.0 | -517.0 | 1006.0 | 7 | 8 | 0.578125 | no effect detected at n=15 |
| SCN-03 | DQN-plain | worst_max(5b) | lower | 21.5 | 5.0 | 58.5 | 6 | 9 | 0.0625 | no effect detected at n=15 |
| SCN-03 | DQN-random-lstm | avg_wait | lower | -0.09586729393915439 | -3.211925665788703 | 1.8814097937823482 | 8 | 7 | 0.84375 | no effect detected at n=15 |
| SCN-03 | DQN-random-lstm | throughput | higher | 941.5 | -1019.0 | 1240.0 | 8 | 7 | 0.4609375 | no effect detected at n=15 |
| SCN-03 | DQN-random-lstm | worst_max(5b) | lower | 9.5 | -104.0 | 57.0 | 8 | 7 | 0.9453125 | no effect detected at n=15 |
| SCN-04 | DQN-plain | avg_wait | lower | 0.17648898500943533 | -0.03692354876745374 | 0.3378678978696441 | 13 | 2 | 0.339599609375 | no effect detected at n=15 |
| SCN-04 | DQN-plain | throughput | higher | -1.5 | -381.525 | 1.0 | 14 | 1 | 0.3612628443247907 | no effect detected at n=15 |
| SCN-04 | DQN-plain | worst_max(5b) | lower | 25.0 | 18.0 | 34.0 | 13 | 2 | 0.00244140625 | hybrid worse |
| SCN-04 | DQN-random-lstm | avg_wait | lower | 0.2434042270607688 | -0.020295721594423055 | 0.6439483549066671 | 11 | 4 | 0.0419921875 | hybrid worse |
| SCN-04 | DQN-random-lstm | throughput | higher | 0.0 | -1.0 | 1.0 | 11 | 4 | 0.578125 | no effect detected at n=15 |
| SCN-04 | DQN-random-lstm | worst_max(5b) | lower | 25.0 | -59.0 | 49.0 | 11 | 4 | 0.7646484375 | no effect detected at n=15 |
| SCN-05 | DQN-plain | avg_wait | lower | -1.2792819523774197 | -2.706395048994327 | 8.952056119980648 | 7 | 8 | 0.9375 | no effect detected at n=15 |
| SCN-05 | DQN-plain | throughput | higher | 1131.0 | 800.0 | 1365.0 | 7 | 8 | 0.296875 | no effect detected at n=15 |
| SCN-05 | DQN-plain | worst_max(5b) | lower | 130.0 | -30.0 | 400.0 | 7 | 8 | 0.109375 | no effect detected at n=15 |
| SCN-05 | DQN-random-lstm | avg_wait | lower | 0.5164637641482801 | -5.382371343448072 | 11.508826447703727 | 6 | 9 | 0.5625 | no effect detected at n=15 |
| SCN-05 | DQN-random-lstm | throughput | higher | 1043.0 | 736.5 | 1318.5 | 6 | 9 | 0.03125 | hybrid better |
| SCN-05 | DQN-random-lstm | worst_max(5b) | lower | 233.0 | -6.5 | 587.0 | 6 | 9 | 0.09375 | no effect detected at n=15 |

