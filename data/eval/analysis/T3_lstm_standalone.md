# T3: LSTM standalone eval - all 6 training attempts

Deployed = lstm-df67afd839d4 (SHIP_WITH_CAVEAT). Only 2 of 6 attempts passed the skill-score gate. MSE is the LSTM's own held-out val/test split, not decomposed per-scenario in the source jsons - no SCN-05-only forecast MSE exists on disk.

| lstm_version | seed | val_mse | test_mse | r2_val | skill_score_nearest_horizon | skill_score_farthest_horizon | gate_verdict | ship | deployed |
|---|---|---|---|---|---|---|---|---|---|
| lstm-3d16a6f573db | 42 | 5.097958110155168 | 0.6263390029632823 | 0.9226676516874697 | -16.271089553833008 | -5.459066390991211 | RETRAIN_OR_DROP | False | False |
| lstm-8aa8fcf9ac44 | 42 | 0.5499554189519004 | 0.2705791350064097 | 0.9916575716395604 | -0.00019121170043945312 | 0.01938861608505249 | RETRAIN_OR_DROP | False | False |
| lstm-ae5c7639ac75 | 42 | 11.431886073941211 | 3.9259182301559887 | 0.8265865193012218 | -39.15557861328125 | -12.555669784545898 | RETRAIN_OR_DROP | False | False |
| lstm-dbd858c520d7 | 42 | 0.5499554189519004 | 0.2705791350064097 | 0.9916575716395604 | -0.00019121170043945312 | 0.01938861608505249 | RETRAIN_OR_DROP | False | False |
| lstm-df67afd839d4 | 42 | 2.205371213328748 | 0.7414713576097045 | 0.966428628205721 | 0.06759995222091675 | 0.1223829984664917 | SHIP_WITH_CAVEAT | True | True |
| lstm-e425e4df765f | 42 | 2.2680733089277227 | 0.7374941531793536 | 0.9654741408001789 | 0.016244590282440186 | 0.10485100746154785 | SHIP_WITH_CAVEAT | True | False |

