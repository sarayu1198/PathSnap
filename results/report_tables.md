# Report tables

Built by `scripts/report_tables.py` from `results/` and `artifacts/`. Metrics have 3 decimals and times 2, everywhere.

## 2.2 Dataset and EDA

### Missing values and data quality (train and student_test)

| Partition | Trips | GPS fixes | NaN values | Out-of-order timestamps | Trips with a gap > 10 s | Longest gap (s) | Jumps > 100 m |
|---|---|---|---|---|---|---|---|
| train | 80 | 208417 | 0 | 0 | 40 | 153064 | 227 |
| student_test | 10 | 19160 | 0 | 0 | 2 | 210 | 0 |

Longest gap: trip 00000090 (train), 153064 s = 42.5 h.

### Road-network connectivity

| Partition | Median nodes | Median arcs | Median two-way arc share | Median largest SCC share | Isolated nodes | Median route connected share |
|---|---|---|---|---|---|---|
| train | 9608 | 19526 | 0.951 | 0.964 | 0 | 1.000 |
| student_test | 6596 | 13145 | 0.933 | 0.961 | 0 | 1.000 |

### Trip 00000001 (train), the trip in `figures/eda_trajectory_00000001.png`

| GPS fixes | Nodes | Arcs | Route arcs | Path length (km) | Weak components | Strong components | Largest SCC share | Two-way arc share | Median GPS-to-route distance (m) |
|---|---|---|---|---|---|---|---|---|---|
| 1315 | 88809 | 165790 | 525 | 50.2 | 175 | 1942 | 0.957 | 0.819 | 4.3 |

### Labels and candidates

| Partition | GPS fixes | Labelled fixes | Unlabelled share (> 50 m from route) | Candidate miss rate | Candidate rows |
|---|---|---|---|---|---|
| train | 208417 | 185161 | 0.112 | 0.007 | 1375294 |
| student_test | 19160 | 19133 | 0.001 | 0.002 | 140950 |

### Feature importance (tuned LightGBM, 27 features)

| Rank | Feature | Gain share |
|---|---|---|
| 1 | dist_minus_min | 0.194 |
| 2 | dist_rank | 0.173 |
| 3 | cos_head_next_move | 0.146 |
| 4 | cos_head_last_move | 0.109 |
| 5 | arc_len | 0.043 |
| 6 | at_endpoint | 0.043 |
| 7 | along_disp | 0.041 |
| 8 | dist_ratio_min | 0.039 |
| 9 | proj_t | 0.035 |
| 10 | dist | 0.028 |
| 11 | along_disp_norm | 0.024 |
| 12 | ctx_max_dist | 0.022 |
| 13 | ctx_mean_dist | 0.020 |
| 14 | win_speed | 0.020 |
| 15 | n_cand | 0.009 |
| 16 | cos_head_win | 0.008 |
| 17 | speed | 0.007 |
| 18 | to_in_deg | 0.006 |
| 19 | abs_head_diff | 0.006 |
| 20 | from_out_deg | 0.006 |
| 21 | to_out_deg | 0.006 |
| 22 | has_reverse | 0.005 |
| 23 | ctx_cand_frac | 0.005 |
| 24 | cos_head | 0.004 |
| 25 | dt_prev | 0.000 |
| 26 | dt_next | 0.000 |
| 27 | head_reliable | 0.000 |

## 3. Model comparison

### 3.1 All 8 models (CV = 5-fold GroupKFold on train; test = student_test)

| Model | Family | Key configuration | CV Macro F1 | CV Top-1 | CV continuity | Test Top-1 | Test continuity | Test Macro F1 | Test log loss | Train time (s) | Inference time, 10 trips (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM | Boosting | 400 trees, lr=0.05, 63 leaves, bagging 0.8, colsample 0.8 | 0.942 ± 0.014 | 0.903 | 0.932 | 0.898 | 0.941 | 0.940 | 0.069 | 12.11 | 0.36 |
| RandomForest | Ensemble (bagging) | 150 trees, max_depth=18, leaf=5, max_samples=0.3 | 0.942 ± 0.014 | 0.902 | 0.926 | 0.895 | 0.936 | 0.939 | 0.069 | 53.79 | 0.29 |
| HistGradientBoosting | Boosting | 300 iters, lr=0.1, 31 leaves | 0.941 ± 0.014 | 0.900 | 0.926 | 0.894 | 0.934 | 0.938 | 0.074 | 8.70 | 0.15 |
| LogisticRegression | Linear | StandardScaler, L2, C=1.0, lbfgs | 0.935 ± 0.014 | 0.898 | 0.929 | 0.893 | 0.940 | 0.930 | 0.086 | 4.81 | 0.03 |
| MLP | Neural network | StandardScaler, (64,32) ReLU, Adam lr=1e-3, early stopping, 600k-row subsample | 0.934 ± 0.015 | 0.898 | 0.932 | 0.888 | 0.943 | 0.931 | 0.078 | 68.19 | 0.10 |
| DecisionTree | Tree-based | max_depth=12, min_samples_leaf=50 | 0.934 ± 0.016 | 0.894 | 0.919 | 0.888 | 0.933 | 0.930 | 0.084 | 49.96 | 0.04 |
| KNN | Instance-based | StandardScaler, k=25, distance-weighted, 150k-row subsample | 0.931 ± 0.015 | 0.890 | 0.915 | 0.880 | 0.928 | 0.926 | 0.120 | 0.08 | 12.42 |
| GaussianNB | Probabilistic | default var_smoothing=1e-9 | 0.913 ± 0.010 | 0.885 | 0.912 | 0.879 | 0.924 | 0.907 | 0.429 | 0.51 | 0.07 |

### 3.2 Over/underfitting evidence (train vs CV gap)

| Model | Train Top-1 | CV Top-1 | Test Top-1 | Train − CV Top-1 | Train Macro F1 | CV Macro F1 | Train − CV Macro F1 | Train log loss | CV log loss |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.949 | 0.903 | 0.898 | 0.046 | 0.968 | 0.942 | 0.026 | 0.043 | 0.075 |
| RandomForest | 0.930 | 0.902 | 0.895 | 0.028 | 0.959 | 0.942 | 0.017 | 0.052 | 0.073 |
| HistGradientBoosting | 0.918 | 0.900 | 0.894 | 0.018 | 0.951 | 0.941 | 0.011 | 0.064 | 0.076 |
| LogisticRegression | 0.899 | 0.898 | 0.893 | 0.000 | 0.935 | 0.935 | 0.001 | 0.089 | 0.090 |
| MLP | 0.921 | 0.898 | 0.888 | 0.023 | 0.951 | 0.934 | 0.017 | 0.063 | 0.086 |
| DecisionTree | 0.920 | 0.894 | 0.888 | 0.026 | 0.950 | 0.934 | 0.016 | 0.063 | 0.095 |
| KNN | 0.917 | 0.890 | 0.880 | 0.027 | 0.944 | 0.931 | 0.013 | 0.095 | 0.131 |
| GaussianNB | 0.885 | 0.885 | 0.879 | 0.000 | 0.914 | 0.913 | 0.001 | 0.411 | 0.421 |
| LightGBM_tuned | 0.988 | 0.905 | 0.902 | 0.083 | 0.994 | 0.941 | 0.053 | 0.015 | n/a |

A large positive train − CV gap means the model fits the training trips better than unseen trips (overfitting). Near-zero gap with a low score means the model is too simple (underfitting).

### Confusion matrices on student_test (candidate level: is this arc the right one)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| LightGBM | 120018 | 1765 | 2185 | 16902 |
| RandomForest | 120092 | 1691 | 2273 | 16814 |
| HistGradientBoosting | 119751 | 2032 | 2076 | 17011 |
| LogisticRegression | 119270 | 2513 | 2148 | 16939 |
| MLP | 119799 | 1984 | 2494 | 16593 |
| DecisionTree | 119590 | 2193 | 2381 | 16706 |
| KNN | 119239 | 2544 | 2385 | 16702 |
| GaussianNB | 117307 | 4476 | 2002 | 17085 |

## 4. Bayesian optimisation

Sampler: Optuna TPE, seed 42. Trials: 30. Objective: mean 3-fold GroupKFold Top-1 on train. Best trial: 13 with CV3 Top-1 0.903. Search time: 1489 s (24.8 min).

### Search space and best values

| Hyperparameter | Type | Low | High | Scale | Step | Best value |
|---|---|---|---|---|---|---|
| n_estimators | int | 100 | 800 | linear | 50 | 550 |
| learning_rate | float | 0.01 | 0.2 | log |  | 0.09957 |
| num_leaves | int | 15 | 255 | log |  | 222 |
| min_child_samples | int | 10 | 500 | log |  | 215 |
| subsample | float | 0.5 | 1.0 | linear |  | 0.8932 |
| colsample_bytree | float | 0.4 | 1.0 | linear |  | 0.446 |
| reg_lambda | float | 1e-3 | 10.0 | log |  | 0.002577 |
| reg_alpha | float | 1e-3 | 10.0 | log |  | 0.001456 |

### Tuned vs untuned LightGBM

| Model | CV5 Top-1 | CV5 Macro F1 | Test Top-1 | Test continuity | Test Macro F1 | Test log loss | Train time (s) | Inference time, 10 trips (s) |
|---|---|---|---|---|---|---|---|---|
| LightGBM_untuned | 0.903 | 0.942 | 0.898 | 0.941 | 0.940 | 0.069 | 11.61 | 0.30 |
| LightGBM_tuned | 0.905 | 0.941 | 0.902 | 0.944 | 0.940 | 0.074 | 18.74 | 0.79 |

## 5. HMM and Viterbi (student_test)

| Trip | Fixes scored | Top-1 classifier | Top-1 + HMM | Top-1 gain | Continuity classifier | Continuity + HMM | Route F1 classifier | Route F1 + HMM | Chain breaks | Viterbi time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 00000000 | 2503 | 0.951 | 0.968 | 0.017 | 0.976 | 0.994 | 0.782 | 0.957 | 2 | 0.04 |
| 00000007 | 1354 | 0.965 | 1.000 | 0.035 | 0.999 | 1.000 | 0.972 | 0.992 | 0 | 0.02 |
| 00000009 | 1049 | 0.888 | 0.894 | 0.007 | 0.902 | 0.940 | 0.851 | 0.946 | 0 | 0.02 |
| 00000040 | 2995 | 0.939 | 0.990 | 0.051 | 0.972 | 0.996 | 0.761 | 0.997 | 0 | 0.05 |
| 00000052 | 262 | 0.973 | 1.000 | 0.027 | 0.946 | 0.966 | 0.639 | 0.989 | 0 | 0.00 |
| 00000056 | 3931 | 0.927 | 0.970 | 0.043 | 0.914 | 0.966 | 0.856 | 0.967 | 1 | 0.06 |
| 00000065 | 768 | 0.917 | 0.927 | 0.010 | 0.954 | 0.982 | 0.702 | 0.980 | 1 | 0.01 |
| 00000076 | 1486 | 0.939 | 0.968 | 0.028 | 0.919 | 0.970 | 0.683 | 0.948 | 0 | 0.02 |
| 00000077 | 3911 | 0.775 | 0.890 | 0.115 | 0.947 | 0.991 | 0.858 | 0.959 | 0 | 0.06 |
| 00000091 | 874 | 0.924 | 0.942 | 0.017 | 0.915 | 0.970 | 0.651 | 0.981 | 0 | 0.01 |
| ALL (weighted by fixes) | 19133 | 0.902 | 0.952 | 0.049 | 0.946 | 0.981 | 0.808 | 0.969 | 4 | 0.04 |

Chosen decoder parameters (`artifacts/hmm_params.json`): w_e = 0.1, beta = 5, uturn_pen = 30, back_pen = 0, back_tol = 8, gap_s = 60.

Grid searched (192 combinations, scored on out-of-fold train probabilities): w_e in [0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 1.0, 2.0]; beta in [1.5, 2.0, 3.0, 5.0, 8.0, 20.0]; uturn_pen in [15.0, 30.0]; back_pen in [0.0, 2.0].

### Best 5 grid points

| w_e | beta | uturn_pen | back_pen | back_tol | gap_s | train_oof_top1 |
|---|---|---|---|---|---|---|
| 0.1 | 5.0 | 30.0 | 0.0 | 8.0 | 60.0 | 0.955 |
| 0.1 | 5.0 | 15.0 | 0.0 | 8.0 | 60.0 | 0.955 |
| 0.15 | 3.0 | 15.0 | 0.0 | 8.0 | 60.0 | 0.954 |
| 0.15 | 3.0 | 30.0 | 0.0 | 8.0 | 60.0 | 0.953 |
| 0.05 | 8.0 | 15.0 | 0.0 | 8.0 | 60.0 | 0.953 |
