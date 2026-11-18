# Report outline

One PDF per team, following sections 2.2 to 7 of the Deliverables PDF. Every number below points to a file in this repo. Take numbers from `results/report_tables.md` (built by `python scripts/report_tables.py`), which uses 3 decimals for metrics and 2 for times, so the same number looks the same everywhere in the report.

Write the text yourselves. The brief says reports are checked for plagiarism and AI-generated text, and any member can be asked about any claim. This file lists what each section needs and where the evidence is. It does not give you sentences to copy.

Use student_test results in sections 3 to 5. Do not report evaluator_test numbers there.

## Title page and intro (half a page)

Team name, members, and the repo link https://github.com/sarayu1198/PathSnap.

Three or four lines on the problem: a noisy GPS track goes in, and the output is the road (directed arc) the car was on at each fix. State the final result: Top-1 0.952 on student_test (tuned LightGBM + HMM, `results/hmm_student_test.csv`, ALL row).

## 2.2 Dataset understanding and EDA

Questions to answer:

1. How did you preprocess the data? Cover cleaning (invalid and duplicate fixes dropped, sorted by time), the per-trip projection to metres, how each fix gets a reference arc from `.route` (direction-aware alignment in `pathsnap/labels.py`), and the 50 m unlabelled rule.
2. What patterns did you see, and which features did you pick and why? Use the six panels of `figures/eda_overview.png`, then the 27 features in `pathsnap/features.py` grouped by type, with one line each on why.

Put in:

| Item | Source | Partition |
|---|---|---|
| Missing-value and data-quality table | `results/report_tables.md` section 2.2, from `results/eda_inventory.csv` | train, student_test |
| Overview figure | `figures/eda_overview.png` | train and student_test (check the figure title) |
| GPS trajectory plot | `figures/eda_trajectory_00000001.png` | train |
| Road-network visualisation | same figure, plus the connectivity table in `results/report_tables.md` | train |
| Folium map screenshot (at least 1) | `report/screenshots/eda_map_train_00000001.png`, taken from `maps/eda_train_00000001.html` | train |
| Labels and candidates table | `results/report_tables.md`, from `results/labels_summary_*.csv` | train, student_test |
| Feature importance | `figures/feature_importance.png`, `results/feature_importance.csv` | train |

Write the partition name under every figure.

Mention the trip 77 label bug: what went wrong with the labels on that trip, how you noticed it, and how the direction penalty in `pathsnap/labels.py` (`DIR_PENALTY_M`, `MOVING_SPEED`, `MOTION_WIN`) fixed it. Only quote before/after numbers you can rerun.

## 3. Model experimentation and comparison

### 3.1 Comparison table

Use the 3.1 table in `results/report_tables.md`. It has family, key configuration, CV Macro F1, CV and test Top-1, CV and test continuity, test Macro F1, test log loss, train time and inference time for all 8 models.

### 3.2 One subsection per model (all 8)

Order: LightGBM, RandomForest, HistGradientBoosting, LogisticRegression, MLP, DecisionTree, KNN, GaussianNB.

For each model answer:

- Why it suits this problem, and its strengths and limits here.
- Its important hyperparameters and values (`results/experiment_log.json`, `config` and `params`), and its loss function.
- Its student_test Top-1, Macro F1 and log loss.
- Whether it overfit or underfit, using the train − CV gap in the 3.2 table of `results/report_tables.md` (from `results/summary_train_cv_test.csv` and `results/train_fit_metrics.csv`).

Tie each answer to the numbers. Facts worth explaining in your own words:

- GaussianNB has a test log loss of 0.429, far above every other model (next worst is KNN at 0.120).
- LogisticRegression and GaussianNB have a train − CV Top-1 gap of 0.000, while LightGBM's is 0.046 and tuned LightGBM's is 0.083.
- KNN is the only model whose inference time (12.42 s for 10 trips) is far above its training time (0.08 s).
- The 8 models span 0.879 to 0.898 Top-1 on student_test, a range of 0.019.

### 3.3 Figures

| Figure | File | Partition |
|---|---|---|
| CV Macro F1 | `figures/cv_macro_f1.png` | train (5-fold GroupKFold) |
| Test Top-1 and Macro F1 | `figures/test_top1_macro_f1.png` | student_test |
| Log loss | `figures/test_log_loss.png` | student_test |
| Confusion matrices | `figures/confusion_matrices.png` (counts also in `results/report_tables.md`) | student_test |
| MLP train/validation loss | `figures/mlp_train_val_loss.png` | train, held-out trips |
| Selection: CV Top-1 | `figures/cv_top1.png` | train |
| Selection: accuracy vs speed | `figures/accuracy_vs_speed.png` | student_test |

The MLP is the only model with an epoch-wise training curve. Do not draw curves for the others.

### Model selection

Explain why LightGBM: best CV Top-1 (0.903), probabilities good enough for the HMM emissions, fast to train and predict, small file. Explain why not RandomForest: same CV Macro F1 (0.942) and nearly the same CV Top-1 (0.902), but 53.79 s training against 12.11 s, and a 116 MB model file.

## 4. Bayesian hyperparameter optimisation

Answer the six questions from the 4th section of `results/report_tables.md`:

1. Which hyperparameters were searched (8 LightGBM parameters).
2. Their ranges (search-space table, taken from `scripts/bayes_opt.py`).
3. The objective: mean 3-fold GroupKFold Top-1 on train.
4. Number of trials: 30, TPE sampler, seed 42.
5. Best values (`artifacts/best_params.json`).
6. Did tuning help? Compare the tuned vs untuned rows.

Put in the tuned vs untuned table, `figures/optuna_history.png` and `figures/tuned_vs_untuned.png`. State the cost: 1489 s of search, 24.8 min. State the trade-off from the table: test Top-1 0.898 to 0.902 and continuity 0.941 to 0.944, but log loss 0.069 to 0.074 and inference 0.30 s to 0.79 s. Tuned LightGBM also has the largest train − CV gap (0.083), which is worth one sentence.

## 5. HMM and Viterbi route reconstruction

Answer from `pathsnap/hmm.py` (module docstring and code) and `artifacts/hmm_params.json`:

1. What the hidden states are.
2. How emissions are computed.
3. How transitions are computed.
4. How gaps and missing candidates are handled.
5. What happens when consecutive fixes land on disconnected roads.
6. Did it help?

Put in the per-trip HMM table from `results/report_tables.md` (section 5), `figures/hmm_vs_classifier.png`, and `figures/hmm_tuning.png` with the grid description (192 combinations on out-of-fold train probabilities).

Where it helped least: trips 00000009 (+0.007) and 00000065 (+0.010) gain the least Top-1. Trip 00000077 has the largest gain (+0.115) but is still the lowest at 0.890, because the car sits still on two-way streets where GPS carries no direction. Overall Top-1 goes from 0.902 to 0.952 (+0.049).

## 6. Geographical visualisation (folium, mandatory)

Screenshot `maps/pred_student_test_00000040.html` into `report/screenshots/pred_map_student_test_00000040.png`. The screenshot must show raw GPS points, the road network, the reference route, the predicted route, the legend and the layer control. Expand the layer control before taking it.

## 7. API, UI and deployment

| Screenshot | Where to take it | Save as |
|---|---|---|
| UI with a trip selected, map and metrics visible | laptop 2, browser | `report/screenshots/ui_student_test_00000040.png` |
| API console with request log lines | laptop 1, uvicorn console | `report/screenshots/api_console.png` |

Three or four lines on the architecture: the UI sends only a trip id; the API loads the track and network without the route, predicts, then reads `.route` on the server to score. The reference route never goes back to the UI (`api/app.py`, `pathsnap/pipeline.py`).

## 8. Conclusion (short)

- Features mattered more than the algorithm: the 8 models span 0.879 to 0.898 test Top-1.
- The HMM gave the largest gain: +0.049 Top-1 (0.902 to 0.952).
- Limits: labels are derived from `.route`, and stationary cars on two-way streets have no usable direction.

## Before submitting

- Every number in the PDF appears in `results/report_tables.md` or the file named next to it.
- Every figure has its partition written under it.
- All four screenshots are in `report/screenshots/` (already taken; retake on the demo laptops if you want a two-laptop setup in the report).
- Tick off `docs/DELIVERABLES_CHECKLIST.md`.
