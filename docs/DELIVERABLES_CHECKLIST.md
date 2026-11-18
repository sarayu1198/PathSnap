# Deliverables checklist

Each requirement from the hackathon documents, and where it lives in this repo. Items marked **TEAM** can only be done by the team (screenshots on the demo laptops, written analysis, final PDF).

## Deliverable 1 – GitHub repository

| Required component | Where |
|---|---|
| Data processing (read, inspect, clean, preprocess) | `pathsnap/data.py`, `scripts/eda.py` |
| Feature engineering | `pathsnap/features.py`, `pathsnap/labels.py`, `scripts/build_dataset.py` |
| Model training (every model) | `scripts/train_models.py` (8 models) |
| Model evaluation, metrics, plots, confusion matrices | `pathsnap/metrics.py`, `scripts/make_plots.py`, `figures/confusion_matrices.png` |
| Bayesian optimisation code, search config, results | `scripts/bayes_opt.py`, `results/optuna_trials.csv`, `artifacts/best_params.json` |
| HMM and Viterbi | `pathsnap/hmm.py`, `scripts/hmm_eval.py` |
| API (endpoints + model loading) | `api/app.py`, `pathsnap/pipeline.py` |
| UI | `ui/index.html`, `ui/serve.py` |
| Deployment (config, deps, instructions) | `requirements.txt`, `README.md` |
| Visualisations (folium + plots) | `pathsnap/viz.py`, `maps/*.html`, `scripts/make_plots.py` |
| Supporting results (CSV, logs, configs) | `results/`, `artifacts/hmm_params.json` |
| Saved model weights | `artifacts/final_model.joblib`, `artifacts/models/*.joblib` (RandomForest excluded: 116 MB > GitHub limit, regenerate with `train_models.py`) |
| README with install / run / deploy | `README.md` |
| No secrets committed | nothing secret in the repo |

## Deliverable 2 – Technical report (PDF) — TEAM writes it; evidence below

| Report section | Evidence in repo |
|---|---|
| 2.2 Preprocessing, patterns, features + why | `results/eda_summary.txt`, `results/eda_inventory.csv`, `pathsnap/features.py` (feature list with comments) |
| Missing-value analysis | `eda_inventory.csv` columns `n_nan`, `n_nonmonotonic_t` (all zero), gap/jump columns |
| GPS trajectory plots | `figures/eda_trajectory_*.png`, `figures/eda_overview.png` |
| Road-network visualisation | `figures/eda_trajectory_*.png`, `maps/eda_*.html`; connectivity columns in `eda_inventory.csv` |
| Folium map (≥1) | `maps/eda_*.html`, `maps/pred_student_test_*.html` |
| Partition used for each analysis | stated in every figure title / file name |
| 3.1 Comparison table (CV macro-F1, Top-1, continuity, test macro-F1, log loss, inference time) | `results/model_comparison.csv` |
| 3.2 Per-model answers (hyperparameters, loss, test metrics, over/underfitting) | configs in `results/experiment_log.json`; fold spread in `results/cv_folds.csv` (CV vs test gap = over/underfitting evidence) |
| 3.3 CV macro-F1 plot | `figures/cv_macro_f1.png` |
| Student-test Top-1 + macro-F1 plot | `figures/test_top1_macro_f1.png` |
| Log loss plot | `figures/test_log_loss.png` |
| Confusion matrices | `figures/confusion_matrices.png` |
| Training curves (only epoch-wise models) | `figures/mlp_train_val_loss.png` (train vs validation log loss on held-out trips, from `scripts/mlp_history.py`) and `figures/mlp_training_curve.png` (sklearn early-stopping history). MLP only; tree/linear models have no epoch history, so none are drawn |
| Plots justifying selection | `figures/cv_top1.png`, `figures/accuracy_vs_speed.png` |
| 4 Bayesian optimisation (params, ranges, objective, trials, best, tuned vs untuned) | `scripts/bayes_opt.py` (search space), `results/optuna_trials.csv`, `artifacts/best_params.json`, `results/tuned_vs_untuned.csv`, `figures/optuna_history.png`, `figures/tuned_vs_untuned.png` |
| 5 HMM (states, emission, transition, gaps, disconnected segments, improvement) | docstring of `pathsnap/hmm.py`, `results/hmm_tuning.csv`, `results/hmm_student_test.csv`, `figures/hmm_vs_classifier.png`, `figures/hmm_tuning.png` |
| 6 Folium: GPS, network, reference, predicted, legend, layer control | `maps/pred_student_test_*.html` (screenshot one for the PDF — TEAM) |
| 7 UI + API screenshots | **TEAM**: screenshot UI on laptop 2 and the uvicorn console on laptop 1 |

## Evaluation-day rules

- API on laptop 1, UI on laptop 2, UI lists the 10 evaluator trips (`GET /trips` on `evaluator_test/`).
- The UI sends only the trip id. The API loads the track and network with `with_route=False`, predicts, and only then reads `.route` to score.
- The reference route is never returned to the UI. The response holds only predicted geometry and metrics.
- Every request is logged in the uvicorn console. Show this to the evaluator.
- No hardcoded outputs: predictions come from `pipeline.py`. The evaluator partition is not read by any training or tuning script.
- Works for all trips: tested on all 10 student-test trips through the API.

## Leakage notes (for the viva)

- CV is GroupKFold by trip, so no trip appears in both train and validation folds.
- StandardScaler sits inside each sklearn pipeline, so it's fitted on training folds only.
- Optuna objective uses train-only 3-fold GroupKFold. Student-test is used only after the search.
- HMM parameters are tuned on out-of-fold train probabilities and reported on student-test.
- Per-fix reference arcs are derived from `.route` (see `pathsnap/labels.py`). They are an approximation; the limitations are discussed in the report.
