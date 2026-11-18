# PathSnap: map-matching and route reconstruction

PathSnap takes a noisy GPS track and the road network around it, and works out which road segment (directed arc) the vehicle was on at every GPS fix. It then rebuilds the continuous route that was driven.

The pipeline runs in four stages:

1. **Candidates.** Every GPS fix gets up to 10 nearby directed arcs (within 60 m).
2. **Classifier.** A tuned LightGBM model gives each (fix, arc) pair a probability of being correct.
3. **HMM / Viterbi.** The decoder combines those probabilities with road-network connectivity, which gives one consistent route instead of independent per-fix guesses.
4. **API + UI.** A FastAPI service (laptop 1) runs the pipeline for a selected trip. A browser UI (laptop 2) shows the result on a map.

## Results (student-test partition, 10 trips, 19,133 scored GPS fixes)

| | Top-1 accuracy | Route continuity | Route F1 |
|---|---|---|---|
| Tuned LightGBM only | 0.9025 | 0.9456 | 0.8077 |
| **Tuned LightGBM + HMM** | **0.9518** | **0.9815** | **0.9692** |

- **Top-1 accuracy:** the share of GPS fixes whose predicted arc equals the reference arc.
- **Route continuity:** the share of consecutive predictions that are the same arc or directly connected arcs.
- **Route F1:** overlap between the set of arcs in the predicted route and the reference route.

Per trip (`results/hmm_student_test.csv`):

| Trip | Fixes | Classifier Top-1 | +HMM Top-1 | +HMM continuity | +HMM route F1 |
|---|---|---|---|---|---|
| 00000000 | 2503 | 0.951 | 0.968 | 0.994 | 0.957 |
| 00000007 | 1354 | 0.965 | 1.000 | 1.000 | 0.992 |
| 00000009 | 1049 | 0.888 | 0.894 | 0.940 | 0.946 |
| 00000040 | 2995 | 0.939 | 0.990 | 0.996 | 0.997 |
| 00000052 | 262 | 0.973 | 1.000 | 0.966 | 0.989 |
| 00000056 | 3931 | 0.927 | 0.970 | 0.966 | 0.967 |
| 00000065 | 768 | 0.917 | 0.927 | 0.982 | 0.980 |
| 00000076 | 1486 | 0.939 | 0.968 | 0.970 | 0.948 |
| 00000077 | 3911 | 0.775 | 0.890 | 0.991 | 0.959 |
| 00000091 | 874 | 0.924 | 0.942 | 0.970 | 0.981 |

### Model comparison

From `results/model_comparison.csv`. CV is 5-fold GroupKFold on the 80 train trips, grouped by trip; test is the 10 student-test trips. Macro F1 and log loss are measured at the candidate level, i.e. "is this arc the right one".

| Model | Family | CV Macro F1 | CV Top-1 | Test Top-1 | Test Macro F1 | Test log loss | Train time |
|---|---|---|---|---|---|---|---|
| **LightGBM** | Boosting | 0.942 | **0.903** | **0.898** | **0.940** | 0.069 | 12 s |
| RandomForest | Bagging | 0.942 | 0.902 | 0.895 | 0.939 | 0.069 | 54 s |
| HistGradientBoosting | Boosting | 0.941 | 0.900 | 0.894 | 0.938 | 0.074 | 9 s |
| LogisticRegression | Linear | 0.935 | 0.898 | 0.893 | 0.930 | 0.086 | 5 s |
| MLP | Neural net | 0.934 | 0.898 | 0.889 | 0.931 | 0.078 | 68 s |
| DecisionTree | Tree | 0.934 | 0.894 | 0.888 | 0.930 | 0.084 | 50 s |
| KNN | Instance-based | 0.931 | 0.890 | 0.880 | 0.926 | 0.120 | <1 s |
| GaussianNB | Probabilistic | 0.913 | 0.885 | 0.879 | 0.907 | 0.429 | <1 s |

LightGBM was selected because it has the highest CV Top-1 accuracy, the metric closest to the actual task, and the best student-test Top-1 and Macro F1. It's also small and fast. Random Forest is statistically tied, but it trains about 4× slower and its saved model is 116 MB.

### Bayesian optimisation

From `results/tuned_vs_untuned.csv`: 30 Optuna TPE trials, objective = mean 3-fold GroupKFold Top-1 accuracy on train.

| | CV5 Top-1 | CV5 Macro F1 | Test Top-1 | Test Macro F1 | Test log loss | Inference (10 trips) |
|---|---|---|---|---|---|---|
| Untuned LightGBM | 0.9027 | 0.9419 | 0.8982 | 0.9396 | 0.0694 | 0.30 s |
| Tuned LightGBM | 0.9046 | 0.9412 | 0.9025 | 0.9401 | 0.0738 | 0.79 s |

Best parameters are in `artifacts/best_params.json`: 550 trees, learning rate 0.0996, 222 leaves, `min_child_samples` 215, subsample 0.89, colsample 0.45.

## Method

### Data

Each trip folder `<id>/` contains:

| File | Contents |
|---|---|
| `<id>.track` | lon, lat, time (s), at about 1 Hz |
| `<id>.nodes` | lon, lat; the row index is the node id |
| `<id>.arcs` | from_node, to_node; the row index is the arc id |
| `<id>.route` | the reference arc sequence (used only for labels and scoring) |

Trips are spread across several continents, so coordinates go through a local equirectangular projection centred on each trip. Over a trip's extent this keeps distance error far below GPS noise (`pathsnap/data.py`). Cleaning drops non-finite or out-of-range fixes and duplicate timestamps, and sorts by time. The raw data has no NaNs and no out-of-order timestamps, but some trips have large gaps: up to 42 h in train trip 00000090 (see `results/eda_summary.txt`).

### Reference labels

`.route` gives the driven arcs but not which GPS fix belongs to which arc. `pathsnap/labels.py` aligns the fixes to the route arcs with a monotone dynamic programme. Each fix gets one route position, positions never go backwards, and total distance is minimised.

Distance alone cannot separate an arc from its reverse, and routes often contain A followed by reverse(A) at turnarounds. So when the vehicle is clearly moving (> 2 m/s over a ±5 s window), an arc pointing against the motion costs an extra 25 m. Fixes more than 50 m from their aligned reference arc are treated as unlabelled and excluded from training and scoring: 11% of train fixes, mostly track sections the reference route doesn't cover, and 0.1% of student-test fixes.

### Candidates and features

Arcs are densified every 10 m into a KD-tree. Each fix takes the 10 nearest arcs within 60 m. The true arc is missing from the candidate set for 0.66% of labelled train fixes and 0.24% of student-test fixes.

Features (27, `pathsnap/features.py`):

- **Distance:** distance to the arc, its rank, its gap and ratio to the nearest candidate, number of candidates.
- **Projection:** position along the arc, whether the projection is clamped at an endpoint, arc length.
- **Heading:** heading agreement between GPS motion and arc direction (instantaneous and ±3-fix window), speed.
- **Context window:** mean and max distance of the ±3 neighbouring fixes to the same arc, and how often the arc is a candidate for those neighbours.
- **Direction through stops:** signed along-arc displacement, plus the heading of the last and next clearly-moving moment. This keeps direction information through stops; these two are the 3rd and 4th most important features (`results/feature_importance.csv`).
- **Topology:** whether a reverse arc exists, node degrees.
- **Time gaps:** time gap to the previous and next fix.

### HMM / Viterbi

Implemented in `pathsnap/hmm.py`:

- **Hidden states:** the candidate arcs of each fix.
- **Emissions:** `w_e * log(p / sum p)`, the classifier probability normalised over the fix's candidates.
- **Transitions:** `-|d_route - d_gps| / (beta * sqrt(dt))`, minus a U-turn penalty and a backward-movement penalty. `d_route` is the road-network distance between the two projected points (bounded Dijkstra); `d_gps` is the straight-line distance between the fixes.
- **Unreachable pairs** get probability 0.
- **Gaps:** if a gap is over 60 s, or no candidate pair is reachable, the chain is broken and decoding restarts. Breaks are counted in the output.
- **Route stitching:** the per-fix arcs are joined with shortest paths to give a continuous route.

Decoder parameters were grid-searched (192 combinations) on **out-of-fold** classifier probabilities for 20 train trips, so the emissions are as noisy as on unseen trips. The best values all sit inside the search range: `w_e = 0.1`, `beta = 5`, `uturn_pen = 30` (`artifacts/hmm_params.json`).

HMM decoding improves Top-1 on every student-test trip. Trip 00000077 is the hardest: many errors happen while the car is stopped for minutes on two-way streets, where GPS gives no direction information.

## Repository layout

```
pathsnap/                 core library
  data.py                 loading, cleaning, local projection
  network.py              arc geometry, KD-tree spatial index, bounded Dijkstra
  labels.py               per-fix reference arcs from .route
  features.py             candidate generation + feature engineering
  metrics.py              candidate metrics, top-1, continuity, route P/R/F1
  hmm.py                  lattice, Viterbi, route stitching
  pipeline.py             end-to-end inference + server-side scoring
  viz.py                  folium maps
scripts/                  one script per experiment step (see below)
api/app.py                FastAPI service (laptop 1)
ui/index.html, serve.py   evaluator UI (laptop 2)
artifacts/                final_model.joblib, hmm_params.json, best_params.json, models/*.joblib
results/                  experiment outputs: CSV, JSON, logs
figures/                  plots used in the report
maps/                     folium maps (EDA + predictions for all student-test trips)
docs/DELIVERABLES_CHECKLIST.md   requirement -> file mapping
report/OUTLINE.md         report plan: what each section needs and where the evidence is
report/screenshots/       folium, UI and API screenshots for the PDF
```

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

The dataset is not in the repo. Put the partition folders side by side:

```
<data folder>/
  train/<trip>/<trip>.{track,nodes,arcs,route}
  student_test/...
  evaluator_test/...      (on evaluation day)
```

Then point `PATHSNAP_DATA` at that folder. If unset, it defaults to the parent folder of the repo.

```powershell
$env:PATHSNAP_DATA = "C:\path\to\data folder"     # PowerShell
```
```bash
export PATHSNAP_DATA=/path/to/data_folder          # bash
```

## Reproducing the experiments

Run from the repo root, in this order. Seeds are fixed at 42. Only `train` and `student_test` are used; no script reads the evaluator partition.

| Step | Command | Writes | Approx. time |
|---|---|---|---|
| 1. EDA | `python scripts/eda.py` | `results/eda_*`, `figures/eda_*`, `maps/eda_*` | 2 min |
| 2. Candidates + features + labels | `python scripts/build_dataset.py` | `artifacts/features_*.pkl`, `results/labels_summary_*.csv` | 1 min |
| 3. 8-model comparison | `python scripts/train_models.py` | `results/model_comparison.csv`, `cv_folds.csv`, `experiment_log.json`, `artifacts/models/` | 20 min |
| 4. MLP train/val loss curve | `python scripts/mlp_history.py` | `results/mlp_epoch_history.csv`, `figures/mlp_train_val_loss.png` | 3 min |
| 5. Bayesian optimisation | `python scripts/bayes_opt.py 30` | `results/optuna_trials.csv`, `tuned_vs_untuned.csv`, `artifacts/final_model.joblib` | 30 min |
| 6. HMM tuning + evaluation | `python scripts/hmm_eval.py LightGBM_tuned` | `artifacts/hmm_params.json`, `results/hmm_*.csv`, `maps/pred_student_test_*.html` | 10 min |
| 7. Figures | `python scripts/make_plots.py` | `figures/*.png` | 1 min |
| 8. Training-set metrics (over/underfitting) | `python scripts/train_fit_eval.py` | `results/train_fit_metrics.csv` | 3 min |
| 9. Train / CV / test summary tables | `python scripts/summary_tables.py` | `results/summary_train_cv_test.{csv,md}` | <1 s |
| 10. Report tables | `python scripts/report_tables.py` | `results/report_tables.md` | <1 s |

The large intermediate tables (`artifacts/features_*.pkl`, `oof_*.pkl`, `test_pred_*.pkl`) and `RandomForest.joblib` (116 MB, over GitHub's file limit) are git-ignored. Steps 2–5 regenerate them.

## Running the demo (two laptops)

The trained model and HMM parameters are committed, so the demo needs no training.

### Laptop 1: API

1. Clone the repo and run `pip install -r requirements.txt`.
2. Copy the evaluator trips into `<data folder>/evaluator_test/` and set `PATHSNAP_DATA`.
3. Start the API:
   ```powershell
   $env:PATHSNAP_PARTITION = "evaluator_test"
   python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```
   If `evaluator_test` doesn't exist, the API falls back to `student_test`.
4. Find this laptop's IP with `ipconfig` (IPv4 address). If Windows asks, allow Python through the firewall on private networks.
5. Keep this console visible. Every request is logged here (trip, number of fixes, time, accuracy), and the evaluator needs to see it.

### Laptop 2: UI

1. Clone the repo; no Python packages are needed besides the standard library.
2. Start the UI server:
   ```powershell
   python ui/serve.py --port 8080
   ```
3. Open `http://localhost:8080/?api=http://<laptop-1-ip>:8000` in a browser. You can also type the API address into the box at the top and press **Connect**.
4. The left panel lists the 10 trips. Clicking one sends only the trip id to the API, then draws:
   - raw GPS fixes (blue)
   - the predicted route (pink)
   - matched points (orange, toggle in the layer control)

   Next to the map it shows Top-1 accuracy (HMM and classifier-only), continuity, route precision, recall and F1, and the timing.

   Adding `&trip=<id>` to the URL runs that trip as soon as the UI connects.

Both laptops must be on the same network. Test it once before the evaluation.

### API reference

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | status, partition, model |
| GET | `/trips` | – | list of trip ids in the served partition |
| POST | `/predict` | `{"trip_id": "00000000"}` | GPS points, matched points, predicted route geometry and arc ids, metrics, timings |

The model receives only the GPS track and the road network: `load_trip(..., with_route=False)`. The reference route is read after prediction, only to compute the metrics, and is never included in the response.

## Limitations

- Reference arcs per fix are derived from the route, not given. Where the vehicle is stationary on a two-way street, the correct direction is ambiguous for both the labels and the model.
- Fixes far from the reference route (sections the route doesn't cover) are not scored.
- The equirectangular projection assumes a trip spans tens of kilometres, not hundreds.
- Bounded Dijkstra uses a cutoff of `max(300 m, 3·d_gps + 100 m)`. Longer detours inside a single GPS step count as unreachable and break the chain.
