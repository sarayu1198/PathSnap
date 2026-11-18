# PathSnap: map-matching and route reconstruction

PathSnap takes a noisy GPS track and the trip's road network, and works out which road segment (arc) the vehicle was on at every GPS fix. It also returns the full driven route.

How it works:

1. **Candidate generation.** Each GPS fix gets up to 10 directed arcs within 60 m. Distances are computed in a local metric projection.
2. **Candidate classifier.** A LightGBM model (selected from 8 models, then tuned with Optuna TPE) scores each (fix, arc) pair. It uses distance, heading, along-track movement, a context window of neighbouring fixes and network degree.
3. **HMM / Viterbi.** The classifier probabilities act as emissions. Transitions compare the road-network distance with the straight-line GPS distance, and also penalise U-turns and backwards movement. Gaps and unreachable steps break the chain cleanly.
4. **FastAPI backend** (laptop 1) and a **browser UI** (laptop 2). The UI only sends a trip id. The API runs the full pipeline and scores the result server-side against the reference route, which never leaves the server.

## Repository layout

```
pathsnap/            core library
  data.py            loading, cleaning, local projection
  network.py         arc geometry, KD-tree spatial index, bounded Dijkstra
  labels.py          per-fix reference arcs from .route (monotone DP alignment)
  features.py        candidate generation + feature engineering
  metrics.py         candidate metrics, top-1, continuity, route P/R/F1
  hmm.py             lattice construction, Viterbi, route stitching
  pipeline.py        end-to-end inference + server-side scoring
  viz.py             folium maps
scripts/
  eda.py             dataset exploration -> results/eda_*, figures/eda_*, maps/eda_*
  build_dataset.py   candidates/features/labels for train + student_test
  train_models.py    8 models, 5-fold GroupKFold CV + student_test evaluation
  bayes_opt.py       Optuna TPE search for LightGBM, tuned vs untuned
  hmm_eval.py        HMM parameter tuning (train OOF) + classifier vs HMM on student_test
  make_plots.py      report figures from results/*.csv
api/app.py           FastAPI service
ui/index.html        evaluator UI (Leaflet), ui/serve.py to host it
artifacts/           final_model.joblib, hmm_params.json, best_params.json, all trained models
results/             CSV/JSON experiment outputs and logs
figures/             PNG plots
maps/                folium HTML maps
```

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

Data is not committed. Point `PATHSNAP_DATA` at the folder that contains the partition folders (`train/`, `student_test/`, `evaluator_test/`), each holding `<trip>/<trip>.{track,nodes,arcs,route}`. By default this is the parent folder of the repo.

```bash
# Windows PowerShell
$env:PATHSNAP_DATA = "C:\path\to\Student folder"
# bash
export PATHSNAP_DATA=/path/to/data
```

## Reproducing the experiments

Run these in order. Each script writes to `results/`, `figures/`, `maps/` and `artifacts/`.

```bash
python scripts/eda.py
python scripts/build_dataset.py
python scripts/train_models.py
python scripts/bayes_opt.py 40
python scripts/hmm_eval.py LightGBM_tuned
python scripts/make_plots.py
```

Every random seed is fixed at 42. Only `train` (CV, tuning) and `student_test` (comparison) are used. The evaluator partition is never read by these scripts.

## Running the demo

**Laptop 1 (API):** copy the evaluator trips into `<PATHSNAP_DATA>/evaluator_test/`, then:

```bash
set PATHSNAP_PARTITION=evaluator_test        # PowerShell: $env:PATHSNAP_PARTITION="evaluator_test"
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Every request is logged in this console: trip id, number of fixes, timing and accuracy. If the `evaluator_test` folder is missing, the API falls back to `student_test`.

Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`.

**Laptop 2 (UI):**

```bash
python ui/serve.py --port 8080
```

Open `http://localhost:8080/?api=http://<laptop-1-ip>:8000`. Both laptops must be on the same network, and Windows may ask you to allow port 8000 through the firewall on laptop 1. The UI lists the trips. Clicking one sends its id to the API and draws the raw GPS, the matched points and the predicted route, with the metrics alongside.
