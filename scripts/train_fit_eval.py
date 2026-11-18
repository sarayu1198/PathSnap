"""In-sample (training-set) metrics of every saved model, for over/underfitting analysis.

Compare with the CV (validation) numbers in results/model_comparison.csv: a large
train-minus-CV gap indicates overfitting.
Output: results/train_fit_metrics.csv
"""
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import candidate_metrics, route_level, top1_predictions  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = ["LogisticRegression", "GaussianNB", "KNN", "DecisionTree", "RandomForest",
          "HistGradientBoosting", "LightGBM", "MLP"]

tr = pd.read_pickle(os.path.join(ROOT, "artifacts", "features_train.pkl"))
tr = tr[tr.labelled].reset_index(drop=True)
rows = []
for name in MODELS + ["final_model"]:
    path = os.path.join(ROOT, "artifacts", "models", f"{name}.joblib") if name != "final_model" \
        else os.path.join(ROOT, "artifacts", "final_model.joblib")
    if not os.path.exists(path):
        print("skip", name)
        continue
    p = joblib.load(path).predict_proba(tr[FEATURES])[:, 1]
    m = candidate_metrics(tr.y.values, p)
    m.update(route_level(top1_predictions(tr.assign(p=p)), "train"))
    rows.append(dict(model="LightGBM_tuned" if name == "final_model" else name,
                     train_macro_f1=m["macro_f1"], train_top1=m["top1_acc"], train_log_loss=m["log_loss"]))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "train_fit_metrics.csv"), index=False)
