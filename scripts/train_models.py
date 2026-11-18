"""Step 3: train and compare candidate classifiers.

- 5-fold GroupKFold on the 80 training trips (groups = trip) -> CV metrics
- refit on all training trips -> student_test metrics
Outputs: results/model_comparison.csv, results/cv_folds.csv, results/experiment_log.json,
         results/mlp_loss_curve.csv, artifacts/models/<name>.joblib, artifacts/oof_<name>.pkl
"""
import json
import os
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import candidate_metrics, route_level, top1_predictions  # noqa: E402

SEED = 42
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
MOD = os.path.join(ROOT, "artifacts", "models")
os.makedirs(MOD, exist_ok=True)


def models():
    """name -> (family, estimator factory, max training rows (None = all), config description)"""
    return {
        "LogisticRegression": ("Linear", lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=500, random_state=SEED)), None,
            "StandardScaler, L2, C=1.0, lbfgs"),
        "GaussianNB": ("Probabilistic", lambda: GaussianNB(), None, "default var_smoothing=1e-9"),
        "KNN": ("Instance-based", lambda: make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)), 150_000,
            "StandardScaler, k=25, distance-weighted, 150k-row subsample"),
        "DecisionTree": ("Tree-based", lambda: DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=50, random_state=SEED), None, "max_depth=12, min_samples_leaf=50"),
        "RandomForest": ("Ensemble (bagging)", lambda: RandomForestClassifier(
            n_estimators=150, max_depth=18, min_samples_leaf=5, max_samples=0.3, n_jobs=-1,
            random_state=SEED), None, "150 trees, max_depth=18, leaf=5, max_samples=0.3"),
        "HistGradientBoosting": ("Boosting", lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED), None,
            "300 iters, lr=0.1, 31 leaves"),
        "LightGBM": ("Boosting", lambda: lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEED, verbose=-1), None,
            "400 trees, lr=0.05, 63 leaves, bagging 0.8, colsample 0.8"),
        "MLP": ("Neural network", lambda: make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                                            learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                                            validation_fraction=0.1, n_iter_no_change=8,
                                            random_state=SEED)), 600_000,
            "StandardScaler, (64,32) ReLU, Adam lr=1e-3, early stopping, 600k-row subsample"),
    }


def subsample(X, y, n, rng):
    if n is None or len(X) <= n:
        return X, y
    idx = rng.choice(len(X), n, replace=False)
    return X.iloc[idx], y.iloc[idx]


def evaluate(df, p, partition):
    df = df.assign(p=p)
    lab = df[df.labelled]
    m = candidate_metrics(lab.y.values, lab.p.values)
    m.update(route_level(top1_predictions(df), partition))
    return m


def main(only=None):
    tr = pd.read_pickle(os.path.join(ROOT, "artifacts", "features_train.pkl"))
    te = pd.read_pickle(os.path.join(ROOT, "artifacts", "features_student_test.pkl"))
    trl = tr[tr.labelled].reset_index(drop=True)
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(trl, trl.y, trl.trip))
    rng = np.random.default_rng(SEED)

    rows, fold_rows, log = [], [], {}
    for name, (family, make, nmax, cfg) in models().items():
        if only and name not in only:
            continue
        print(f"=== {name}", flush=True)
        oof = np.zeros(len(trl))
        for k, (a, b) in enumerate(folds):
            Xa, ya = subsample(trl.iloc[a][FEATURES], trl.iloc[a].y, nmax, rng)
            est = make().fit(Xa, ya)
            oof[b] = est.predict_proba(trl.iloc[b][FEATURES])[:, 1]
            fm = evaluate(trl.iloc[b], oof[b], "train")
            fold_rows.append(dict(model=name, fold=k, **fm))
            print(f"  fold {k}: top1={fm['top1_acc']:.4f} macroF1={fm['macro_f1']:.4f}", flush=True)
        trl[["trip", "obs", "arc", "y", "labelled"]].assign(p=oof).to_pickle(
            os.path.join(ROOT, "artifacts", f"oof_{name}.pkl"))
        fdf = pd.DataFrame([r for r in fold_rows if r["model"] == name])

        Xa, ya = subsample(trl[FEATURES], trl.y, nmax, rng)
        t0 = time.time(); est = make().fit(Xa, ya); train_time = time.time() - t0
        t0 = time.time(); pte = est.predict_proba(te[FEATURES])[:, 1]; inf_time = time.time() - t0
        tm = evaluate(te, pte, "student_test")
        lab = te.labelled.values
        cm = confusion_matrix(te.y.values[lab], (pte[lab] >= 0.5).astype(int)).tolist()
        joblib.dump(est, os.path.join(MOD, f"{name}.joblib"))
        te[["trip", "obs", "arc", "y", "labelled"]].assign(p=pte).to_pickle(
            os.path.join(ROOT, "artifacts", f"test_pred_{name}.pkl"))
        if name == "MLP":
            mlp = est[-1]
            pd.DataFrame({"epoch": np.arange(1, len(mlp.loss_curve_) + 1), "train_loss": mlp.loss_curve_,
                          "val_accuracy": mlp.validation_scores_}).to_csv(
                os.path.join(RES, "mlp_loss_curve.csv"), index=False)

        row = dict(model=name, family=family, config=cfg,
                   cv_macro_f1=fdf.macro_f1.mean(), cv_macro_f1_std=fdf.macro_f1.std(),
                   cv_top1=fdf.top1_acc.mean(), cv_top1_std=fdf.top1_acc.std(),
                   cv_log_loss=fdf.log_loss.mean(), cv_continuity=fdf.route_continuity.mean(),
                   test_accuracy=tm["accuracy"], test_precision=tm["precision"], test_recall=tm["recall"],
                   test_f1=tm["f1"], test_macro_f1=tm["macro_f1"], test_log_loss=tm["log_loss"],
                   test_top1=tm["top1_acc"], test_continuity=tm["route_continuity"],
                   train_rows=len(Xa), train_time_s=train_time, inference_time_s=inf_time,
                   inference_ms_per_obs=1000 * inf_time / te.groupby(["trip", "obs"]).ngroups)
        rows.append(row)
        log[name] = dict(row, confusion_matrix=cm, seed=SEED, params=str(est.get_params()))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    suffix = "" if not only else "_" + "_".join(only)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"model_comparison{suffix}.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(RES, f"cv_folds{suffix}.csv"), index=False)
    with open(os.path.join(RES, f"experiment_log{suffix}.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
