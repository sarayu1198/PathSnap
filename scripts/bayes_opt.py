"""Step 4: Bayesian hyperparameter optimisation (Optuna TPE) of the selected model (LightGBM).

Objective: maximise mean observation-level Top-1 accuracy over a 3-fold
GroupKFold (groups = trip) on the 80 training trips.  Student-test data is only
used after the search, to compare tuned vs untuned.  Evaluator data is never used.

usage: python scripts/bayes_opt.py [n_trials]
Outputs: results/optuna_trials.csv, results/tuned_vs_untuned.csv, artifacts/best_params.json,
         artifacts/final_model.joblib, artifacts/oof_LightGBM_tuned.pkl, artifacts/test_pred_LightGBM_tuned.pkl
"""
import json
import os
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import candidate_metrics, route_level, top1_predictions  # noqa: E402

SEED = 42
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, ART = os.path.join(ROOT, "results"), os.path.join(ROOT, "artifacts")

UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8)


def make(params):
    return lgb.LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1, **params)


def top1(df, p):
    return route_level(top1_predictions(df.assign(p=p)), "train")["top1_acc"]


def main(n_trials=40):
    tr = pd.read_pickle(os.path.join(ART, "features_train.pkl"))
    te = pd.read_pickle(os.path.join(ART, "features_student_test.pkl"))
    trl = tr[tr.labelled].reset_index(drop=True)
    X, y = trl[FEATURES], trl.y
    folds3 = list(GroupKFold(n_splits=3).split(X, y, trl.trip))

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 500, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
        scores, lls = [], []
        t0 = time.time()
        for a, b in folds3:
            m = make(params).fit(X.iloc[a], y.iloc[a])
            p = m.predict_proba(X.iloc[b])[:, 1]
            scores.append(top1(trl.iloc[b], p))
            lls.append(candidate_metrics(y.iloc[b].values, p)["log_loss"])
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(scores)))
        trial.set_user_attr("seconds", time.time() - t0)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="lightgbm_top1")
    study.enqueue_trial({k: v for k, v in UNTUNED.items() if k != "subsample_freq"})  # trial 0 = untuned baseline
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number}: {t.value:.4f} {t.params}", flush=True)])
    search_s = time.time() - t0
    study.trials_dataframe().to_csv(os.path.join(RES, "optuna_trials.csv"), index=False)
    best = dict(study.best_params, subsample_freq=1)
    with open(os.path.join(ART, "best_params.json"), "w") as f:
        json.dump(dict(best_params=best, best_cv3_top1=study.best_value, n_trials=n_trials,
                       search_seconds=search_s, sampler="TPE", seed=SEED), f, indent=2)

    # ---- tuned vs untuned with identical protocol (5-fold GroupKFold CV + student_test)
    folds5 = list(GroupKFold(n_splits=5).split(X, y, trl.trip))
    rows = []
    for name, params in [("LightGBM_untuned", UNTUNED), ("LightGBM_tuned", best)]:
        oof = np.zeros(len(trl)); f1s, t1s = [], []
        for a, b in folds5:
            m = make(params).fit(X.iloc[a], y.iloc[a])
            oof[b] = m.predict_proba(X.iloc[b])[:, 1]
            f1s.append(candidate_metrics(y.iloc[b].values, oof[b])["macro_f1"])
            t1s.append(top1(trl.iloc[b], oof[b]))
        t0 = time.time(); m = make(params).fit(X, y); fit_s = time.time() - t0
        t0 = time.time(); pte = m.predict_proba(te[FEATURES])[:, 1]; inf_s = time.time() - t0
        lab = te.labelled.values
        cm = candidate_metrics(te.y.values[lab], pte[lab])
        rl = route_level(top1_predictions(te.assign(p=pte)), "student_test")
        rows.append(dict(model=name, cv5_macro_f1=np.mean(f1s), cv5_top1=np.mean(t1s), cv5_top1_std=np.std(t1s),
                         test_top1=rl["top1_acc"], test_continuity=rl["route_continuity"],
                         test_macro_f1=cm["macro_f1"], test_log_loss=cm["log_loss"],
                         train_time_s=fit_s, inference_time_s=inf_s))
        print(rows[-1], flush=True)
        if name == "LightGBM_tuned":
            joblib.dump(m, os.path.join(ART, "final_model.joblib"))
            trl[["trip", "obs", "arc", "y", "labelled"]].assign(p=oof).to_pickle(os.path.join(ART, "oof_LightGBM_tuned.pkl"))
            te[["trip", "obs", "arc", "y", "labelled"]].assign(p=pte).to_pickle(os.path.join(ART, "test_pred_LightGBM_tuned.pkl"))
            imp = pd.DataFrame({"feature": FEATURES, "gain": m.booster_.feature_importance("gain")})
            imp.sort_values("gain", ascending=False).to_csv(os.path.join(RES, "feature_importance.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "tuned_vs_untuned.csv"), index=False)
    print("search time %.0fs" % search_s)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
