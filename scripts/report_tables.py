"""Build every table the report needs from the files in results/ and artifacts/.

Run from the repo root:  python scripts/report_tables.py
Writes results/report_tables.md. Metrics use 3 decimals and times use 2,
so copy numbers from this file and they will match across the report.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, ART = ROOT / "results", ROOT / "artifacts"
OUT = RES / "report_tables.md"
MODEL_ORDER = ["LightGBM", "RandomForest", "HistGradientBoosting", "LogisticRegression",
               "MLP", "DecisionTree", "KNN", "GaussianNB"]


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def section_eda() -> str:
    inv = pd.read_csv(RES / "eda_inventory.csv", dtype={"trip": str})
    lab_tr = pd.read_csv(RES / "labels_summary_train.csv", dtype={"trip": str})
    lab_te = pd.read_csv(RES / "labels_summary_student_test.csv", dtype={"trip": str})
    rows = []
    for part, g in inv.groupby("partition", sort=False):
        rows.append({
            "Partition": part, "Trips": len(g), "GPS fixes": int(g.n_gps.sum()),
            "NaN values": int(g.n_nan.sum()), "Out-of-order timestamps": int(g.n_nonmonotonic_t.sum()),
            "Trips with a gap > 10 s": int((g.n_gaps_gt10s > 0).sum()),
            "Longest gap (s)": f"{g.max_gap_s.max():.0f}",
            "Jumps > 100 m": int(g.n_jumps_gt100m.sum()),
        })
    out = ["## 2.2 Dataset and EDA", "",
           "### Missing values and data quality (train and student_test)", "", md(pd.DataFrame(rows)), ""]
    longest = inv.loc[inv.max_gap_s.idxmax()]
    out += [f"Longest gap: trip {longest.trip} ({longest.partition}), {longest.max_gap_s:.0f} s "
            f"= {longest.max_gap_s/3600:.1f} h.", ""]

    rows = []
    for part, g in inv.groupby("partition", sort=False):
        rows.append({
            "Partition": part,
            "Median nodes": f"{g.n_nodes.median():.0f}", "Median arcs": f"{g.n_arcs.median():.0f}",
            "Median two-way arc share": f3(g.twoway_arc_frac.median()),
            "Median largest SCC share": f3(g.largest_scc_frac.median()),
            "Isolated nodes": int(g.isolated_nodes.sum()),
            "Median route connected share": f3(g.route_connected_frac.median()),
        })
    out += ["### Road-network connectivity", "", md(pd.DataFrame(rows)), ""]

    t1 = inv[(inv.partition == "train") & (inv.trip == "00000001")].iloc[0]
    out += ["### Trip 00000001 (train), the trip in `figures/eda_trajectory_00000001.png`", "",
            md(pd.DataFrame([{
                "GPS fixes": int(t1.n_gps), "Nodes": int(t1.n_nodes), "Arcs": int(t1.n_arcs),
                "Route arcs": int(t1.n_route_arcs), "Path length (km)": f"{t1.path_len_km:.1f}",
                "Weak components": int(t1.weak_components), "Strong components": int(t1.strong_components),
                "Largest SCC share": f3(t1.largest_scc_frac), "Two-way arc share": f3(t1.twoway_arc_frac),
                "Median GPS-to-route distance (m)": f"{t1.ref_dist_median_m:.1f}",
            }])), ""]

    rows = []
    for part, lab in (("train", lab_tr), ("student_test", lab_te)):
        rows.append({
            "Partition": part, "GPS fixes": int(lab.n_obs.sum()), "Labelled fixes": int(lab.n_labelled.sum()),
            "Unlabelled share (> 50 m from route)": f3(1 - lab.n_labelled.sum() / lab.n_obs.sum()),
            "Candidate miss rate": f3((lab.cand_miss_rate * lab.n_labelled).sum() / lab.n_labelled.sum()),
            "Candidate rows": int(lab.n_rows.sum()),
        })
    out += ["### Labels and candidates", "", md(pd.DataFrame(rows)), ""]

    fi = pd.read_csv(RES / "feature_importance.csv")
    fi["Rank"] = range(1, len(fi) + 1)
    fi["Gain share"] = (fi.gain / fi.gain.sum()).map(f3)
    out += [f"### Feature importance (tuned LightGBM, {len(fi)} features)", "",
            md(fi[["Rank", "feature", "Gain share"]].rename(columns={"feature": "Feature"})), ""]
    return "\n".join(out)


def section_models() -> str:
    mc = pd.read_csv(RES / "model_comparison.csv").set_index("model").loc[MODEL_ORDER].reset_index()
    t = pd.DataFrame({
        "Model": mc.model, "Family": mc.family, "Key configuration": mc.config,
        "CV Macro F1": [f"{a:.3f} ± {b:.3f}" for a, b in zip(mc.cv_macro_f1, mc.cv_macro_f1_std)],
        "CV Top-1": mc.cv_top1.map(f3), "CV continuity": mc.cv_continuity.map(f3),
        "Test Top-1": mc.test_top1.map(f3), "Test continuity": mc.test_continuity.map(f3),
        "Test Macro F1": mc.test_macro_f1.map(f3), "Test log loss": mc.test_log_loss.map(f3),
        "Train time (s)": mc.train_time_s.map(f2), "Inference time, 10 trips (s)": mc.inference_time_s.map(f2),
    })
    out = ["## 3. Model comparison", "",
           "### 3.1 All 8 models (CV = 5-fold GroupKFold on train; test = student_test)", "", md(t), ""]

    s = pd.read_csv(RES / "summary_train_cv_test.csv").set_index("model")
    tf = pd.read_csv(RES / "train_fit_metrics.csv").set_index("model")
    rows = []
    for m in MODEL_ORDER + ["LightGBM_tuned"]:
        r = s.loc[m]
        cv_ll = mc.set_index("model").cv_log_loss.get(m)
        rows.append({
            "Model": m, "Train Top-1": f3(r.train_top1), "CV Top-1": f3(r.cv_top1),
            "Test Top-1": f3(r.test_top1), "Train − CV Top-1": f3(r.train_top1 - r.cv_top1),
            "Train Macro F1": f3(r.train_macro_f1), "CV Macro F1": f3(r.cv_macro_f1),
            "Train − CV Macro F1": f3(r.train_macro_f1 - r.cv_macro_f1),
            "Train log loss": f3(tf.loc[m].train_log_loss) if m in tf.index else "n/a",
            "CV log loss": f3(cv_ll) if cv_ll is not None else "n/a",
        })
    out += ["### 3.2 Over/underfitting evidence (train vs CV gap)", "", md(pd.DataFrame(rows)), "",
            "A large positive train − CV gap means the model fits the training trips better than unseen trips "
            "(overfitting). Near-zero gap with a low score means the model is too simple (underfitting).", ""]

    log = json.loads((RES / "experiment_log.json").read_text())
    rows = []
    for m in MODEL_ORDER:
        cm = log[m].get("confusion_matrix")
        if cm:
            (tn, fp), (fn, tp) = cm
            rows.append({"Model": m, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    out += ["### Confusion matrices on student_test (candidate level: is this arc the right one)", "",
            md(pd.DataFrame(rows)), ""]
    return "\n".join(out)


def section_bayes() -> str:
    src = (ROOT / "scripts" / "bayes_opt.py").read_text()
    rows = []
    for kind, name, lo, hi, rest in re.findall(
            r'suggest_(int|float)\("(\w+)",\s*([\d.e-]+),\s*([\d.e-]+)([^)]*)\)', src):
        rows.append({"Hyperparameter": name, "Type": kind, "Low": lo, "High": hi,
                     "Scale": "log" if "log=True" in rest else "linear",
                     "Step": (re.search(r"step=(\d+)", rest) or [None, ""])[1]})
    space = pd.DataFrame(rows)
    bp = json.loads((ART / "best_params.json").read_text())
    space["Best value"] = [
        (f"{bp['best_params'][h]:.4g}" if isinstance(bp["best_params"][h], float) else str(bp["best_params"][h]))
        for h in space.Hyperparameter]
    tr = pd.read_csv(RES / "optuna_trials.csv")
    tv = pd.read_csv(RES / "tuned_vs_untuned.csv")
    tvt = pd.DataFrame({
        "Model": tv.model, "CV5 Top-1": tv.cv5_top1.map(f3), "CV5 Macro F1": tv.cv5_macro_f1.map(f3),
        "Test Top-1": tv.test_top1.map(f3), "Test continuity": tv.test_continuity.map(f3),
        "Test Macro F1": tv.test_macro_f1.map(f3), "Test log loss": tv.test_log_loss.map(f3),
        "Train time (s)": tv.train_time_s.map(f2), "Inference time, 10 trips (s)": tv.inference_time_s.map(f2),
    })
    out = ["## 4. Bayesian optimisation", "",
           f"Sampler: Optuna {bp['sampler']}, seed {bp['seed']}. Trials: {bp['n_trials']}. "
           f"Objective: mean 3-fold GroupKFold Top-1 on train. "
           f"Best trial: {int(tr.loc[tr.value.idxmax(), 'number'])} with CV3 Top-1 {f3(bp['best_cv3_top1'])}. "
           f"Search time: {bp['search_seconds']:.0f} s ({bp['search_seconds']/60:.1f} min).", "",
           "### Search space and best values", "", md(space), "",
           "### Tuned vs untuned LightGBM", "", md(tvt), ""]
    return "\n".join(out)


def section_hmm() -> str:
    h = pd.read_csv(RES / "hmm_student_test.csv", dtype={"trip": str})
    t = pd.DataFrame({
        "Trip": h.trip, "Fixes scored": h.n_scored,
        "Top-1 classifier": h.top1_classifier.map(f3), "Top-1 + HMM": h.top1_hmm.map(f3),
        "Top-1 gain": (h.top1_hmm - h.top1_classifier).map(f3),
        "Continuity classifier": h.continuity_classifier.map(f3), "Continuity + HMM": h.continuity_hmm.map(f3),
        "Route F1 classifier": h.route_f1_classifier.map(f3), "Route F1 + HMM": h.route_f1_hmm.map(f3),
        "Chain breaks": h.hmm_breaks, "Viterbi time (s)": h.viterbi_s.map(f2),
    })
    hp = json.loads((ART / "hmm_params.json").read_text())
    tun = pd.read_csv(RES / "hmm_tuning.csv").sort_values("train_oof_top1", ascending=False)
    grid = {c: sorted(tun[c].unique()) for c in ["w_e", "beta", "uturn_pen", "back_pen"]}
    top = tun.head(5).copy()
    top["train_oof_top1"] = top.train_oof_top1.map(f3)
    out = ["## 5. HMM and Viterbi (student_test)", "", md(t), "",
           "Chosen decoder parameters (`artifacts/hmm_params.json`): "
           + ", ".join(f"{k} = {v:g}" for k, v in hp.items()) + ".", "",
           f"Grid searched ({len(tun)} combinations, scored on out-of-fold train probabilities): "
           + "; ".join(f"{k} in {[float(x) for x in v]}" for k, v in grid.items()) + ".", "",
           "### Best 5 grid points", "", md(top), ""]
    return "\n".join(out)


def main():
    parts = ["# Report tables", "",
             "Built by `scripts/report_tables.py` from `results/` and `artifacts/`. "
             "Metrics have 3 decimals and times 2, everywhere.", "",
             section_eda(), section_models(), section_bayes(), section_hmm()]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
