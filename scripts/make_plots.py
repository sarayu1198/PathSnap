"""Step 6: figures for the report, generated only from files in results/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=140); plt.close()


def main():
    mc = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
    folds = pd.read_csv(os.path.join(RES, "cv_folds.csv"))
    order = mc.sort_values("cv_top1").model.tolist()

    # CV macro-F1 per model (box over folds)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([folds[folds.model == m].macro_f1 for m in order], tick_labels=order)
    ax.set_ylabel("Macro F1 (candidate level)"); ax.set_title("5-fold GroupKFold CV Macro F1 (train, 80 trips)")
    plt.xticks(rotation=25); save("cv_macro_f1.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([folds[folds.model == m].top1_acc for m in order], tick_labels=order)
    ax.set_ylabel("Observation-level Top-1 accuracy"); ax.set_title("5-fold GroupKFold CV Top-1 accuracy (train)")
    plt.xticks(rotation=25); save("cv_top1.png")

    # student-test Top-1 & macro-F1
    x = np.arange(len(order)); w = 0.38
    s = mc.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, s.test_top1, w, label="Top-1 accuracy"); ax.bar(x + w / 2, s.test_macro_f1, w, label="Macro F1")
    ax.set_xticks(x, order, rotation=25); ax.set_ylim(min(s.test_top1.min(), s.test_macro_f1.min()) - 0.03, 1)
    ax.set_title("Student-test (10 trips): Top-1 accuracy and Macro F1"); ax.legend()
    for i, v in enumerate(s.test_top1):
        ax.text(i - w / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)
    save("test_top1_macro_f1.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(order, s.test_log_loss, color="#c44e52"); ax.set_yscale("log")
    ax.set_title("Student-test log loss (lower is better)"); plt.xticks(rotation=25); save("test_log_loss.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(s.inference_ms_per_obs, s.test_top1)
    for m, r in s.iterrows():
        ax.annotate(m, (r.inference_ms_per_obs, r.test_top1), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("inference ms per GPS fix (log)"); ax.set_ylabel("student-test Top-1")
    ax.set_title("Accuracy vs inference cost"); save("accuracy_vs_speed.png")

    # confusion matrices
    log = json.load(open(os.path.join(RES, "experiment_log.json")))
    n = len(log); cols = 4; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows))
    for ax, (m, d) in zip(np.ravel(axs), log.items()):
        cm = np.array(d["confusion_matrix"]); cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(m, fontsize=9); ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
    for ax in np.ravel(axs)[n:]:
        ax.axis("off")
    save("confusion_matrices.png")

    p = os.path.join(RES, "mlp_loss_curve.csv")
    if os.path.exists(p):
        lc = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(7, 4)); ax2 = ax.twinx()
        ax.plot(lc.epoch, lc.train_loss, "o-", label="train log loss"); ax.set_ylabel("train loss")
        ax2.plot(lc.epoch, lc.val_accuracy, "s-", color="#c44e52", label="validation accuracy"); ax2.set_ylabel("val accuracy")
        ax.set_xlabel("epoch"); ax.set_title("MLP training history (only model with epoch-wise history)")
        fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.18)); save("mlp_training_curve.png")

    p = os.path.join(RES, "optuna_trials.csv")
    if os.path.exists(p):
        tr = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(tr.number, tr.value, "o", alpha=0.6, label="trial"); ax.plot(tr.number, tr.value.cummax(), "-", label="best so far")
        ax.set_xlabel("trial"); ax.set_ylabel("3-fold CV Top-1"); ax.set_title("Optuna TPE search history"); ax.legend()
        save("optuna_history.png")
    p = os.path.join(RES, "tuned_vs_untuned.csv")
    if os.path.exists(p):
        tu = pd.read_csv(p).set_index("model")
        cols_ = ["cv5_macro_f1", "cv5_top1", "test_top1", "test_macro_f1"]
        tu[cols_].T.plot.bar(figsize=(8, 4)); plt.ylim(tu[cols_].values.min() - 0.02, 1)
        plt.title("LightGBM untuned vs tuned"); plt.xticks(rotation=0); save("tuned_vs_untuned.png")
    p = os.path.join(RES, "feature_importance.csv")
    if os.path.exists(p):
        fi = pd.read_csv(p).sort_values("gain")
        plt.figure(figsize=(7, 6)); plt.barh(fi.feature, fi.gain); plt.title("Final model feature importance (gain)")
        save("feature_importance.png")
    p = os.path.join(RES, "hmm_student_test.csv")
    if os.path.exists(p):
        h = pd.read_csv(p)
        x = np.arange(len(h)); w = 0.38
        fig, ax = plt.subplots(2, 1, figsize=(11, 7))
        ax[0].bar(x - w / 2, h.top1_classifier, w, label="classifier only"); ax[0].bar(x + w / 2, h.top1_hmm, w, label="classifier + HMM")
        ax[0].set_xticks(x, h.trip, rotation=15, fontsize=8); ax[0].set_ylabel("Top-1 accuracy"); ax[0].legend()
        ax[0].set_ylim(min(h.top1_classifier.min(), h.top1_hmm.min()) - 0.05, 1); ax[0].set_title("Student-test: classifier vs HMM")
        ax[1].bar(x - w / 2, h.continuity_classifier, w, label="classifier only"); ax[1].bar(x + w / 2, h.continuity_hmm, w, label="classifier + HMM")
        ax[1].set_xticks(x, h.trip, rotation=15, fontsize=8); ax[1].set_ylabel("Route continuity"); ax[1].legend()
        ax[1].set_ylim(min(h.continuity_classifier.min(), h.continuity_hmm.min()) - 0.05, 1)
        save("hmm_vs_classifier.png")
    p = os.path.join(RES, "hmm_tuning.csv")
    if os.path.exists(p):
        ht = pd.read_csv(p)
        piv = ht.groupby(["w_e", "beta"]).train_oof_top1.max().unstack()
        plt.figure(figsize=(6, 4)); plt.imshow(piv.values, cmap="viridis", aspect="auto")
        plt.xticks(range(len(piv.columns)), piv.columns); plt.yticks(range(len(piv.index)), piv.index)
        plt.xlabel("beta (m)"); plt.ylabel("emission weight w_e"); plt.colorbar(label="train OOF Top-1")
        for (i, j), v in np.ndenumerate(piv.values):
            plt.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=8)
        plt.title("HMM decoder grid search (best over other params)"); save("hmm_tuning.png")


if __name__ == "__main__":
    main()
