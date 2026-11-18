"""Combine train / CV / test results into one table (for the report).

Reads results/train_fit_metrics.csv, model_comparison.csv, tuned_vs_untuned.csv,
hmm_student_test.csv. Writes results/summary_train_cv_test.csv and .md
"""
import os

import pandas as pd

RES = os.path.join(os.path.dirname(__file__), "..", "results")

tr = pd.read_csv(os.path.join(RES, "train_fit_metrics.csv")).set_index("model")
mc = pd.read_csv(os.path.join(RES, "model_comparison.csv")).set_index("model")
tu = pd.read_csv(os.path.join(RES, "tuned_vs_untuned.csv")).set_index("model")
hmm = pd.read_csv(os.path.join(RES, "hmm_student_test.csv")).iloc[-1]

rows = []
for m in mc.index:
    rows.append(dict(model=m, train_top1=tr.loc[m, "train_top1"], cv_top1=mc.loc[m, "cv_top1"],
                     test_top1=mc.loc[m, "test_top1"], train_macro_f1=tr.loc[m, "train_macro_f1"],
                     cv_macro_f1=mc.loc[m, "cv_macro_f1"], test_macro_f1=mc.loc[m, "test_macro_f1"]))
t = tu.loc["LightGBM_tuned"]
rows.append(dict(model="LightGBM_tuned", train_top1=tr.loc["LightGBM_tuned", "train_top1"], cv_top1=t.cv5_top1,
                 test_top1=t.test_top1, train_macro_f1=tr.loc["LightGBM_tuned", "train_macro_f1"],
                 cv_macro_f1=t.cv5_macro_f1, test_macro_f1=t.test_macro_f1))
rows.append(dict(model="LightGBM_tuned + HMM (final)", test_top1=hmm.top1_hmm))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "summary_train_cv_test.csv"), index=False)


def md(cols, title):
    out = [f"### {title}", "", "| Model | Train | CV | Test |", "|---|---|---|---|"]
    for _, r in df.iterrows():
        cells = ["n/a" if pd.isna(r[c]) else f"{r[c]:.3f}" for c in cols]
        out.append(f"| {r.model} | " + " | ".join(cells) + " |")
    return "\n".join(out)


with open(os.path.join(RES, "summary_train_cv_test.md"), "w") as f:
    f.write("# Train / CV / test summary\n\n"
            "Train = model scored on its own training data (80 train trips). "
            "CV = 5-fold GroupKFold by trip on the train trips. Test = 10 student-test trips.\n\n")
    f.write(md(["train_top1", "cv_top1", "test_top1"], "Top-1 accuracy") + "\n\n")
    f.write(md(["train_macro_f1", "cv_macro_f1", "test_macro_f1"], "Macro F1 (candidate level)") + "\n")
print(open(os.path.join(RES, "summary_train_cv_test.md")).read())
