"""Epoch-wise training/validation loss for the MLP (the only model with an epoch history).

Same architecture as in train_models.py.  Validation = 16 held-out training trips
(GroupShuffleSplit by trip), so the curve shows generalisation to unseen trips.
Output: results/mlp_epoch_history.csv, figures/mlp_train_val_loss.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.features import FEATURES  # noqa: E402

SEED, EPOCHS, N_TRAIN = 42, 40, 600_000
ROOT = os.path.join(os.path.dirname(__file__), "..")

tr = pd.read_pickle(os.path.join(ROOT, "artifacts", "features_train.pkl"))
tr = tr[tr.labelled].reset_index(drop=True)
a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(tr, tr.y, tr.trip))
rng = np.random.default_rng(SEED)
a = rng.choice(a, min(N_TRAIN, len(a)), replace=False)
sc = StandardScaler().fit(tr.loc[a, FEATURES])
Xa, ya = sc.transform(tr.loc[a, FEATURES]), tr.y.values[a]
Xb, yb = sc.transform(tr.loc[b, FEATURES]), tr.y.values[b]

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                    random_state=SEED)
rows = []
for ep in range(1, EPOCHS + 1):
    mlp.partial_fit(Xa, ya, classes=[0, 1])
    rows.append(dict(epoch=ep,
                     train_log_loss=log_loss(ya, mlp.predict_proba(Xa)[:, 1], labels=[0, 1]),
                     val_log_loss=log_loss(yb, mlp.predict_proba(Xb)[:, 1], labels=[0, 1])))
    print(rows[-1], flush=True)
h = pd.DataFrame(rows)
h.to_csv(os.path.join(ROOT, "results", "mlp_epoch_history.csv"), index=False)
plt.figure(figsize=(7, 4))
plt.plot(h.epoch, h.train_log_loss, "o-", ms=3, label=f"train ({len(a):,} rows)")
plt.plot(h.epoch, h.val_log_loss, "s-", ms=3, label=f"validation (16 held-out trips)")
best = h.val_log_loss.idxmin()
plt.axvline(h.epoch[best], color="grey", ls="--", lw=1, label=f"min val loss @ epoch {h.epoch[best]}")
plt.xlabel("epoch"); plt.ylabel("log loss"); plt.title("MLP (64,32): training vs validation loss")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "mlp_train_val_loss.png"), dpi=140)
