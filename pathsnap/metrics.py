"""Evaluation metrics: candidate-level classification + observation/route level."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from .data import partition_dir


@lru_cache(maxsize=256)
def _arcs(trip: str, partition: str) -> np.ndarray:
    return np.loadtxt(os.path.join(partition_dir(partition), trip, trip + ".arcs"), dtype=np.int64, ndmin=2)


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(
        accuracy=accuracy_score(y, yhat),
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        f1=f1_score(y, yhat, zero_division=0),
        macro_f1=f1_score(y, yhat, average="macro", zero_division=0),
        log_loss=log_loss(y, pc, labels=[0, 1]),
    )


def top1_predictions(df: pd.DataFrame, score_col="p") -> pd.DataFrame:
    """Highest-scoring candidate per (trip, obs)."""
    idx = df.groupby(["trip", "obs"], sort=True)[score_col].idxmax()
    return df.loc[idx, ["trip", "obs", "arc", "y", "labelled"]].reset_index(drop=True)


def continuity(arcs_seq: np.ndarray, arcs_table: np.ndarray) -> float:
    """Fraction of consecutive predictions that are the same arc or directly connected."""
    if len(arcs_seq) < 2:
        return 1.0
    a, b = arcs_seq[:-1], arcs_seq[1:]
    ok = (a == b) | (arcs_table[a, 1] == arcs_table[b, 0])
    return float(ok.mean())


def route_level(pred: pd.DataFrame, partition: str, pred_col="arc") -> dict:
    """pred: one row per (trip, obs) with columns trip, obs, <pred_col>, ref_arc?, y, labelled.
    Returns top-1 accuracy (labelled obs), route continuity, and per-trip sequence metrics."""
    lab = pred[pred.labelled]
    acc = float(lab["y"].mean())
    conts, seq_exact = [], []
    for trip, g in pred.groupby("trip", sort=False):
        g = g.sort_values("obs")
        conts.append(continuity(g[pred_col].values, _arcs(trip, partition)))
    return dict(top1_acc=acc, route_continuity=float(np.mean(conts)))


def sequence_metrics(pred_arcs: np.ndarray, ref_route: np.ndarray) -> dict:
    """Route-level overlap between predicted traversed arcs and the reference route."""
    P, R = set(pred_arcs.tolist()), set(ref_route.tolist())
    inter = len(P & R)
    prec = inter / max(len(P), 1)
    rec = inter / max(len(R), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(route_precision=prec, route_recall=rec, route_f1=f1)
