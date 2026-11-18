"""End-to-end inference: raw trip -> candidates -> classifier -> HMM/Viterbi -> route.

Ground truth is never used here.  `score_trip` is a separate server-side step
that reads the reference route only after a prediction has been produced.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np

from .data import Trip
from .features import FEATURES, build_candidates
from .hmm import HMMParams, build_lattice, classifier_only, stitch_route, viterbi
from .labels import align_route
from .metrics import continuity, sequence_metrics
from .network import RoadNetwork

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")


class PathSnapModel:
    def __init__(self, model_path=None, hmm_path=None):
        model_path = model_path or os.path.join(ART, "final_model.joblib")
        hmm_path = hmm_path or os.path.join(ART, "hmm_params.json")
        self.clf = joblib.load(model_path)
        with open(hmm_path) as f:
            self.hmm = HMMParams(**json.load(f))

    def predict(self, trip: Trip):
        timings = {}
        t0 = time.perf_counter()
        net = RoadNetwork(trip)
        timings["network_index_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cand = build_candidates(trip, net)
        timings["features_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cand["p"] = self.clf.predict_proba(cand[FEATURES])[:, 1]
        timings["classifier_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        lat = build_lattice(cand, trip.t, trip.xy, net, gap_s=self.hmm.gap_s)
        pred_hmm, breaks = viterbi(lat, self.hmm)
        timings["hmm_s"] = time.perf_counter() - t0
        pred_clf = classifier_only(lat)
        route = stitch_route(pred_hmm, net)
        return dict(net=net, obs=lat.obs, pred_hmm=pred_hmm, pred_clf=pred_clf, route=route,
                    n_breaks=breaks, n_candidates=len(cand), timings=timings)


def score_trip(trip: Trip, net: RoadNetwork, out: dict, label_max_dist: float = 50.0) -> dict:
    """Server-side evaluation against the reference route (never sent to the model)."""
    ref_arc, _, ref_dist = align_route(trip.xy, trip.route, net, trip.t)
    obs = out["obs"]
    lab = ref_dist[obs] <= label_max_dist
    ref = ref_arc[obs]
    res = dict(
        n_obs=int(trip.n_obs), n_scored=int(lab.sum()),
        top1_acc_hmm=float((out["pred_hmm"] == ref)[lab].mean()),
        top1_acc_classifier=float((out["pred_clf"] == ref)[lab].mean()),
        continuity_hmm=continuity(out["pred_hmm"], trip.arcs),
        continuity_classifier=continuity(out["pred_clf"], trip.arcs),
    )
    res.update(sequence_metrics(out["route"], trip.route))
    return res
