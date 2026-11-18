"""Step 5: tune HMM decoder parameters and compare classifier-only vs classifier+HMM.

Decoder parameters are tuned on OUT-OF-FOLD classifier probabilities for the
training trips (so emissions are as noisy as they will be on unseen trips),
then evaluated on the student_test trips.

usage: python scripts/hmm_eval.py <model_name>   (uses artifacts/oof_<name>.pkl and test_pred_<name>.pkl)
Outputs: artifacts/hmm_params.json, results/hmm_tuning.csv, results/hmm_student_test.csv,
         maps/pred_student_test_<trip>.html
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.hmm import HMMParams, build_lattice, classifier_only, stitch_route, viterbi  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.metrics import continuity, sequence_metrics  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, ART, MAPS = (os.path.join(ROOT, d) for d in ("results", "artifacts", "maps"))
N_TUNE_TRIPS = 20
SEED = 42


def prepare(partition, probs: pd.DataFrame, trips):
    feats = pd.read_pickle(os.path.join(ART, f"features_{partition}.pkl"))
    feats = feats[feats.trip.isin(trips)]
    df = feats.merge(probs[["trip", "obs", "arc", "p"]], on=["trip", "obs", "arc"], how="inner")
    out = {}
    for tid, g in df.groupby("trip"):
        trip = load_trip(tid, partition)
        net = RoadNetwork(trip)
        ref_arc, _, ref_dist = align_route(trip.xy, trip.route, net, trip.t)
        lat = build_lattice(g, trip.t, trip.xy, net)
        out[tid] = dict(trip=trip, net=net, lat=lat, ref=ref_arc[lat.obs], lab=ref_dist[lat.obs] <= 50.0)
    return out


def score(d, pred):
    lab = d["lab"]
    return (pred == d["ref"])[lab].sum(), lab.sum()


def main(model):
    rng = np.random.default_rng(SEED)
    oof = pd.read_pickle(os.path.join(ART, f"oof_{model}.pkl"))
    tune_trips = sorted(rng.choice(list_trips("train"), N_TUNE_TRIPS, replace=False).tolist())
    t0 = time.time()
    tune = prepare("train", oof, tune_trips)
    print(f"lattices for {len(tune)} train trips built in {time.time() - t0:.0f}s", flush=True)

    grid = dict(w_e=[0.5, 1.0, 2.0], beta=[3.0, 8.0, 20.0], uturn_pen=[2.0, 6.0, 15.0, 30.0], back_pen=[0.0, 2.0])
    rows = []
    base_c = sum(score(d, classifier_only(d["lat"]))[0] for d in tune.values())
    base_n = sum(d["lab"].sum() for d in tune.values())
    for vals in itertools.product(*grid.values()):
        prm = HMMParams(**dict(zip(grid.keys(), vals)))
        c = n = 0
        for d in tune.values():
            pred, _ = viterbi(d["lat"], prm)
            a, b = score(d, pred); c += a; n += b
        rows.append(dict(prm.to_dict(), train_oof_top1=c / n))
        print(rows[-1], flush=True)
    tun = pd.DataFrame(rows).sort_values("train_oof_top1", ascending=False)
    tun.to_csv(os.path.join(RES, "hmm_tuning.csv"), index=False)
    best = {k: tun.iloc[0][k] for k in HMMParams().to_dict()}
    best = HMMParams(**{k: float(v) for k, v in best.items()})
    with open(os.path.join(ART, "hmm_params.json"), "w") as f:
        json.dump(best.to_dict(), f, indent=2)
    print("classifier-only OOF top1 on tuning trips:", base_c / base_n, "best HMM:", tun.iloc[0].to_dict())

    # ---------------- student_test comparison
    tp = pd.read_pickle(os.path.join(ART, f"test_pred_{model}.pkl"))
    test = prepare("student_test", tp, list_trips("student_test"))
    out = []
    for tid, d in test.items():
        t0 = time.time()
        ph, br = viterbi(d["lat"], best)
        dec_t = time.time() - t0
        pc = classifier_only(d["lat"])
        route_h = stitch_route(ph, d["net"])
        route_c = stitch_route(pc, d["net"])
        ch, nh = score(d, ph); cc, _ = score(d, pc)
        r = dict(trip=tid, n_obs=len(d["ref"]), n_scored=int(nh),
                 top1_classifier=cc / nh, top1_hmm=ch / nh,
                 continuity_classifier=continuity(pc, d["trip"].arcs), continuity_hmm=continuity(ph, d["trip"].arcs),
                 hmm_breaks=br, viterbi_s=dec_t)
        r.update({k + "_classifier": v for k, v in sequence_metrics(route_c, d["trip"].route).items()})
        r.update({k + "_hmm": v for k, v in sequence_metrics(route_h, d["trip"].route).items()})
        # whole-trip sequence accuracy: fraction of trips where every scored fix is correct is ~0, so we
        # report the longest run of consecutive correct fixes as a sequence-level indicator
        corr = (ph == d["ref"]) | ~d["lab"]
        runs = np.diff(np.r_[0, np.flatnonzero(~corr) , len(corr)]) - 1
        r["longest_correct_run_hmm"] = int(runs.max()) if len(runs) else len(corr)
        out.append(r)
        trip_map(d["trip"], d["net"], pred_arcs=route_h, ref_route=d["trip"].route,
                 title=f"student_test {tid}: HMM top-1 {ch / nh:.3f}").save(
            os.path.join(MAPS, f"pred_student_test_{tid}.html"))
        print(r, flush=True)
    res = pd.DataFrame(out)
    w = res.n_scored / res.n_scored.sum()
    tot = {c: (res[c] * w).sum() for c in res.columns if c not in ("trip", "n_obs", "n_scored", "hmm_breaks")}
    tot.update(trip="ALL (weighted by fixes)", n_obs=res.n_obs.sum(), n_scored=res.n_scored.sum(), hmm_breaks=res.hmm_breaks.sum())
    res = pd.concat([res, pd.DataFrame([tot])], ignore_index=True)
    res.to_csv(os.path.join(RES, f"hmm_student_test.csv"), index=False)
    print(res.tail(1).T)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "LightGBM")
