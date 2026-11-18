"""Step 2: build candidate/feature tables for the train and student_test partitions.

Output: artifacts/features_<partition>.pkl  (one row per (trip, obs, candidate arc))
        results/labels_summary_<partition>.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.features import build_candidates  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402

LABEL_MAX_DIST = 50.0  # fixes further than this from their aligned reference arc are unlabelled
ROOT = os.path.join(os.path.dirname(__file__), "..")


def process(partition):
    rows, summ = [], []
    for tid in list_trips(partition):
        t0 = time.time()
        trip = load_trip(tid, partition)
        net = RoadNetwork(trip)
        ref_arc, pos, ref_dist = align_route(trip.xy, trip.route, net, trip.t)
        labelled = ref_dist <= LABEL_MAX_DIST
        df = build_candidates(trip, net)
        df.insert(0, "trip", tid)
        df["y"] = (df["arc"].values == ref_arc[df["obs"].values]).astype(np.int8)
        df["labelled"] = labelled[df["obs"].values]
        rows.append(df)
        hit = np.zeros(trip.n_obs, bool)
        hit[df.loc[df.y == 1, "obs"].values] = True
        summ.append(dict(trip=tid, n_obs=trip.n_obs, n_labelled=int(labelled.sum()),
                         n_rows=len(df), cand_miss_rate=float(1 - hit[labelled].mean()),
                         ref_dist_median=float(np.median(ref_dist)),
                         ref_dist_p95=float(np.percentile(ref_dist, 95)),
                         route_len=len(trip.route),
                         route_arcs_covered=float(len(np.unique(pos)) / len(trip.route)),
                         seconds=time.time() - t0))
        print(partition, summ[-1])
    out = pd.concat(rows, ignore_index=True)
    out.to_pickle(os.path.join(ROOT, "artifacts", f"features_{partition}.pkl"))
    pd.DataFrame(summ).to_csv(os.path.join(ROOT, "results", f"labels_summary_{partition}.csv"), index=False)
    lab = out[out.labelled]
    print(partition, "rows", len(out), "pos rate", lab.y.mean())


if __name__ == "__main__":
    for p in sys.argv[1:] or ["train", "student_test"]:
        process(p)
