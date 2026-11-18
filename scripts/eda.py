"""Step 1: exploratory data analysis of the raw dataset (train + student_test only).

Outputs: results/eda_inventory.csv, results/eda_summary.txt, figures/eda_*.png, maps/eda_*.html
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.data import list_trips, load_trip, partition_dir  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG, RES, MAPS = (os.path.join(ROOT, d) for d in ("figures", "results", "maps"))
for d in (FIG, RES, MAPS):
    os.makedirs(d, exist_ok=True)


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


def main():
    rows, all_dt, all_ref = [], {"train": [], "student_test": []}, {"train": [], "student_test": []}
    for part in ["train", "student_test"]:
        for tid in list_trips(part):
            base = os.path.join(partition_dir(part), tid, tid)
            raw = np.loadtxt(base + ".track", ndmin=2)
            trip = load_trip(tid, part)
            net = RoadNetwork(trip)
            dt = np.diff(raw[:, 2])
            step = haversine(raw[:-1, 0], raw[:-1, 1], raw[1:, 0], raw[1:, 1])
            V, E = len(trip.node_xy), len(trip.arcs)
            g = coo_matrix((np.ones(E), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(V, V))
            n_weak, _ = connected_components(g, directed=True, connection="weak")
            n_strong, lab_s = connected_components(g, directed=True, connection="strong")
            isolated = int(((net.in_deg + net.out_deg) == 0).sum())
            r = trip.route
            route_conn = float(np.mean(trip.arcs[r[:-1], 1] == trip.arcs[r[1:], 0])) if len(r) > 1 else 1.0
            _, _, ref_dist = align_route(trip.xy, r, net, trip.t)
            all_dt[part].append(dt); all_ref[part].append(ref_dist)
            ext = trip.xy.max(0) - trip.xy.min(0)
            rows.append(dict(
                partition=part, trip=tid, n_gps=len(raw), n_nan=int(np.isnan(raw).sum()),
                n_nonmonotonic_t=int((dt <= 0).sum()), n_nodes=V, n_arcs=E,
                n_route_arcs=len(r), route_unique_arcs=len(np.unique(r)), route_connected_frac=route_conn,
                duration_s=raw[-1, 2] - raw[0, 2], frac_dt_1s=float((dt == 1).mean()),
                max_gap_s=float(dt.max()), n_gaps_gt10s=int((dt > 10).sum()),
                max_step_m=float(step.max()), n_jumps_gt100m=int((step > 100).sum()),
                path_len_km=float(step.sum() / 1000), extent_x_km=ext[0] / 1000, extent_y_km=ext[1] / 1000,
                lon_center=float(np.median(raw[:, 0])), lat_center=float(np.median(raw[:, 1])),
                weak_components=n_weak, strong_components=n_strong,
                largest_scc_frac=float(np.bincount(lab_s).max() / V), isolated_nodes=isolated,
                twoway_arc_frac=float((net.reverse >= 0).mean()),
                ref_dist_median_m=float(np.median(ref_dist)), ref_dist_p95_m=float(np.percentile(ref_dist, 95)),
                frac_fixes_gt50m=float((ref_dist > 50).mean()),
                track_kb=os.path.getsize(base + ".track") / 1024, arcs_kb=os.path.getsize(base + ".arcs") / 1024))
            print(rows[-1]["partition"], tid, flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(RES, "eda_inventory.csv"), index=False)

    with open(os.path.join(RES, "eda_summary.txt"), "w") as f:
        for part, g in inv.groupby("partition"):
            f.write(f"===== {part}: {len(g)} trips, {g.n_gps.sum()} GPS fixes\n")
            f.write(g.drop(columns=["trip", "partition"]).describe().T.to_string() + "\n\n")
        f.write("Trips with gaps > 60 s or jumps > 100 m:\n")
        f.write(inv[(inv.max_gap_s > 60) | (inv.n_jumps_gt100m > 0)][
            ["partition", "trip", "max_gap_s", "n_gaps_gt10s", "n_jumps_gt100m", "frac_fixes_gt50m"]].to_string())

    tr = inv[inv.partition == "train"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(tr.n_gps, bins=30, color="#4c72b0"); ax[0, 0].set_title("GPS fixes per trip (train)")
    ax[0, 0].set_xlabel("fixes")
    dts = np.concatenate(all_dt["train"])
    vals, cnts = np.unique(np.clip(dts, 0, 11), return_counts=True)
    ax[0, 1].bar(vals, cnts / cnts.sum(), color="#55a868"); ax[0, 1].set_yscale("log")
    ax[0, 1].set_title("Sampling interval (train, 11 = >10 s)"); ax[0, 1].set_xlabel("dt (s)")
    rd = np.concatenate(all_ref["train"])
    ax[0, 2].hist(np.clip(rd, 0, 80), bins=80, color="#c44e52")
    ax[0, 2].set_title("Distance GPS fix -> aligned reference arc (train)"); ax[0, 2].set_xlabel("m (clipped at 80)")
    ax[1, 0].scatter(tr.n_arcs, tr.n_gps, s=12); ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("arcs in network"); ax[1, 0].set_ylabel("GPS fixes"); ax[1, 0].set_title("Network size vs track length")
    ax[1, 1].hist(tr.largest_scc_frac, bins=20, color="#8172b2"); ax[1, 1].set_title("Largest strongly-connected component frac")
    ax[1, 2].scatter(inv.lon_center, inv.lat_center, c=(inv.partition == "train"), cmap="coolwarm", s=14)
    ax[1, 2].set_title("Trip locations (red=train, blue=student_test)"); ax[1, 2].set_xlabel("lon"); ax[1, 2].set_ylabel("lat")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_overview.png"), dpi=130); plt.close()

    # trajectory + network plot for two example trips
    for part, tid in [("train", "00000001"), ("student_test", "00000052")]:
        trip = load_trip(tid, part); net = RoadNetwork(trip)
        fig, ax = plt.subplots(figsize=(8, 8))
        from pathsnap.viz import local_network_arcs
        arcs = local_network_arcs(trip, net, 300)
        for a in arcs:
            p, q = net.p0[a], net.p1[a]
            ax.plot([p[0], q[0]], [p[1], q[1]], color="#bbbbbb", lw=0.8)
        for a in trip.route:
            p, q = net.p0[a], net.p1[a]
            ax.plot([p[0], q[0]], [p[1], q[1]], color="green", lw=3, alpha=0.6)
        ax.plot(trip.xy[:, 0], trip.xy[:, 1], ".", ms=2, color="#1e40af")
        ax.set_aspect("equal"); ax.set_title(f"{part} trip {tid}: GPS (blue), reference route (green), network (grey)")
        ax.set_xlabel("x (m, local projection)"); ax.set_ylabel("y (m)")
        plt.savefig(os.path.join(FIG, f"eda_trajectory_{tid}.png"), dpi=130); plt.close()
        trip_map(trip, net, ref_route=trip.route, title=f"EDA - {part} trip {tid}").save(
            os.path.join(MAPS, f"eda_{part}_{tid}.html"))


if __name__ == "__main__":
    main()
