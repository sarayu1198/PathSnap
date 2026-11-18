"""Candidate generation and per-(observation, candidate-arc) feature engineering.

Every GPS fix i gets a small candidate set C_i of nearby directed arcs. Each
(i, arc) pair becomes one row; the classifier estimates P(arc is the reference
arc of fix i | features). These probabilities are later used as HMM emissions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Trip
from .network import RoadNetwork, point_to_segments

CAND_RADIUS = 60.0   # m  (chosen from EDA: covers ~99% of reference distances)
CAND_K = 10          # max candidates per fix
CONTEXT = 3          # +- fixes used for window features

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min", "n_cand",
    "proj_t", "at_endpoint", "arc_len",
    "cos_head", "abs_head_diff", "speed", "head_reliable",
    "ctx_mean_dist", "ctx_max_dist", "ctx_cand_frac",
    "along_disp", "along_disp_norm", "cos_head_win",
    "has_reverse", "from_out_deg", "to_in_deg", "to_out_deg",
    "dt_prev", "dt_next",
    "win_speed", "cos_head_last_move", "cos_head_next_move",
]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def gps_kinematics(trip: Trip):
    xy, t = trip.xy, trip.t
    N = len(t)
    prv = np.r_[0, np.arange(N - 1)]
    nxt = np.r_[np.arange(1, N), N - 1]
    d = xy[nxt] - xy[prv]
    dt = np.maximum(t[nxt] - t[prv], 1e-6)
    speed = np.hypot(d[:, 0], d[:, 1]) / dt
    heading = np.arctan2(d[:, 1], d[:, 0])
    # wider window heading (more robust at low speed)
    w = CONTEXT
    lo = np.clip(np.arange(N) - w, 0, N - 1)
    hi = np.clip(np.arange(N) + w, 0, N - 1)
    dw = xy[hi] - xy[lo]
    head_win = np.arctan2(dw[:, 1], dw[:, 0])
    disp_win = np.hypot(dw[:, 0], dw[:, 1])
    dt_prev = np.r_[np.inf, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.inf]
    # heading of the most recent / next "clearly moving" fix, carried through stops
    win_speed = disp_win / np.maximum(t[hi] - t[lo], 1e-6)
    moving = win_speed > 2.0
    idx = np.where(moving, np.arange(N), -1)
    last = np.maximum.accumulate(idx)
    idx_n = np.where(moving, np.arange(N), N)
    nxt_m = np.minimum.accumulate(idx_n[::-1])[::-1]
    head_last = np.where(last >= 0, head_win[np.maximum(last, 0)], np.nan)
    head_next = np.where(nxt_m < N, head_win[np.minimum(nxt_m, N - 1)], np.nan)
    return (speed, heading, head_win, disp_win, lo, hi, dt_prev, dt_next,
            head_last, head_next, win_speed)


def build_candidates(trip: Trip, net: RoadNetwork, radius=CAND_RADIUS, k=CAND_K) -> pd.DataFrame:
    cands = net.candidates(trip.xy, radius, k)
    N = trip.n_obs
    obs = np.concatenate([np.full(len(c[0]), i) for i, c in enumerate(cands)])
    arc = np.concatenate([c[0] for c in cands]).astype(np.int64)
    dist = np.concatenate([c[1] for c in cands])
    t = np.concatenate([c[2] for c in cands])
    rank = np.concatenate([np.arange(len(c[0])) for c in cands])
    ncand = np.array([len(c[0]) for c in cands])
    mind = np.array([c[1][0] if len(c[1]) else np.nan for c in cands])

    (speed, heading, head_win, disp_win, lo, hi, dt_prev, dt_next,
     head_last, head_next, win_speed) = gps_kinematics(trip)
    brg = net.bearing[arc]
    hd = _wrap(heading[obs] - brg)
    hdw = _wrap(head_win[obs] - brg)

    # context window: distance of neighbouring fixes to the same arc
    xs, ys = trip.xy[:, 0], trip.xy[:, 1]
    ax, ay, bx, by = net.p0[arc, 0], net.p0[arc, 1], net.p1[arc, 0], net.p1[arc, 1]
    ctx = []
    for off in range(-CONTEXT, CONTEXT + 1):
        j = np.clip(obs + off, 0, N - 1)
        dj, tj = point_to_segments(xs[j], ys[j], ax, ay, bx, by)
        ctx.append(dj)
    ctx = np.vstack(ctx)
    d_lo, t_lo = point_to_segments(xs[lo[obs]], ys[lo[obs]], ax, ay, bx, by)
    d_hi, t_hi = point_to_segments(xs[hi[obs]], ys[hi[obs]], ax, ay, bx, by)
    L = net.length[arc]
    # along-track displacement projected on the arc's (unclamped) direction
    ux, uy = (bx - ax) / np.maximum(L, 1e-6), (by - ay) / np.maximum(L, 1e-6)
    dxw = xs[hi[obs]] - xs[lo[obs]]
    dyw = ys[hi[obs]] - ys[lo[obs]]
    along = dxw * ux + dyw * uy

    # fraction of window fixes for which this arc is also a candidate
    E = np.int64(len(net.arcs))
    keys = np.sort(obs.astype(np.int64) * E + arc)
    cnt = np.zeros(len(arc)); tot = np.zeros(len(arc))
    for off in range(-CONTEXT, CONTEXT + 1):
        if off == 0:
            continue
        j = obs + off
        valid = (j >= 0) & (j < N)
        q = j.astype(np.int64) * E + arc
        pos = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
        cnt += valid & (keys[pos] == q)
        tot += valid
    frac = cnt / np.maximum(tot, 1)

    df = pd.DataFrame({
        "obs": obs, "arc": arc,
        "dist": dist, "dist_rank": rank,
        "dist_minus_min": dist - mind[obs],
        "dist_ratio_min": (dist + 1.0) / (mind[obs] + 1.0),
        "n_cand": ncand[obs],
        "proj_t": t,
        "at_endpoint": ((t <= 1e-6) | (t >= 1 - 1e-6)).astype(np.int8),
        "arc_len": L,
        "cos_head": np.cos(hd),
        "abs_head_diff": np.abs(hd),
        "speed": speed[obs],
        "head_reliable": (speed[obs] > 2.0).astype(np.int8),
        "ctx_mean_dist": ctx.mean(0),
        "ctx_max_dist": ctx.max(0),
        "ctx_cand_frac": frac,
        "along_disp": along,
        "along_disp_norm": along / np.maximum(disp_win[obs], 1.0),
        "cos_head_win": np.cos(hdw),
        "has_reverse": (net.reverse[arc] >= 0).astype(np.int8),
        "from_out_deg": net.out_deg[net.arcs[arc, 0]],
        "to_in_deg": net.in_deg[net.arcs[arc, 1]],
        "to_out_deg": net.out_deg[net.arcs[arc, 1]],
        "dt_prev": np.minimum(dt_prev[obs], 600.0),
        "dt_next": np.minimum(dt_next[obs], 600.0),
        "win_speed": win_speed[obs],
        "cos_head_last_move": np.nan_to_num(np.cos(_wrap(head_last[obs] - brg)), nan=0.0),
        "cos_head_next_move": np.nan_to_num(np.cos(_wrap(head_next[obs] - brg)), nan=0.0),
    })
    return df
