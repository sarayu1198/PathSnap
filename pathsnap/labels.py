"""Derive a per-GPS-observation reference arc from the trip-level .route file.

The .route file only gives the chronological arc sequence, not which arc each
GPS fix belongs to.  We align the N fixes to the M route arcs with a monotone
dynamic programme: every fix i is assigned route position j(i) with
j(i) non-decreasing, minimising sum_i dist(fix_i, route_arc_j(i)).
Skipping route arcs is allowed (short arcs can be traversed within 1 s).
This is a reference label, not independently verified ground truth: the
report should discuss its limitations (e.g. stationary points at junctions).
"""
from __future__ import annotations

import numpy as np

from .network import RoadNetwork, point_to_segments


DIR_PENALTY_M = 25.0   # extra cost for putting a moving fix on an arc that points against its motion
MOVING_SPEED = 2.0     # m/s
MOTION_WIN = 5         # fixes either side used to estimate direction of travel


def align_route(xy: np.ndarray, route: np.ndarray, net: RoadNetwork, t: np.ndarray | None = None):
    """Returns (ref_arc (N,), ref_pos (N,), ref_dist (N,)).

    Distance alone cannot separate an arc from its reverse, and routes often
    contain A followed by reverse(A) at turnarounds.  So when the vehicle is
    clearly moving, an arc whose direction opposes the GPS motion costs an
    extra DIR_PENALTY_M metres.
    """
    N, M = len(xy), len(route)
    p0, p1 = net.p0[route], net.p1[route]
    L = np.maximum(net.length[route], 1e-6)
    ux, uy = (p1[:, 0] - p0[:, 0]) / L, (p1[:, 1] - p0[:, 1]) / L
    tt = np.arange(N, dtype=float) if t is None else np.asarray(t, float)
    # motion over a +-MOTION_WIN window: 1 s differences are dominated by GPS jitter when slow
    lo = np.clip(np.arange(N) - MOTION_WIN, 0, N - 1)
    hi = np.clip(np.arange(N) + MOTION_WIN, 0, N - 1)
    v = (xy[hi] - xy[lo]) / np.maximum(tt[hi] - tt[lo], 1e-6)[:, None]
    moving = np.hypot(v[:, 0], v[:, 1]) > MOVING_SPEED
    back = np.empty((N, M), dtype=np.uint16 if M < 65535 else np.uint32)
    prev = None
    for i in range(N):
        d, _ = point_to_segments(xy[i, 0], xy[i, 1], p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1])
        if moving[i]:
            d = d + DIR_PENALTY_M * ((v[i, 0] * ux + v[i, 1] * uy) < 0)
        if prev is None:
            cur = d
            back[i] = 0
        else:
            # prefix minimum of previous row and its argmin
            pm = np.minimum.accumulate(prev)
            idx = np.arange(M)
            is_new = np.r_[True, prev[1:] < pm[:-1]]
            arg = np.maximum.accumulate(np.where(is_new, idx, 0))
            cur = d + pm
            back[i] = arg
        prev = cur
    j = int(np.argmin(prev))
    pos = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        pos[i] = j
        j = int(back[i, j])
    ref_arc = route[pos]
    ref_dist, _ = point_to_segments(xy[:, 0], xy[:, 1], net.p0[ref_arc, 0], net.p0[ref_arc, 1],
                                    net.p1[ref_arc, 0], net.p1[ref_arc, 1])
    return ref_arc, pos, ref_dist
