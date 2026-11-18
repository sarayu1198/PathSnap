"""HMM / Viterbi route reconstruction.

Hidden state at fix i: one of the candidate arcs C_i (with its projection point).
Emission:   log P(z_i | s) = w_e * log(p_clf(s) / sum_{s' in C_i} p_clf(s'))
Transition: log P(s_i | s_{i-1}) = -|d_route - d_gps| / (beta * sqrt(dt))
                                   - uturn_pen   * [s_i is the reverse arc of s_{i-1}]
                                   - back_pen    * [same arc, moved backwards > back_tol m]
  d_route = network distance between the two projection points (bounded Dijkstra),
  d_gps   = straight-line distance between the two fixes.
  Unreachable pairs get -inf.  If every pair at a step is -inf, or the time gap is
  larger than gap_s, the chain is broken and decoding restarts from the emissions
  (the break is reported; the two pieces are joined by a shortest path for display).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .network import RoadNetwork


@dataclass
class HMMParams:
    w_e: float = 1.0
    beta: float = 5.0
    uturn_pen: float = 4.0
    back_pen: float = 1.0
    back_tol: float = 8.0
    gap_s: float = 60.0

    def to_dict(self):
        return asdict(self)


@dataclass
class TripLattice:
    """Parameter-independent precomputation for one trip."""
    obs: np.ndarray                 # (N,) observation indices with >=1 candidate
    arcs: list                      # list of arrays of candidate arcs
    probs: list                     # list of arrays of classifier probabilities
    dt: np.ndarray                  # (N,) time since previous kept obs
    d_gps: np.ndarray               # (N,)
    d_route: list                   # list of (Kprev, Kcur) arrays (inf = unreachable); [0] is None
    same_back: list                 # list of (Kprev, Kcur) arrays: backward movement on same arc (m)
    uturn: list                     # list of (Kprev, Kcur) bool arrays


def build_lattice(cand: pd.DataFrame, t: np.ndarray, xy: np.ndarray, net: RoadNetwork,
                  prob_col: str = "p", gap_s: float = 60.0) -> TripLattice:
    cand = cand.sort_values(["obs", "dist"])
    groups = cand.groupby("obs", sort=True)
    obs = np.fromiter(groups.groups.keys(), dtype=np.int64)
    arcs, probs, ts = [], [], []
    for _, g in groups:
        arcs.append(g["arc"].values.astype(np.int64))
        probs.append(g[prob_col].values.astype(float))
        ts.append(g["proj_t"].values.astype(float))
    N = len(obs)
    dt = np.r_[np.inf, np.diff(t[obs])]
    d_gps = np.r_[0.0, np.hypot(*np.diff(xy[obs], axis=0).T)]
    L, frm, to, rev = net.length, net.arcs[:, 0], net.arcs[:, 1], net.reverse
    d_route, same_back, uturn = [None], [None], [None]
    for i in range(1, N):
        a, b = arcs[i - 1], arcs[i]
        ta, tb = ts[i - 1], ts[i]
        D = np.full((len(a), len(b)), np.inf)
        if dt[i] > gap_s:
            d_route.append(D); same_back.append(np.zeros_like(D)); uturn.append(np.zeros(D.shape, bool))
            continue
        cutoff = max(300.0, 3.0 * d_gps[i] + 100.0)
        same = a[:, None] == b[None, :]
        along = (tb[None, :] - ta[:, None]) * L[a][:, None]
        D[same] = np.abs(along[same])
        back = np.where(same, np.maximum(-along, 0.0), 0.0)
        rem = (1.0 - ta) * L[a]
        head = tb * L[b]
        for r in range(len(a)):
            sp = net.sp_from(int(to[a[r]]), cutoff)
            for c in range(len(b)):
                if same[r, c]:
                    continue
                d = sp.get(int(frm[b[c]]))
                if d is not None:
                    D[r, c] = rem[r] + d + head[c]
        d_route.append(D)
        same_back.append(back)
        uturn.append(rev[a][:, None] == b[None, :])
    return TripLattice(obs, arcs, probs, dt, d_gps, d_route, same_back, uturn)


def viterbi(lat: TripLattice, prm: HMMParams):
    """Returns (pred_arc per kept obs, n_breaks)."""
    N = len(lat.obs)
    eps = 1e-6
    emis = []
    for p in lat.probs:
        q = np.clip(p, eps, 1.0)
        emis.append(prm.w_e * np.log(q / q.sum()))
    delta = emis[0].copy()
    backptr = [None]
    breaks = 0
    for i in range(1, N):
        D = lat.d_route[i]
        dtt = lat.dt[i]
        if dtt > prm.gap_s:
            T = np.full(D.shape, -np.inf)
        else:
            beta = prm.beta * np.sqrt(max(dtt, 1.0))
            T = -np.abs(D - lat.d_gps[i]) / beta
            T -= prm.uturn_pen * lat.uturn[i]
            T -= prm.back_pen * (lat.same_back[i] > prm.back_tol)
        S = delta[:, None] + T
        best = S.argmax(0)
        val = S[best, np.arange(S.shape[1])]
        if not np.isfinite(val).any():
            # chain break: the previous segment ends at its own best state
            breaks += 1
            best = np.full(S.shape[1], int(np.argmax(delta)))
            val = np.full(S.shape[1], 0.0)  # restart scores from the emissions only
        # otherwise, states with no valid predecessor get -inf and drop out
        delta = val + emis[i]
        backptr.append(best)
    states = np.empty(N, dtype=np.int64)
    states[-1] = int(np.argmax(delta))
    for i in range(N - 1, 0, -1):
        states[i - 1] = backptr[i][states[i]]
    pred = np.array([lat.arcs[i][s] for i, s in enumerate(states)])
    return pred, breaks


def classifier_only(lat: TripLattice):
    return np.array([a[int(np.argmax(p))] for a, p in zip(lat.arcs, lat.probs)])


def stitch_route(pred_arcs: np.ndarray, net: RoadNetwork, max_gap_m: float = 3000.0):
    """Collapse per-fix arcs into a continuous arc sequence, filling holes with shortest paths."""
    seq = []
    for a in pred_arcs:
        a = int(a)
        if seq and seq[-1] == a:
            continue
        if seq:
            u, v = int(net.arcs[seq[-1], 1]), int(net.arcs[a, 0])
            if u != v:
                path = net.shortest_arc_path(u, v, max_gap_m)
                if path:
                    seq.extend(path)
        seq.append(a)
    return np.array(seq, dtype=np.int64)
