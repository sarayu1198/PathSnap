"""Road-network geometry, spatial index and bounded shortest paths."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .data import Trip


def point_to_segments(px, py, ax, ay, bx, by):
    """Distance from point(s) to segment(s) (broadcastable).
    Returns (dist, t) with t in [0,1] the projection fraction along a->b."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = np.where(L2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t


class RoadNetwork:
    def __init__(self, trip: Trip, sample_step: float = 10.0):
        self.node_xy = trip.node_xy
        self.arcs = trip.arcs
        a, b = self.arcs[:, 0], self.arcs[:, 1]
        self.p0 = self.node_xy[a]
        self.p1 = self.node_xy[b]
        d = self.p1 - self.p0
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = np.arctan2(d[:, 1], d[:, 0])  # math angle (rad), east = 0
        V = len(self.node_xy)
        self.out_deg = np.bincount(a, minlength=V)
        self.in_deg = np.bincount(b, minlength=V)

        # reverse-arc lookup (b->a exists?)
        key = a * V + b
        rkey = b * V + a
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, len(key) - 1)
        hit = key[order][pos] == rkey
        self.reverse = np.where(hit, order[pos], -1)

        # adjacency (CSR) for Dijkstra
        o = np.argsort(a, kind="stable")
        self.adj_arc = o
        self.adj_ptr = np.concatenate([[0], np.cumsum(np.bincount(a, minlength=V))])

        # spatial index: densified samples along every arc
        self.sample_step = sample_step
        nseg = np.maximum(1, np.ceil(self.length / sample_step).astype(int))
        arc_rep = np.repeat(np.arange(len(a)), nseg + 1)
        frac = self._fracs(nseg)
        pts = self.p0[arc_rep] + frac[:, None] * d[arc_rep]
        self.sample_arc = arc_rep
        self.tree = cKDTree(pts)
        self._sp_cache: dict[int, dict[int, float]] = {}

    @staticmethod
    def _fracs(nseg):
        total = int((nseg + 1).sum())
        starts = np.repeat(np.cumsum(np.concatenate([[0], nseg[:-1] + 1])), nseg + 1)
        k = np.arange(total) - starts
        return k / np.repeat(nseg, nseg + 1)

    # ------------------------------------------------------------------ candidates
    def candidates(self, xy: np.ndarray, radius: float, k_max: int, k_fallback: int = 3):
        """For each point return arrays (arc_ids, dist, t) of up to k_max nearest arcs
        within `radius`; if none are within radius, the k_fallback nearest arcs."""
        r = radius + self.sample_step / 2 + 1.0
        lists = self.tree.query_ball_point(xy, r, workers=-1)
        out = []
        for i, idx in enumerate(lists):
            if len(idx) == 0:
                _, idx = self.tree.query(xy[i], k=64)
                idx = np.atleast_1d(idx)
                idx = idx[idx < len(self.sample_arc)]
            arcs = np.unique(self.sample_arc[np.asarray(idx, dtype=np.int64)])
            dist, t = point_to_segments(xy[i, 0], xy[i, 1], self.p0[arcs, 0], self.p0[arcs, 1],
                                        self.p1[arcs, 0], self.p1[arcs, 1])
            o = np.argsort(dist, kind="stable")
            arcs, dist, t = arcs[o], dist[o], t[o]
            within = dist <= radius
            n = min(k_max, int(within.sum())) if within.any() else min(k_fallback, len(arcs))
            out.append((arcs[:n], dist[:n], t[:n]))
        return out

    # ------------------------------------------------------------------ routing
    def sp_from(self, src: int, cutoff: float) -> dict[int, float]:
        """Bounded Dijkstra from node src (distances in metres, <= cutoff). Cached."""
        cached = self._sp_cache.get(src)
        if cached is not None and cached.get(-1, 0.0) >= cutoff:
            return cached
        dist = {src: 0.0}
        pq = [(0.0, src)]
        ptr, adj, to, ln = self.adj_ptr, self.adj_arc, self.arcs[:, 1], self.length
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf):
                continue
            for e in adj[ptr[u]:ptr[u + 1]]:
                v = int(to[e])
                nd = d + ln[e]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        dist[-1] = cutoff  # marker: search radius used
        self._sp_cache[src] = dist
        return dist

    def shortest_arc_path(self, src_node: int, dst_node: int, cutoff: float = 5000.0):
        """Arc sequence of a shortest path src_node -> dst_node (for drawing gaps)."""
        if src_node == dst_node:
            return []
        dist, prev = {src_node: 0.0}, {}
        pq = [(0.0, src_node)]
        ptr, adj, to, ln = self.adj_ptr, self.adj_arc, self.arcs[:, 1], self.length
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst_node:
                break
            if d > dist.get(u, np.inf):
                continue
            for e in adj[ptr[u]:ptr[u + 1]]:
                v = int(to[e]); nd = d + ln[e]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd; prev[v] = int(e)
                    heapq.heappush(pq, (nd, v))
        if dst_node not in prev:
            return None
        path, v = [], dst_node
        while v != src_node:
            e = prev[v]; path.append(e); v = int(self.arcs[e, 0])
        return path[::-1]
