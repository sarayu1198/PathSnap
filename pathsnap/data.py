"""Raw-data loading and local metric projection.

Each trip folder <id>/ contains <id>.track, <id>.nodes, <id>.arcs and (for
labelled partitions) <id>.route.  See README / report section 2.2.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius (m)

DEFAULT_DATA_ROOT = os.environ.get(
    "PATHSNAP_DATA",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)


def partition_dir(partition: str, root: str | None = None) -> str:
    return os.path.join(root or DEFAULT_DATA_ROOT, partition)


def list_trips(partition: str, root: str | None = None) -> list[str]:
    d = partition_dir(partition, root)
    return sorted(t for t in os.listdir(d) if os.path.isdir(os.path.join(d, t)))


class LocalProjection:
    """Azimuthal-equidistant-like local tangent plane (equirectangular about the
    trip centroid).  For extents of a few tens of km the scale error is well
    below GPS noise, and it keeps every trip in metres regardless of where on
    the globe it is (trips span all continents, so a single UTM zone won't do).
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = EARTH_R * np.cos(np.radians(lat0)) * np.pi / 180.0
        self.ky = EARTH_R * np.pi / 180.0

    def fwd(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def inv(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / self.ky + self.lat0


@dataclass
class Trip:
    trip_id: str
    t: np.ndarray            # (N,) timestamps (s)
    lonlat: np.ndarray       # (N,2) raw GPS lon/lat
    xy: np.ndarray           # (N,2) projected GPS (m)
    node_lonlat: np.ndarray  # (V,2)
    node_xy: np.ndarray      # (V,2)
    arcs: np.ndarray         # (E,2) from_node, to_node
    route: np.ndarray | None = None   # reference arc sequence (server-side only)
    proj: LocalProjection | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return len(self.t)


def _load_txt(path, dtype=float, ncols=None):
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    if ncols is not None and a.size == 0:
        a = a.reshape(0, ncols)
    return a


def load_trip(trip_id: str, partition: str, root: str | None = None, with_route: bool = True) -> Trip:
    base = os.path.join(partition_dir(partition, root), trip_id, trip_id)
    track = _load_txt(base + ".track", float, 3)
    nodes = _load_txt(base + ".nodes", float, 2)
    arcs = _load_txt(base + ".arcs", np.int64, 2)

    # --- cleaning: drop non-finite rows, sort by time, drop duplicate timestamps
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    track = track[ok]
    order = np.argsort(track[:, 2], kind="stable")
    track = track[order]
    keep = np.ones(len(track), bool)
    keep[1:] = np.diff(track[:, 2]) > 0
    track = track[keep]

    lon0, lat0 = np.median(track[:, 0]), np.median(track[:, 1])
    proj = LocalProjection(lon0, lat0)
    x, y = proj.fwd(track[:, 0], track[:, 1])
    nx, ny = proj.fwd(nodes[:, 0], nodes[:, 1])

    route = None
    if with_route and os.path.exists(base + ".route"):
        route = np.loadtxt(base + ".route", dtype=np.int64, ndmin=1)

    return Trip(
        trip_id=trip_id,
        t=track[:, 2],
        lonlat=track[:, :2],
        xy=np.column_stack([x, y]),
        node_lonlat=nodes,
        node_xy=np.column_stack([nx, ny]),
        arcs=arcs,
        route=route,
        proj=proj,
        extra={"n_raw_rows": int(ok.size), "n_dropped": int(ok.size - len(track))},
    )
