"""Folium map generation (used by EDA, HMM evaluation and the report)."""
from __future__ import annotations

import folium
import numpy as np

from .data import Trip
from .network import RoadNetwork


def _lines(trip: Trip, arc_ids):
    nl = trip.node_lonlat
    return [[(nl[u, 1], nl[u, 0]), (nl[v, 1], nl[v, 0])] for u, v in trip.arcs[np.asarray(arc_ids, int)]]


def local_network_arcs(trip: Trip, net: RoadNetwork, buffer_m: float = 150.0, max_arcs: int = 15000):
    """Arcs whose sample points lie within buffer_m of any GPS fix (one direction only)."""
    hits = net.tree.query_ball_point(trip.xy[::3], buffer_m)
    s = np.unique(np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)] or [np.zeros(0, int)]))
    arcs = np.unique(net.sample_arc[s])
    rev = net.reverse[arcs]
    arcs = arcs[(rev < 0) | (arcs < rev)]  # draw each two-way street once
    return arcs[:max_arcs]


def trip_map(trip: Trip, net: RoadNetwork, pred_arcs=None, ref_route=None, title="",
             gps_every: int = 1, show_network: bool = True) -> folium.Map:
    lat, lon = trip.lonlat[:, 1], trip.lonlat[:, 0]
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=15,
                   tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    if show_network:
        fg = folium.FeatureGroup(name="Road network (local)", show=True)
        for seg in _lines(trip, local_network_arcs(trip, net)):
            folium.PolyLine(seg, color="#9e9e9e", weight=2, opacity=0.6).add_to(fg)
        fg.add_to(m)

    if ref_route is not None:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        for seg in _lines(trip, ref_route):
            folium.PolyLine(seg, color="#2e7d32", weight=7, opacity=0.7).add_to(fg)
        fg.add_to(m)

    if pred_arcs is not None:
        fg = folium.FeatureGroup(name="Predicted route (PathSnap)", show=True)
        for seg in _lines(trip, pred_arcs):
            folium.PolyLine(seg, color="#d81b60", weight=4, opacity=0.9).add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup(name="Raw GPS fixes", show=True)
    for i in range(0, trip.n_obs, gps_every):
        folium.CircleMarker((lat[i], lon[i]), radius=2, color="#1e40af", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"fix {i}, t={trip.t[i]:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.Marker((lat[0], lon[0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    legend = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
         padding: 8px 12px; border: 1px solid #888; border-radius: 6px; font: 13px sans-serif;">
      <b>{title}</b><br>
      <span style="color:#1e40af">&#9679;</span> Raw GPS fixes<br>
      <span style="color:#9e9e9e">&#9473;&#9473;</span> Road network<br>
      {'<span style="color:#2e7d32">&#9473;&#9473;</span> Reference route<br>' if ref_route is not None else ''}
      {'<span style="color:#d81b60">&#9473;&#9473;</span> Predicted route<br>' if pred_arcs is not None else ''}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m
