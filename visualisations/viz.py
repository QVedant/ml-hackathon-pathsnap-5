"""Folium maps: GPS points, local road network, reference route, predicted route. Owner: Vedant."""
import folium
import numpy as np
from scipy.spatial import cKDTree


def trip_map(trip, pred_arcs=None, network_radius_m=150, title=""):
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    # road network near the trajectory only (full network is up to 300k arcs)
    lat0 = np.radians(tr.lat.mean())
    sc = np.array([111320 * np.cos(lat0), 110540])
    tree = cKDTree(tr[["lon", "lat"]].values * sc)
    mid = (nodes.values[arcs.u] + nodes.values[arcs.v]) / 2 * sc
    near = np.where(tree.query(mid)[0] < network_radius_m)[0]
    net = folium.FeatureGroup("Road network (nearby)")
    for k in near:
        a, b = nodes.iloc[arcs.u[k]], nodes.iloc[arcs.v[k]]
        folium.PolyLine([(a.lat, a.lon), (b.lat, b.lon)], color="#999", weight=1).add_to(net)
    net.add_to(m)

    def arc_layer(seq, name, color, w):
        g = folium.FeatureGroup(name)
        for k in seq:
            a, b = nodes.iloc[arcs.u[k]], nodes.iloc[arcs.v[k]]
            folium.PolyLine([(a.lat, a.lon), (b.lat, b.lon)], color=color, weight=w, opacity=0.8).add_to(g)
        g.add_to(m)

    if trip.get("route") is not None:
        arc_layer(trip["route"], "Reference route", "green", 6)
    if pred_arcs is not None:
        arc_layer(dict.fromkeys(pred_arcs), "Predicted route", "red", 3)
    gps = folium.FeatureGroup("Raw GPS")
    for la, lo in zip(tr.lat.values[::2], tr.lon.values[::2]):
        folium.CircleMarker((la, lo), radius=2, color="blue", fill=True).add_to(gps)
    gps.add_to(m)
    folium.LayerControl().add_to(m)
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:50px;z-index:999;background:white;padding:6px;'
            f'border:1px solid #333">{title}<br><span style="color:blue">● GPS</span> '
            f'<span style="color:green">━ reference</span> <span style="color:red">━ predicted</span></div>'))
    return m
