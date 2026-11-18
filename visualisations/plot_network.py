"""Road-network plot for one train trip: every arc, the GPS trace and the reference route.
Run: uv run python -m visualisations.plot_network"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from data_processing.io import load_trip
from data_processing.candidates import Network

TRIP = "00000099"
trip = load_trip(TRIP, "train")
net = Network(trip["nodes"], trip["arcs"])
x0, y0 = net.proj(trip["track"].lon.values, trip["track"].lat.values)
pad = 400
keep = ((net.ax > x0.min() - pad) & (net.ax < x0.max() + pad) & (net.ay > y0.min() - pad) & (net.ay < y0.max() + pad))
seg = np.stack([np.c_[net.ax, net.ay], np.c_[net.bx, net.by]], axis=1)[keep]
r = trip["route"]
fig, ax = plt.subplots(figsize=(10, 6))
ax.add_collection(LineCollection(seg, colors="#9aa3b2", linewidths=0.5, label=f"road arcs ({keep.sum():,} of {len(net.u):,})"))
ax.add_collection(LineCollection(np.stack([np.c_[net.ax[r], net.ay[r]], np.c_[net.bx[r], net.by[r]]], axis=1), colors="green", linewidths=3, label="reference route"))
ax.plot(x0, y0, ".", ms=1.5, color="blue", label="GPS")
ax.set_xlim(x0.min() - pad, x0.max() + pad); ax.set_ylim(y0.min() - pad, y0.max() + pad)
ax.set_aspect("equal"); ax.set_xlabel(f"UTM easting (m), EPSG:{net.proj.epsg}"); ax.set_ylabel("UTM northing (m)")
ax.set_title(f"Road network around trip {TRIP} (train)"); ax.legend(loc="best", markerscale=6)
plt.tight_layout(); plt.savefig("results/plots/eda_network.png", dpi=130)
print("saved results/plots/eda_network.png")
