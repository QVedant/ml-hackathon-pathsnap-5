"""Feature engineering: one row per (GPS point, candidate arc). Owner: Venkatesh.
Run: uv run python -m src.features"""
import sys, time
import numpy as np
import pandas as pd
from .io import load_trip, clean_track, trip_ids
from .geo import to_xy, bearing, angle_diff
from .candidates import Network, label_points

K, RADIUS = 8, 50.0

FEATURES = ["dist", "dist_rank", "dist_gap_nearest", "heading_diff", "speed", "dt",
            "proj_frac", "off_end", "arc_length", "out_deg", "in_deg", "n_cand", "conn_prev_nearest"]


def track_kinematics(px, py, t):
    """Smoothed speed/bearing (rolling median over 5 points) - raw positions are kept for distances."""
    sx = pd.Series(px).rolling(5, center=True, min_periods=1).median().values
    sy = pd.Series(py).rolling(5, center=True, min_periods=1).median().values
    dx = np.gradient(sx); dy = np.gradient(sy)
    dt = np.diff(t, prepend=t[0] - 1.0)
    dt[dt <= 0] = 1.0
    speed = np.hypot(np.diff(px, prepend=px[0]), np.diff(py, prepend=py[0])) / dt
    return bearing(dx, dy), speed, dt


def trip_frame(trip, k=K, radius=RADIUS, with_labels=True, net=None):
    net = net or Network(trip["nodes"], trip["arcs"])
    tr = clean_track(trip["track"], trip["nodes"])
    px, py = to_xy(tr.lon.values, tr.lat.values)
    hd, speed, dt = track_kinematics(px, py, tr.t.values)
    cands = net.candidates(px, py, k, radius)
    lab, lab_d = label_points(net, px, py, trip["route"]) if with_labels and trip["route"] is not None else (None, None)
    rows = []
    prev_nearest = -1
    for i, (a, d, f) in enumerate(cands):
        n = len(a)
        if n:
            conn = (a == prev_nearest) | ((prev_nearest >= 0) & (net.u[a] == (net.v[prev_nearest] if prev_nearest >= 0 else -1)))
            df = pd.DataFrame({
                "trip": trip["id"], "point": i, "t": tr.t.values[i], "arc": a,
                "dist": d, "dist_rank": np.arange(n), "dist_gap_nearest": d - d[0],
                "heading_diff": angle_diff(hd[i], net.bearing[a]),
                "speed": speed[i], "dt": dt[i],
                "proj_frac": np.clip(f, 0, 1), "off_end": ((f < 0) | (f > 1)).astype(np.int8),
                "arc_length": net.length[a], "out_deg": net.out_deg[a], "in_deg": net.in_deg[a],
                "n_cand": n, "conn_prev_nearest": conn.astype(np.int8)})
            if lab is not None:
                df["label"] = (a == lab[i]).astype(np.int8)
                df["true_arc"] = lab[i]
            rows.append(df)
            prev_nearest = a[0]
    out = pd.concat(rows, ignore_index=True)
    info = {"trip": trip["id"], "n_points": len(px), "n_points_raw": len(trip["track"]),
            "n_no_cand": sum(len(c[0]) == 0 for c in cands)}
    if lab is not None:
        hit = out.groupby("point").label.max()
        info["cand_miss_rate"] = 1 - hit.sum() / len(px)
        info["median_gps_to_route_m"] = float(np.median(lab_d))
    return out, info


def build(partition):
    frames, infos = [], []
    for tid in trip_ids(partition):
        t0 = time.time()
        df, info = trip_frame(load_trip(tid, partition))
        info["secs"] = round(time.time() - t0, 1)
        print(partition, info, flush=True)
        frames.append(df); infos.append(info)
    pd.concat(frames, ignore_index=True).to_parquet(f"results/features_{partition}.parquet")
    pd.DataFrame(infos).to_csv(f"results/candidates_{partition}.csv", index=False)


if __name__ == "__main__":
    for p in (sys.argv[1:] or ["test", "train"]):
        build(p)
