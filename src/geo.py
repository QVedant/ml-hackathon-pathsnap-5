"""Geometry helpers: lon/lat -> metres (UTM 53N, the zone covering the dataset near Kyoto). Owner: Venkatesh."""
import numpy as np
from pyproj import Transformer

_T = Transformer.from_crs("EPSG:4326", "EPSG:32653", always_xy=True)


def to_xy(lon, lat):
    x, y = _T.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def bearing(dx, dy):
    """Compass bearing in degrees [0,360) from x/y deltas."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments + projection fraction (unclipped)."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    f = np.where(L2 > 0, ((px - ax) * vx + (py - ay) * vy) / np.where(L2 > 0, L2, 1), 0.0)
    fc = np.clip(f, 0, 1)
    qx, qy = ax + fc * vx, ay + fc * vy
    return np.hypot(px - qx, py - qy), f
