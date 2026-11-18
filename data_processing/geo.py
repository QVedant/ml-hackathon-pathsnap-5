"""Geometry helpers. Owner: Venkatesh.
Trips are spread worldwide (lon -100..138), so each trip is projected to ITS OWN UTM zone, chosen from the
centre of its road network. Within one zone UTM distance error is < 0.1%; one fixed zone gave 14-57% error."""
import numpy as np
from pyproj import Transformer


def utm_epsg(lon, lat):
    zone = int((float(lon) + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


class Projector:
    def __init__(self, lon0, lat0):
        self.epsg = utm_epsg(lon0, lat0)
        self._t = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)

    def __call__(self, lon, lat):
        x, y = self._t.transform(np.asarray(lon), np.asarray(lat))
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
