"""Road network preparation, candidate generation and point->reference-arc labelling. Owner: Venkatesh."""
import numpy as np
from scipy.spatial import cKDTree
from .geo import Projector, bearing, point_segment

SAMPLE_STEP = 15.0  # metres between sample points along each arc for the KD-tree


class Network:
    def __init__(self, nodes, arcs):
        self.proj = Projector(nodes.lon.median(), nodes.lat.median())  # per-trip UTM zone
        self.nx, self.ny = self.proj(nodes.lon.values, nodes.lat.values)
        self.u = arcs.u.values.astype(np.int64)
        self.v = arcs.v.values.astype(np.int64)
        self.ax, self.ay = self.nx[self.u], self.ny[self.u]
        self.bx, self.by = self.nx[self.v], self.ny[self.v]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing(self.bx - self.ax, self.by - self.ay)
        n = len(self.nx)
        self.out_deg = np.bincount(self.u, minlength=n)[self.v]   # choices after the arc
        self.in_deg = np.bincount(self.v, minlength=n)[self.u]
        # sample points along arcs -> KD-tree
        k = np.maximum(1, np.ceil(self.length / SAMPLE_STEP).astype(int))
        arc_idx = np.repeat(np.arange(len(self.u)), k + 1)
        frac = np.concatenate([np.linspace(0, 1, ki + 1) for ki in k])
        sx = self.ax[arc_idx] + frac * (self.bx - self.ax)[arc_idx]
        sy = self.ay[arc_idx] + frac * (self.by - self.ay)[arc_idx]
        self.sample_arc = arc_idx
        self.tree = cKDTree(np.c_[sx, sy])

    def candidates(self, px, py, k=8, radius=50.0):
        """For each point return list of (arc_ids, dists, fracs) for up to k nearest arcs within radius."""
        hits = self.tree.query_ball_point(np.c_[px, py], r=radius + SAMPLE_STEP)
        out = []
        for i, h in enumerate(hits):
            if not h:
                out.append((np.empty(0, np.int64), np.empty(0), np.empty(0)))
                continue
            a = np.unique(self.sample_arc[h])
            d, f = point_segment(px[i], py[i], self.ax[a], self.ay[a], self.bx[a], self.by[a])
            m = d <= radius
            a, d, f = a[m], d[m], f[m]
            o = np.argsort(d)[:k]
            out.append((a[o], d[o], f[o]))
        return out


def label_points(net: Network, px, py, route, max_step=20):
    """Assign each GPS point a reference arc from the ordered route by DP alignment.
    State = index j in the route. Cost = point-to-arc distance. Between consecutive GPS points the
    route index may stay or move forward by up to max_step arcs (never backwards), so the labelling
    follows route order (handles loops / revisits) and is globally optimal, unlike a greedy pointer.
    Returns (labels, dist_to_label)."""
    ra = np.asarray(route)
    M = len(ra)
    D = np.empty((len(px), M))
    for i in range(len(px)):
        D[i], _ = point_segment(px[i], py[i], net.ax[ra], net.ay[ra], net.bx[ra], net.by[ra])
    cost = D[0].copy()
    back = np.empty((len(px), M), np.int32)
    back[0] = np.arange(M)
    idx = np.arange(M)
    for i in range(1, len(px)):
        best = cost.copy(); arg = idx.copy()
        for s in range(1, max_step + 1):  # come from j-s
            c = np.full(M, np.inf); c[s:] = cost[:-s]
            better = c < best
            best[better] = c[better]; arg[better] = idx[better] - s
        cost = best + D[i]
        back[i] = arg
    j = int(np.argmin(cost))
    path = np.empty(len(px), np.int64)
    for i in range(len(px) - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    return ra[path], D[np.arange(len(px)), path]
