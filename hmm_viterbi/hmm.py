"""HMM + Viterbi route reconstruction over classifier-scored candidates. Owner: Venkatesh.

Hidden state at time t  = one candidate arc of GPS point t.
Emission  log B_t(j)    = log P(arc j is correct | features)   (from the selected classifier)
Transition log A(i->j)  = -|d_network(i,j) - d_gps(t-1,t)| / beta   (Newson & Krumm style)
                          -inf if j is not reachable from i within the cutoff.
Viterbi: delta_t(j) = max_i [delta_{t-1}(i) + log A(i->j)] + log B_t(j)
If every transition into point t is -inf (disconnected / missing data) the chain is broken and
decoding restarts at t from emissions only.
"""
import heapq
from collections import defaultdict
import numpy as np
import pandas as pd

CUTOFF = 1500.0  # metres of network search per transition


class Router:
    def __init__(self, net):
        self.net = net
        self.adj = defaultdict(list)
        for k, (u, v) in enumerate(zip(net.u, net.v)):
            self.adj[int(u)].append((int(v), float(net.length[k])))
        self.cache = {}

    def dists_from(self, src):
        """Dijkstra from node src, truncated at CUTOFF metres (cached)."""
        if src in self.cache:
            return self.cache[src]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, x = heapq.heappop(pq)
            if d > dist.get(x, np.inf) or d > CUTOFF:
                continue
            for y, w in self.adj[x]:
                nd = d + w
                if nd < dist.get(y, np.inf) and nd <= CUTOFF:
                    dist[y] = nd
                    heapq.heappush(pq, (nd, y))
        self.cache[src] = dist
        return dist

    def route_dist(self, a, fa, b, fb):
        """Network distance from position fa along arc a to position fb along arc b."""
        net = self.net
        if a == b:
            gap = (fb - fa) * net.length[a]
            return gap if gap >= -5.0 else np.inf  # allow tiny backward jitter
        d = self.dists_from(int(net.v[a])).get(int(net.u[b]), np.inf)
        return (1 - fa) * net.length[a] + d + fb * net.length[b]


def viterbi(frame_trip: pd.DataFrame, px, py, router, beta=10.0, score_col="score", emis_w=1.0):
    """frame_trip: rows of one trip with point, arc, proj_frac, dt and score columns.
    Returns (DataFrame point->pred_arc, n_breaks)."""
    groups = [g for _, g in frame_trip.sort_values(["point", "dist"]).groupby("point", sort=True)]
    pts = [int(g.point.iloc[0]) for g in groups]
    arcs = [g.arc.values for g in groups]
    fracs = [g.proj_frac.values for g in groups]
    em = [emis_w * np.log(np.clip(g[score_col].values, 1e-6, 1)) for g in groups]
    dts = [float(g.dt.iloc[0]) for g in groups]
    T = len(groups)
    delta = [em[0]]
    back = [np.full(len(arcs[0]), -1)]
    breaks = 0
    for t in range(1, T):
        p0, p1 = pts[t - 1], pts[t]
        dgc = np.hypot(px[p1] - px[p0], py[p1] - py[p0])
        b = beta * max(1.0, dts[t])  # larger GPS gap -> more tolerant transitions
        A = np.full((len(arcs[t - 1]), len(arcs[t])), -np.inf)
        for i, (ai, fi) in enumerate(zip(arcs[t - 1], fracs[t - 1])):
            for j, (aj, fj) in enumerate(zip(arcs[t], fracs[t])):
                dr = router.route_dist(ai, fi, aj, fj)
                if np.isfinite(dr):
                    A[i, j] = -abs(dr - dgc) / b
        scores = delta[-1][:, None] + A
        best = scores.argmax(0)
        val = scores[best, np.arange(len(arcs[t]))]
        if not np.isfinite(val).any():  # broken chain -> restart
            breaks += 1
            delta.append(em[t]); back.append(np.full(len(arcs[t]), -1))
        else:
            delta.append(val + em[t]); back.append(np.where(np.isfinite(val), best, -1))
    # backtrack (segment by segment if chain was broken)
    path = np.empty(T, np.int64)
    j = int(np.argmax(delta[-1]))
    for t in range(T - 1, -1, -1):
        path[t] = arcs[t][j]
        bj = back[t][j]
        if t > 0:
            j = int(bj) if bj >= 0 else int(np.argmax(delta[t - 1]))
    return pd.DataFrame({"point": pts, "pred_arc": path}), breaks
