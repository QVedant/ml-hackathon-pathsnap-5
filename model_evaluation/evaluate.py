"""Metric helpers shared by model training and HMM evaluation. Owner: Ved."""
import numpy as np
import pandas as pd


def top1_predictions(frame: pd.DataFrame, score_col="score") -> pd.DataFrame:
    """Per GPS point, the candidate with highest score. Returns trip, point, pred_arc, true_arc."""
    idx = frame.groupby(["trip", "point"])[score_col].idxmax()
    top = frame.loc[idx, ["trip", "point", "arc", "true_arc"]].rename(columns={"arc": "pred_arc"})
    return top.sort_values(["trip", "point"]).reset_index(drop=True)


def top1_accuracy(pred: pd.DataFrame) -> float:
    """Observation-level Top-1: predicted arc == reference arc (points without candidates excluded
    upstream; they are reported separately as candidate misses)."""
    return float((pred.pred_arc == pred.true_arc).mean())


def candidate_miss_rate(frame: pd.DataFrame) -> float:
    return float(1 - frame.groupby(["trip", "point"]).label.max().mean())


def route_continuity(arc_seq, net) -> float:
    """Share of consecutive predictions that are physically possible: same arc, or the next arc starts
    where the previous one ends."""
    a = np.asarray(arc_seq)
    if len(a) < 2:
        return 1.0
    p, n = a[:-1], a[1:]
    ok = (p == n) | (net.v[p] == net.u[n])
    return float(ok.mean())


def sequence_metrics(arc_seq, route):
    """Route-level view: compress predicted per-point arcs to an ordered arc list and compare with the
    reference route as sets (precision / recall / F1 of traversed arcs)."""
    pred = pd.unique(np.asarray(arc_seq))
    P, R = set(pred.tolist()), set(np.asarray(route).tolist())
    inter = len(P & R)
    prec = inter / len(P) if P else 0.0
    rec = inter / len(R) if R else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
