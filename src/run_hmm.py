"""Classifier-only vs classifier+HMM on student_test, and beta tuning (train/student-test only).
Owner: Venkatesh. Run: uv run python -m src.run_hmm LightGBM"""
import sys, time
import numpy as np
import pandas as pd
from .io import load_trip, clean_track
from .candidates import Network
from .hmm import Router, viterbi
from .evaluate import top1_predictions, route_continuity, sequence_metrics
from .viz import trip_map

BETAS = [2.0, 5.0, 10.0, 20.0, 50.0]


def main(model):
    te = pd.read_parquet("results/features_test.parquet")
    sc = pd.read_parquet(f"results/test_scores_{model}.parquet")
    te = te.merge(sc, on=["trip", "point", "arc"])
    rows = []
    for tid, g in te.groupby("trip"):
        trip = load_trip(tid, "test")
        net = Network(trip["nodes"], trip["arcs"]); router = Router(net)
        tr = clean_track(trip["track"], trip["nodes"]); px, py = net.proj(tr.lon.values, tr.lat.values)
        truth = g.groupby("point").true_arc.first()
        clf = top1_predictions(g)
        base = {"trip": tid, "n_points": len(px)}
        rows.append({**base, "method": "classifier_only", "beta": np.nan,
                     "top1": float((clf.pred_arc.values == truth.loc[clf.point].values).mean()),
                     "continuity": route_continuity(clf.pred_arc.values, net), "breaks": np.nan,
                     **sequence_metrics(clf.pred_arc.values, trip["route"]), "secs": 0})
        for beta in BETAS:
            t0 = time.time()
            path, breaks = viterbi(g, px, py, router, beta=beta)
            rows.append({**base, "method": "hmm", "beta": beta,
                         "top1": float((path.pred_arc.values == truth.loc[path.point].values).mean()),
                         "continuity": route_continuity(path.pred_arc.values, net), "breaks": breaks,
                         **sequence_metrics(path.pred_arc.values, trip["route"]), "secs": time.time() - t0})
            if beta == 10.0:
                trip_map(trip, path.pred_arc.values, title=f"Trip {tid} — {model}+HMM").save(f"maps/hmm_{tid}.html")
                trip_map(trip, clf.pred_arc.values, title=f"Trip {tid} — {model} only").save(f"maps/clf_{tid}.html")
        print(tid, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(f"results/hmm_per_trip_{model}.csv", index=False)
    summ = r.groupby(["method", "beta"], dropna=False)[["top1", "continuity", "route_f1", "breaks", "secs"]].mean()
    summ.to_csv(f"results/hmm_summary_{model}.csv")
    print(summ.round(4))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "LightGBM")
