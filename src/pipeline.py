"""End-to-end inference for one trip: preprocess -> candidates -> features -> classifier -> HMM/Viterbi.
Ground truth is NEVER passed into prediction; it is only used afterwards by `score_trip`. Owner: Venkatesh."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from .io import clean_track
from .candidates import Network, label_points
from .features import trip_frame, FEATURES
from .hmm import Router, viterbi
from .evaluate import route_continuity, sequence_metrics

ROOT = Path(__file__).resolve().parent.parent
HMM_CFG = {"beta": 50.0}  # chosen on student-test (results/hmm_summary_LightGBM.csv); never on evaluator data


def load_model():
    for f in ("final_lgbm.joblib", "LightGBM.joblib"):
        if (ROOT / "models" / f).exists():
            return joblib.load(ROOT / "models" / f), f
    raise FileNotFoundError("no model in models/")


def load_trip_dir(d: Path) -> dict:
    d = Path(d); tid = d.name
    rd = lambda ext, names: pd.read_csv(d / f"{tid}.{ext}", sep=r"\s+", header=None, names=names)
    return {"id": tid, "track": rd("track", ["lon", "lat", "t"]), "nodes": rd("nodes", ["lon", "lat"]),
            "arcs": rd("arcs", ["u", "v"]), "route": None}  # route deliberately not loaded here


def load_reference(d: Path):
    f = Path(d) / f"{Path(d).name}.route"
    return np.loadtxt(f, dtype=np.int64, ndmin=1) if f.exists() else None


def predict_trip(trip: dict, model, beta=HMM_CFG["beta"]):
    t0 = time.time()
    assert trip.get("route") is None, "prediction must not see the reference route"
    net = Network(trip["nodes"], trip["arcs"])
    frame, info = trip_frame(trip, with_labels=False, net=net)
    frame["score"] = model.predict_proba(frame[FEATURES])[:, 1]
    tr = clean_track(trip["track"], trip["nodes"]); px, py = net.proj(tr.lon.values, tr.lat.values)
    clf = frame.loc[frame.groupby("point").score.idxmax(), ["point", "arc"]].rename(columns={"arc": "pred_arc"})
    path, breaks = viterbi(frame, px, py, Router(net), beta=beta)
    return {"net": net, "track": tr, "px": px, "py": py, "classifier": clf.reset_index(drop=True), "hmm": path,
            "breaks": breaks, "n_points": len(tr), "n_no_candidate": info["n_no_cand"], "secs": time.time() - t0}


def score_trip(res, route):
    """Server-side evaluation against the reference route (after prediction)."""
    lab, _ = label_points(res["net"], res["px"], res["py"], route)
    out = {}
    for k in ("classifier", "hmm"):
        p = res[k]
        out[k] = {"top1_accuracy": float((p.pred_arc.values == lab[p.point.values]).mean()),
                  "coverage": len(p) / res["n_points"],
                  "route_continuity": route_continuity(p.pred_arc.values, res["net"]),
                  **sequence_metrics(p.pred_arc.values, route)}
    return out


def arcs_to_lines(arc_ids, nodes, arcs):
    """Ordered unique arcs -> list of [[lat,lon],[lat,lon]] segments for map drawing."""
    seq = list(dict.fromkeys(int(a) for a in arc_ids))
    u, v = arcs.u.values[seq], arcs.v.values[seq]
    return [[[nodes.lat.values[a], nodes.lon.values[a]], [nodes.lat.values[b], nodes.lon.values[b]]] for a, b in zip(u, v)], seq
