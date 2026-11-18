"""PathSnap API (Laptop 1). Owner: Vedant Sharma.
Run: uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
Evaluation trips folder: env PATHSNAP_EVAL_DIR (default data/student_test for development).
The reference route is read only AFTER prediction, to compute accuracy, and is never sent to the UI."""
import os, time, logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.pipeline import ROOT, load_model, load_trip_dir, load_reference, predict_trip, score_trip, arcs_to_lines, HMM_CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PathSnap] %(message)s")
log = logging.getLogger("pathsnap")

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "student_test"))
MODEL, MODEL_FILE = load_model()
log.info("loaded model %s | eval dir %s | HMM %s", MODEL_FILE, EVAL_DIR, HMM_CFG)

app = FastAPI(title="PathSnap map-matching API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_FILE, "eval_dir": str(EVAL_DIR)}


@app.get("/trips")
def trips():
    ids = sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir())
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    d = EVAL_DIR / req.trip_id
    if not d.is_dir() or "/" in req.trip_id or ".." in req.trip_id:
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    t0 = time.time()
    log.info("POST /predict trip=%s : running pipeline", req.trip_id)
    trip = load_trip_dir(d)                       # no route
    res = predict_trip(trip, MODEL)
    metrics = None
    ref = load_reference(d)                       # server-side only, after prediction
    if ref is not None:
        metrics = score_trip(res, ref)
    lines, seq = arcs_to_lines(res["hmm"].pred_arc.values, trip["nodes"], trip["arcs"])
    clf_lines, _ = arcs_to_lines(res["classifier"].pred_arc.values, trip["nodes"], trip["arcs"])
    tr = res["track"]
    secs = time.time() - t0
    if metrics:
        log.info("trip=%s points=%d | classifier top1=%.4f | HMM top1=%.4f continuity=%.4f route_f1=%.4f | %.2fs",
                 req.trip_id, res["n_points"], metrics["classifier"]["top1_accuracy"], metrics["hmm"]["top1_accuracy"],
                 metrics["hmm"]["route_continuity"], metrics["hmm"]["route_f1"], secs)
    return {"trip_id": req.trip_id, "n_points": res["n_points"], "n_no_candidate": res["n_no_candidate"],
            "hmm_breaks": res["breaks"], "processing_time_s": round(secs, 2), "model": MODEL_FILE,
            "gps": tr[["lat", "lon"]].values[::2].round(6).tolist(),
            "predicted_route": lines, "classifier_route": clf_lines, "predicted_arc_ids": seq,
            "metrics": metrics}


# optional: serve the UI from this laptop too (backup if laptop 2 fails)
if (ROOT / "ui").is_dir():
    app.mount("/ui", StaticFiles(directory=ROOT / "ui", html=True), name="ui")
