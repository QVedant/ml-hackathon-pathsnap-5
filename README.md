# PathSnap — Team 5 (GPS map-matching)

Noisy 1 Hz GPS → road arcs. Pipeline: **candidate arcs per GPS point → ML classifier scores each candidate → HMM/Viterbi picks a connected route.**

## Setup (uv)
```bash
uv sync                                   # Python 3.12 + pinned deps from uv.lock
ln -s "/path/to/Student folder" data      # data/train/<trip>/..., data/student_test/<trip>/...
```

## Run (Phase 1)
```bash
uv run python -m src.features             # preprocessing + candidates + features -> results/features_*.parquet
uv run python -m src.train_models         # 5 models, GroupKFold CV + student-test -> results/model_comparison.csv
uv run python -m src.run_hmm LightGBM     # classifier-only vs classifier+HMM, beta sweep, Folium maps
uv run python -m src.plots                # comparison plots -> results/plots/
uv run jupyter lab notebooks/01_eda.ipynb # EDA
```

## Phase 2
```bash
uv run python -m src.tune 40              # Bayesian optimisation (Optuna TPE, GroupKFold by trip) -> models/final_lgbm.joblib
```

## Demo deployment (two laptops, same Wi-Fi/hotspot)
**Laptop 1 — model + API**
```bash
uv sync
# evaluator trips folder: one sub-folder per trip (<id>/<id>.track/.nodes/.arcs[/.route])
export PATHSNAP_EVAL_DIR=/path/to/evaluator_trips      # default: data/student_test
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000  # keep this terminal visible: request logs
ip -4 addr | grep inet                                  # note the LAN IP, e.g. 192.168.1.20
# if blocked: sudo ufw allow 8000   (Windows: allow python in firewall prompt)
```
**Laptop 2 — UI** (no Python packages needed; Leaflet is bundled)
```bash
cd ui && python3 -m http.server 5500
# open http://localhost:5500/?api=http://<laptop1-ip>:8000
```
Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`. The reference route is read on the server only after prediction, to compute accuracy, and is never returned to the UI.
Backup: if laptop 2 fails, the UI is also served by the API at `http://<laptop1-ip>:8000/ui/`.

## Layout
| Path | Content | Owner |
|---|---|---|
| `src/io.py`, `src/geo.py` | loading, cleaning, UTM-53N projection | Venkatesh |
| `src/candidates.py` | road network, KD-tree candidate generation (K=8, R=50 m), point→route labels | Venkatesh |
| `src/features.py` | per (point, candidate) features | Venkatesh |
| `src/hmm.py`, `src/run_hmm.py`, `src/pipeline.py` | HMM + Viterbi, end-to-end inference | Venkatesh |
| `src/tune.py` | Optuna TPE tuning | Vaibhav |
| `api/main.py` | FastAPI backend | Vedant Sharma |
| `ui/index.html` | evaluator UI | Vedant Srivastava |
| `notebooks/01_eda.ipynb`, `src/viz.py` | EDA, Folium maps | Vedant Sharma |
| `src/train_models.py` | LogReg, KNN, RandomForest, LightGBM, MLP | Vaibhav |
| `src/evaluate.py`, `src/plots.py` | metrics, plots | Vedant Srivastava |
| `results/` | CSV results, plots · `maps/` Folium HTML · `models/` saved models | |

Data partitions: 80 train trips (fit + GroupKFold CV), 10 student_test trips (model comparison, decoder tuning). Evaluator trips are never used in development.
