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

## Layout
| Path | Content | Owner |
|---|---|---|
| `src/io.py`, `src/geo.py` | loading, cleaning, UTM-53N projection | Venkatesh |
| `src/candidates.py` | road network, KD-tree candidate generation (K=8, R=50 m), point→route labels | Venkatesh |
| `src/features.py` | per (point, candidate) features | Venkatesh |
| `src/hmm.py`, `src/run_hmm.py` | HMM + Viterbi decoding and evaluation | Venkatesh |
| `notebooks/01_eda.ipynb`, `src/viz.py` | EDA, Folium maps | Vedant Srivastava |
| `src/train_models.py` | LogReg, KNN, RandomForest, LightGBM, MLP | Vaibhav |
| `src/evaluate.py`, `src/plots.py` | metrics, plots | Vedant Sharma |
| `results/` | CSV results, plots · `maps/` Folium HTML · `models/` saved models | |

Data partitions: 80 train trips (fit + GroupKFold CV), 10 student_test trips (model comparison, decoder tuning). Evaluator trips are never used in development.
