# PathSnap — Team 5 (GPS map-matching)

Noisy GPS trajectories → the road segments actually driven.
Pipeline: **candidate arcs per GPS point → LightGBM scores each candidate → HMM/Viterbi decodes a connected route**, served by a FastAPI backend (laptop 1) to an evaluator UI (laptop 2).

## Repository structure
| Folder | Guideline component | Contents |
|---|---|---|
| `data_processing/` | Data processing | `io.py` read/clean raw `.track/.nodes/.arcs/.route`; `geo.py` per-trip UTM projection; `candidates.py` road network, KD-tree candidate generation (K=8, R=50 m), point→route labelling; `01_eda.ipynb` dataset exploration |
| `feature_engineering/` | Feature engineering | `features.py` 13 features per (GPS point, candidate arc) |
| `model_training/` | Model training | `train_models.py` LogReg, KNN, RandomForest, LightGBM, MLP with GroupKFold CV |
| `model_evaluation/` | Model evaluation | `evaluate.py` metrics (Top-1, macro-F1, log loss, miss rate, continuity, route F1); `plots.py` comparison plots, confusion matrices |
| `hyperparameter_optimisation/` | Hyperparameter optimisation | `tune.py` Bayesian optimisation with Optuna TPE |
| `hmm_viterbi/` | HMM and Viterbi | `hmm.py` decoder; `run_hmm.py` classifier-only vs classifier+HMM, β sweep |
| `api/` | API | `main.py` FastAPI endpoints; `pipeline.py` model loading and end-to-end inference |
| `ui/` | UI | `index.html` evaluator interface (Leaflet and fonts bundled in `ui/vendor/`) |
| `deployment/` | Deployment | `requirements.txt`, `run_api.sh/.bat`, `run_ui.sh/.bat`, two-laptop guide |
| `visualisations/` | Visualisations | `viz.py` Folium map generation; `maps/` generated interactive maps |
| `results/` | Supporting results | CSVs (model comparison, CV folds, Optuna trials, HMM results), logs, `plots/` |
| `models/` | Model and configuration artifacts | `final_lgbm.joblib`, `final_config.json`, baseline models |

## Setup
```bash
uv sync                                   # Python 3.12 + pinned dependencies (pyproject.toml, uv.lock)
ln -s "/path/to/Student folder" data      # data/train/<trip>/..., data/student_test/<trip>/...
```
Without `uv`: `pip install -r deployment/requirements.txt`.

## Reproduce the experiments (run from the repo root)
```bash
uv run python -m feature_engineering.features            # preprocessing + candidates + features -> results/features_*.parquet
uv run python -m model_training.train_models             # 5 models, GroupKFold CV + student-test -> results/model_comparison.csv
uv run python -m hyperparameter_optimisation.tune 40     # Optuna TPE, 40 trials -> models/final_lgbm.joblib, results/optuna_trials.csv
uv run python -m hmm_viterbi.run_hmm LightGBM_tuned      # classifier-only vs +HMM, Folium maps -> visualisations/maps/
uv run python -m model_evaluation.plots                  # plots -> results/plots/
uv run jupyter lab data_processing/01_eda.ipynb          # EDA
```

## Run the demo
See [`deployment/README.md`](deployment/README.md). Short version:
```bash
deployment/run_api.sh /path/to/evaluator_test            # laptop 1
deployment/run_ui.sh                                     # laptop 2, then open http://localhost:5500 and enter laptop 1's IP
```

## Data partitions
80 train trips (fitting, GroupKFold CV, Bayesian optimisation), 10 student-test trips (model comparison, decoder β). Evaluator trips are used only by the running API at demo time. Their reference routes are read on the server after prediction to compute accuracy and never feed the model or the UI.

## Team
| Member | Phase 1 | Phase 2 |
|---|---|---|
| Venkatesh | preprocessing, EDA | HMM/Viterbi, inference pipeline |
| Vedant Sharma | feature engineering | FastAPI backend |
| Vedant Srivastava | model selection (the 5 models) | evaluator UI, maps |
| Vaibhav | model comparison and final pick | Bayesian hyperparameter optimisation |
