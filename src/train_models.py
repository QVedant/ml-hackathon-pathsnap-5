"""Train + compare 5 candidate classifiers. Owner: Vaibhav.
Task: binary - is this candidate arc the true arc for this GPS point? Probabilities feed the HMM.
Run: uv run python -m src.train_models"""
import json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, confusion_matrix)
from lightgbm import LGBMClassifier
from .features import FEATURES
from .evaluate import top1_predictions, top1_accuracy, candidate_miss_rate, route_continuity
from .io import load_trip
from .candidates import Network

SEED = 42
MAX_ROUTE_OFFSET_M = 50.0
rng = np.random.default_rng(SEED)

# name -> (family, factory, max training rows (None = all), config text)
MODELS = {
    "LogReg": ("Linear / probabilistic",
               lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
               None, "C=1, balanced, L2"),
    "KNN": ("Instance-based",
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
            150_000, "k=25, distance-weighted, scaled, 150k-row subsample"),
    "RandomForest": ("Ensemble (bagging)",
                     lambda: RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=5,
                                                    class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
                     None, "200 trees, depth 16, leaf 5, balanced"),
    "LightGBM": ("Boosting",
                 lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                        subsample_freq=1, colsample_bytree=0.8, is_unbalance=True,
                                        random_state=SEED, verbose=-1),
                 None, "400 trees, lr 0.05, 63 leaves, unbalance-aware"),
    "MLP": ("Neural network",
            lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                                  batch_size=512, max_iter=60, early_stopping=True,
                                                                  random_state=SEED)),
            400_000, "64-32 ReLU, adam, early stopping, 400k-row subsample"),
}


def subsample(df, n):
    return df if n is None or len(df) <= n else df.sample(n, random_state=SEED)


def main():
    tr = pd.read_parquet("results/features_train.parquet")
    # data-quality filter: trips whose GPS lies far from its own reference route (bad labels)
    q = pd.read_csv("results/candidates_train.csv", dtype={"trip": str})
    bad = q.loc[q.median_gps_to_route_m > MAX_ROUTE_OFFSET_M, "trip"].tolist()
    print("excluded train trips (median GPS-route offset > 50 m):", bad)
    tr = tr[~tr.trip.isin(bad)].reset_index(drop=True)
    te = pd.read_parquet("results/features_test.parquet")
    nets = {t: Network(**{k: v for k, v in load_trip(t, "test").items() if k in ("nodes", "arcs")})
            for t in te.trip.unique()}
    gkf = GroupKFold(n_splits=5)
    rows, cv_log = [], []
    for name, (family, make, cap, cfg) in MODELS.items():
        print(f"== {name}", flush=True)
        f1s, lls = [], []
        for k, (a, b) in enumerate(gkf.split(tr, tr.label, tr.trip)):
            fit = subsample(tr.iloc[a], cap)
            m = make().fit(fit[FEATURES], fit.label)
            p = m.predict_proba(tr.iloc[b][FEATURES])[:, 1]
            f1s.append(f1_score(tr.label.iloc[b], p > 0.5, average="macro"))
            lls.append(log_loss(tr.label.iloc[b], p))
            cv_log.append({"model": name, "fold": k, "macro_f1": f1s[-1], "log_loss": lls[-1]})
        fit = subsample(tr, cap)
        t0 = time.time(); m = make().fit(fit[FEATURES], fit.label); t_train = time.time() - t0
        t0 = time.time(); p = m.predict_proba(te[FEATURES])[:, 1]; t_inf = time.time() - t0
        yhat = (p > 0.5).astype(int)
        s = te.assign(score=p)
        top = top1_predictions(s)
        cont = np.mean([route_continuity(g.pred_arc.values, nets[t]) for t, g in top.groupby("trip")])
        cm = confusion_matrix(te.label, yhat)
        rows.append({"model": name, "family": family, "config": cfg,
                     "cv_macro_f1": np.mean(f1s), "cv_macro_f1_std": np.std(f1s), "cv_log_loss": np.mean(lls),
                     "test_accuracy": accuracy_score(te.label, yhat),
                     "test_precision": precision_score(te.label, yhat), "test_recall": recall_score(te.label, yhat),
                     "test_macro_f1": f1_score(te.label, yhat, average="macro"),
                     "test_log_loss": log_loss(te.label, p), "test_top1": top1_accuracy(top),
                     "route_continuity": cont, "train_time_s": t_train,
                     "infer_time_ms_per_1k_rows": 1000 * t_inf / len(te) * 1000,
                     "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]})
        print(rows[-1], flush=True)
        s[["trip", "point", "arc", "score"]].to_parquet(f"results/test_scores_{name}.parquet")
        joblib.dump(m, f"models/{name}.joblib")
        if name == "MLP":
            mlp = m[-1]
            pd.DataFrame({"train_loss": mlp.loss_curve_,
                          "val_score": mlp.validation_scores_}).to_csv("results/mlp_curve.csv", index_label="epoch")
    res = pd.DataFrame(rows)
    res.to_csv("results/model_comparison.csv", index=False)
    pd.DataFrame(cv_log).to_csv("results/cv_folds.csv", index=False)
    print(f"miss rate (test): {candidate_miss_rate(te):.4f}")
    print(res[["model", "cv_macro_f1", "test_macro_f1", "test_log_loss", "test_top1", "route_continuity"]])


if __name__ == "__main__":
    main()
