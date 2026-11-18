"""Bayesian hyperparameter optimisation of LightGBM with Optuna TPE (not grid / random). Owner: Vaibhav.
Objective: mean observation-level Top-1 accuracy over 3-fold GroupKFold on train trips (trips never split).
Run: uv run python -m hyperparameter_optimisation.tune [n_trials]"""
import sys, time, json
import numpy as np
import pandas as pd
import optuna, joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from lightgbm import LGBMClassifier
from feature_engineering.features import FEATURES
from model_evaluation.evaluate import top1_predictions, top1_accuracy
from model_training.train_models import MAX_ROUTE_OFFSET_M, SEED

tr = pd.read_parquet("results/features_train.parquet")
q = pd.read_csv("results/candidates_train.csv", dtype={"trip": str})
tr = tr[~tr.trip.isin(q.loc[q.median_gps_to_route_m > MAX_ROUTE_OFFSET_M, "trip"])].reset_index(drop=True)
te = pd.read_parquet("results/features_test.parquet")
FOLDS = list(GroupKFold(n_splits=3).split(tr, tr.label, tr.trip))


def make(p):
    return LGBMClassifier(**p, subsample_freq=1, is_unbalance=True, random_state=SEED, verbose=-1, n_jobs=-1)


def cv_eval(p):
    top, f1, ll = [], [], []
    for a, b in FOLDS:
        m = make(p).fit(tr.iloc[a][FEATURES], tr.label.iloc[a])
        v = tr.iloc[b]; pr = m.predict_proba(v[FEATURES])[:, 1]
        top.append(top1_accuracy(top1_predictions(v.assign(score=pr))))
        f1.append(f1_score(v.label, pr > 0.5, average="macro")); ll.append(log_loss(v.label, pr))
    return np.mean(top), np.mean(f1), np.mean(ll)


def objective(trial):
    p = {"n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
         "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
         "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}
    t, f, l = cv_eval(p)
    trial.set_user_attr("cv_macro_f1", f); trial.set_user_attr("cv_log_loss", l)
    return t


def test_eval(p):
    t0 = time.time(); m = make(p).fit(tr[FEATURES], tr.label); tt = time.time() - t0
    t0 = time.time(); pr = m.predict_proba(te[FEATURES])[:, 1]; ti = time.time() - t0
    return m, pr, {"test_top1": top1_accuracy(top1_predictions(te.assign(score=pr))),
                   "test_macro_f1": f1_score(te.label, pr > 0.5, average="macro"),
                   "test_log_loss": log_loss(te.label, pr), "train_time_s": tt,
                   "infer_time_ms_per_1k_rows": 1e6 * ti / len(te)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    base = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1e-3)  # ~= LightGBM default 0 (log-scale lower bound)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                study_name="lgbm_tpe")
    study.enqueue_trial(base)  # trial 0 = untuned config, for a fair comparison
    t0 = time.time(); study.optimize(objective, n_trials=n, show_progress_bar=False); cost = time.time() - t0
    study.trials_dataframe().to_csv("results/optuna_trials.csv", index=False)
    best = study.best_params
    rows = []
    for name, p in [("untuned", base), ("tuned", best)]:
        t, f, l = cv_eval(p)
        m, pr, r = test_eval(p)
        rows.append({"model": f"LightGBM_{name}", "cv_top1": t, "cv_macro_f1": f, "cv_log_loss": l, **r, "params": json.dumps(p)})
        if name == "tuned":
            joblib.dump(m, "models/final_lgbm.joblib")
            te.assign(score=pr)[["trip", "point", "arc", "score"]].to_parquet("results/test_scores_LightGBM_tuned.parquet")
    pd.DataFrame(rows).to_csv("results/tuning_comparison.csv", index=False)
    json.dump({"best_params": best, "best_cv_top1": study.best_value, "n_trials": n,
               "sampler": "TPE", "cv": "GroupKFold(3) by trip", "search_time_s": cost},
              open("models/final_config.json", "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        v = study.trials_dataframe().value
        plt.figure(figsize=(7, 4)); plt.plot(v, "o", alpha=.5, label="trial"); plt.plot(v.cummax(), label="best so far")
        plt.xlabel("trial"); plt.ylabel("CV Top-1"); plt.title("Optuna TPE optimisation history"); plt.legend()
        plt.tight_layout(); plt.savefig("results/plots/optuna_history.png")
        imp = optuna.importance.get_param_importances(study)
        pd.Series(imp).to_csv("results/optuna_param_importance.csv")
    except Exception as e:
        print("plot error", e)
    print(pd.DataFrame(rows).drop(columns="params").round(4).to_string())
