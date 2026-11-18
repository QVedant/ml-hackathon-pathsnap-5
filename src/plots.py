"""Comparison plots from real experiment csvs. Owner: Ved. Run: uv run python -m src.plots"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

R = "results/"; P = "results/plots/"
res = pd.read_csv(R + "model_comparison.csv")
folds = pd.read_csv(R + "cv_folds.csv")

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(folds, x="model", y="macro_f1", ax=ax); ax.set_title("5-fold GroupKFold macro-F1 (train)")
plt.tight_layout(); plt.savefig(P + "cv_macro_f1.png")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
res.plot.bar(x="model", y=["test_top1", "test_macro_f1"], ax=ax[0], rot=20); ax[0].set_ylim(0.5, 1); ax[0].set_title("Student-test Top-1 / macro-F1")
res.plot.bar(x="model", y=["cv_log_loss", "test_log_loss"], ax=ax[1], rot=20); ax[1].set_title("Log loss (lower better)")
res.plot.bar(x="model", y="infer_time_ms_per_1k_rows", ax=ax[2], rot=20, legend=False); ax[2].set_title("Inference ms / 1k rows")
plt.tight_layout(); plt.savefig(P + "model_comparison.png")

fig, axs = plt.subplots(1, len(res), figsize=(4 * len(res), 3.5))
for a, (_, r) in zip(axs, res.iterrows()):
    cm = [[r.tn, r.fp], [r.fn, r.tp]]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"]); a.set_title(r.model)
plt.tight_layout(); plt.savefig(P + "confusion_matrices.png")

try:
    c = pd.read_csv(R + "mlp_curve.csv")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(c.train_loss, label="train loss")
    ax2 = ax.twinx(); ax2.plot(c.val_score, "r", label="val accuracy"); ax.set_xlabel("epoch")
    ax.set_title("MLP training history"); fig.legend(); plt.tight_layout(); plt.savefig(P + "mlp_curve.png")
except FileNotFoundError:
    pass

import glob
for f in glob.glob(R + "hmm_per_trip_*.csv"):
    h = pd.read_csv(f)
    best = h[h.method == "hmm"].groupby("beta").top1.mean().idxmax()
    d = pd.concat([h[h.method == "classifier_only"].assign(m="classifier only"),
                   h[(h.method == "hmm") & (h.beta == best)].assign(m=f"classifier + HMM (beta={best})")])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    sns.barplot(d, x="trip", y="top1", hue="m", ax=ax[0]); ax[0].set_title("Top-1 per student-test trip"); ax[0].tick_params(axis="x", rotation=45)
    sns.barplot(d, x="trip", y="continuity", hue="m", ax=ax[1]); ax[1].set_title("Route continuity"); ax[1].tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(P + "hmm_vs_classifier.png")
print("plots written")
