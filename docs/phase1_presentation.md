# PathSnap — Phase 1 presentation notes (Team 5)
All numbers are from `results/` in this repository. Partitions: 80 train trips (fit + 5-fold GroupKFold CV by trip), 10 student-test trips (comparison and decoder tuning). Evaluator trips were not used.

---
## Slide 1 — Problem and data (Vedant Sharma)
- Input: noisy GPS `.track` (lon, lat, t). Each trip has its own road network: `.nodes` (lon, lat) and directed `.arcs` (from, to). `.route` is the ordered list of true arc ids, with no timestamps.
- A GPS row does **not** map to a route row, so we have to create per-point labels ourselves.
- Size: train has 208,417 GPS points and 3.9 M arcs in total. Networks range from 649 to 307,392 arcs. Student-test has 19,160 points.

## Slide 2 — EDA findings (Vedant Sharma) · `notebooks/01_eda.ipynb`, `maps/eda_trip_00000099.html`
- 0 missing values, 0 duplicate timestamps, 0 out-of-order timestamps.
- 94.3% of steps are exactly 1 s. There are 1,216 gaps longer than 5 s in train. Some trips have very long gaps (stops or signal loss).
- Network: about 95% of arcs are two-way (A→B and B→A both exist), so heading is essential. On average the largest weakly connected component holds 97% of nodes.
- **Data-quality issue:** in trips 33 and 54 the GPS sits about 150 m from the reference route for the whole trip. Their labels are unreliable, so they are excluded from training (rule: median GPS-to-route distance > 50 m).

## Slide 3 — Preprocessing and candidates (Venkatesh)
- Sort by t, drop NaN/duplicate timestamps, drop points more than ~1 km outside the network bounding box.
- Project lon/lat to **UTM zone 53N (EPSG:32653)**. Degrees are not metres: at 35°N, 1° of longitude is about 91 km while 1° of latitude is about 111 km.
- Candidates: sample points every 15 m along each arc, build a KD-tree, then compute the exact point-to-segment distance. Keep the **K=8 nearest arcs within R=50 m**.
- Labels: a **DP alignment** of GPS points to the ordered route. It is monotonic, so it never goes backwards along the route, and it is globally optimal. It replaced a greedy pointer, which failed on trip 8 (miss rate 41% → 2%).
- **Candidate miss rate**: 2.0% on train (clean trips) and 0.45% on student-test. That is the ceiling on Top-1.

## Slide 4 — Features and why (Venkatesh)
| Feature | Why |
|---|---|
| dist to arc | GPS noise: the true road is usually the closest |
| dist rank, gap to nearest | relative closeness matters in dense downtown areas |
| heading diff (smoothed GPS bearing vs arc bearing) | separates parallel roads and the two directions of a two-way road |
| speed, dt | heading is unreliable when stopped; gaps reduce reliability |
| projection fraction, off-end flag | a point projecting beyond the arc's end belongs to the next arc |
| arc length, in/out degree | short arcs at junctions are easy to confuse |
| n candidates | how ambiguous the area is |
| connected to previous nearest arc | cheap sequential context |
Problem framing: **binary classification per (point, candidate)**, giving P(this arc is correct). That gives us one comparable probability per candidate, which is exactly what the HMM needs as emissions. 17% of candidate rows are positive, so we use class weighting.

## Slide 5 — 5 models and the selection (Vaibhav) · `results/model_comparison.csv`, `results/plots/model_comparison.png`, `cv_macro_f1.png`, `confusion_matrices.png`, `mlp_curve.png`
| Model | Family | CV macro-F1 (±std) | Test macro-F1 | Test log loss | **Test Top-1** | Continuity | Train s | Infer ms/1k |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | linear | 0.901 ± 0.009 | 0.893 | 0.195 | 0.867 | 0.933 | 2.2 | 0.14 |
| KNN (k=25, distance-weighted) | instance | 0.924 ± 0.009 | 0.916 | 0.320 | 0.864 | 0.929 | 0.3 | 345.6 |
| Random Forest (200 trees) | bagging | 0.921 ± 0.008 | 0.914 | 0.147 | 0.870 | 0.929 | 76.7 | 3.2 |
| **LightGBM (400 trees)** | boosting | 0.917 ± 0.008 | 0.910 | 0.152 | **0.875** | 0.931 | 7.4 | 6.0 |
| MLP (64-32) | neural net | 0.925 ± 0.009 | 0.920 | **0.125** | 0.871 | 0.932 | 25.9 | 0.56 |

- **Primary metric: student-test observation-level Top-1.** The route depends on picking the right candidate *per point*. Macro-F1 at a 0.5 threshold does not measure that.
- **Final pick: LightGBM.** It has the best Top-1. Its log loss of 0.152 is good enough for HMM emissions, it trains 10× faster than RF, and CV std is low, so it is stable across trips. It is also the best fit for Bayesian tuning in Phase 2.
- MLP is a close second: best log loss and F1, but 0.4 pt lower Top-1. KNN has good F1 but poorly calibrated probabilities (log loss 0.32) and is about 60× slower at inference than LightGBM, so it is a poor fit for the decoder.
- Overfitting check: CV and test F1 are within about 0.01 for every model, so there is no overfitting. Trips never cross folds (GroupKFold).
- Training data: KNN was trained on a 150k-row subsample and MLP on 400k (cost). The other models used all about 1 M rows.

## Slide 6 — HMM and Viterbi (Venkatesh, Vedant Srivastava) · `results/hmm_summary_LightGBM.csv`, `plots/hmm_vs_classifier.png`, `maps/hmm_*.html` vs `maps/clf_*.html`
- Hidden states: the candidate arcs of each GPS point. Emission: log P from LightGBM.
- Transition: `−|d_network − d_gps| / β`, where d_network is the Dijkstra shortest path along directed arcs (cutoff 1.5 km). An unreachable arc gets −∞, so impossible jumps are forbidden. For gaps, β is multiplied by dt.
- Recurrence: δ_t(j) = max_i [δ_{t−1}(i) + log A_ij] + log B_t(j), then backtrack. If every transition is −∞, the chain breaks and restarts (0.6 breaks per trip on average).

| Student-test (mean over 10 trips) | Top-1 | Continuity | Route F1 |
|---|---|---|---|
| LightGBM only | 0.889 | 0.931 | 0.870 |
| + HMM β=10 | 0.896 | 0.973 | 0.902 |
| + HMM β=50 | **0.902** | 0.966 | 0.900 |
- The HMM mostly fixes continuity: invalid jumps drop from 6.9% to 2.7%. Pointwise gains are smaller because the classifier is already strong. β trades accuracy against smoothness. It runs in about 0.5 s per trip.
- Where it does not help: points without a correct candidate (miss rate), and labels near junctions, where the reference itself is ambiguous.

---
## Likely viva questions
- *Why UTM and not degrees?* Euclidean distance in degrees is distorted (cos lat). UTM gives metres locally.
- *Why binary per candidate instead of multiclass over arcs?* Arc ids differ per trip, so they are not shared classes. Per-candidate features generalise across trips.
- *Why GroupKFold?* Consecutive points are highly correlated. Mixing points from one trip across folds leaks information.
- *How were labels made?* By DP alignment to the ordered route. The labels are approximate, as the reference route is not verified ground truth.
- *Why not pick by training accuracy?* All models are within about 1% of each other. We pick on held-out trips and on the metric that matters for routing (Top-1), plus probability quality for the HMM.
- *How are gaps handled?* dt is a feature, β scales with dt, and the chain breaks and restarts when nothing is reachable.
- *Phase 2:* Bayesian tuning with Optuna TPE (not grid/random) for LightGBM, with GroupKFold Top-1 as the objective.
