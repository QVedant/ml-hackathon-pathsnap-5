from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

R = Path(__file__).resolve().parent
P = R.parent / "results" / "plots"
S = R / "screenshots"
FONT = "Quicksand Medium"
BLACK = RGBColor(0, 0, 0)

doc = Document()
for sec in doc.sections:
    sec.left_margin = sec.right_margin = Cm(2)
    sec.top_margin = sec.bottom_margin = Cm(1.8)


def style_font(st, size=None, bold=None):
    st.font.name = FONT
    st.font.color.rgb = BLACK
    if size:
        st.font.size = Pt(size)
    if bold is not None:
        st.font.bold = bold
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONT)
    for t in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(t)) is not None:
            del rf.attrib[qn(t)]
    ppr = st.element.find(qn("w:pPr"))
    if ppr is not None:
        b = ppr.find(qn("w:pBdr"))
        if b is not None:
            ppr.remove(b)


style_font(doc.styles["Normal"], 10.5)
style_font(doc.styles["Title"], 20, True)
for i, sz in ((1, 15), (2, 12.5), (3, 11)):
    style_font(doc.styles[f"Heading {i}"], sz, True)
    doc.styles[f"Heading {i}"].font.italic = False
style_font(doc.styles["List Bullet"])
style_font(doc.styles["List Number"])
style_font(doc.styles["Table Grid"], 9)


def h(text, level):
    doc.add_heading(text, level)


def p(text):
    doc.add_paragraph(text)


def bullets(items):
    for t in items:
        doc.add_paragraph(t, style="List Bullet")


def numbered(items):
    for t in items:
        doc.add_paragraph(t, style="List Number")


def table(rows):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            run = c.paragraphs[0].add_run(str(val))
            run.font.name = FONT
            run.font.size = Pt(8.5)
            run.font.color.rgb = BLACK
            run.bold = i == 0
    doc.add_paragraph()


fig_no = 0
shot_no = 0


def figure(path, text, level, width=16.5):
    global fig_no
    fig_no += 1
    h(f"Figure {fig_no}: {text}", level)
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def screenshot(path, text, level, width=16.5):
    global shot_no
    shot_no += 1
    h(f"Screenshot {shot_no}: {text}", level)
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


# Title block
doc.add_heading("PathSnap: GPS Map-Matching with a Candidate Classifier and HMM Viterbi Decoding", 0)
table([
    ["Name", "SRN", "Section"],
    ["Venkata Srivaibhav Nukaraju (Vaibhav)", "PES2UG24CS579", "J"],
    ["Venkatesh", "SRN_VENKATESH", "SECTION"],
    ["Vedant Sharma", "SRN_VEDANT_SHARMA", "SECTION"],
    ["Vedant Srivastava", "SRN_VEDANT_SRIVASTAVA", "SECTION"],
])
p("Team 5. Course: UE24CS352A Machine Learning, PES University, RR Campus. PathSnap Hackathon, 18 November 2026.")
p("Repository: github.com/QVedant/ml-hackathon-pathsnap-5")

# 2.2
h("2.2 Dataset Understanding and Exploratory Data Analysis", 1)
p("The dataset has three partitions. The train partition has 80 trips and serves fitting, cross-validation and Bayesian optimisation. "
  "The student_test partition has 10 trips and serves model comparison and decoder tuning. "
  "The 10 evaluator trips stay outside development. All exploratory analysis uses the train partition unless a table names another partition.")

h("Raw files", 2)
table([
    ["File", "Columns", "Meaning"],
    [".track", "longitude, latitude, timestamp (s)", "Noisy GPS fixes of the vehicle"],
    [".nodes", "longitude, latitude", "Road-network nodes; the row index is the node id"],
    [".arcs", "from_node, to_node", "Directed road segments; the row index is the arc id. A two-way street has two arcs."],
    [".route", "arc_id", "Ordered reference arcs. The file has no timestamps, so one arc covers many GPS points."],
])
p("Each trip has its own road network. A GPS row does not match a route row. Per-point labels therefore come from an alignment step.")

h("Inventory and missing-value analysis", 2)
table([
    ["Partition", "Trips", "GPS points", "Points per trip (mean / max)", "Arcs per trip", "Missing values", "Duplicate timestamps", "Out-of-order timestamps", "Gaps over 5 s"],
    ["train", "80", "208,417", "2,605 / 19,313", "649 to 307,392", "0", "0", "0", "1,216"],
    ["student_test", "10", "19,160", "1,916 / 3,958", "up to 56,027", "0", "0", "0", "44"],
])
bullets([
    "94.3% of consecutive fixes are exactly 1 s apart. The longest train gap is 153,064 s.",
    "Every route arc id exists in the matching arc file.",
    "Trips cover the whole world: longitude from -100 to 138 degrees, latitude from -35 to 62 degrees.",
    "Each network has 2 to 175 weakly connected components. The largest component holds 97% of nodes on average. No node is isolated.",
    "About 95% of arcs have a reverse twin, so most streets are two-way.",
    "In trips 33 and 54 the GPS lies about 150 m from the reference route for the whole trip. Their labels are unreliable.",
])
figure(P / "eda_sampling.png", "Sampling interval, points per trip and route length (train and student_test)", 3)
figure(P / "eda_extent.png", "GPS trajectories and GPS density for trip 00000099 (train)", 3, 14)
figure(P / "folium_eda_trip_00000099.png", "Folium map of trip 00000099 with GPS, reference route and road network (train)", 3, 14.5)

h("Preprocessing", 2)
numbered([
    "Cleaning sorts each track by timestamp. It drops missing rows, duplicate timestamps and points more than about 1 km outside the network bounding box.",
    "Projection converts longitude and latitude to metres. Each trip uses the UTM zone at the median of its road nodes. The error inside one zone is below 0.1%. A single fixed zone gave distance errors of 14% to 57% on distant trips.",
    "A 5-point rolling median smooths positions for heading and speed only. Distances use raw positions.",
    "Candidate generation samples points every 15 m along each arc into a KD-tree. It keeps the 8 nearest arcs within 50 m of each GPS point.",
    "Labelling aligns the GPS sequence to the ordered route by dynamic programming. The route index can stay or advance by up to 20 arcs per fix. The alignment minimises the total point-to-arc distance.",
    "A quality filter removes train trips with a median GPS-to-route distance above 50 m. Trips 33 and 54 fail this test.",
    "A standard scaler, fitted on train only, scales the features for Logistic Regression, KNN and MLP.",
])
p("The candidate miss rate is the share of GPS points whose reference arc is not a candidate. It is 1.85% on clean train trips and 0.41% on student_test. "
  "This value limits the accuracy of every classifier. Train gives 1,084,912 candidate rows, and 16.9% of them are positive.")

h("Patterns and features", 2)
p("Each row is one GPS point paired with one candidate arc. The target says whether the arc is the reference arc. "
  "Arc ids differ between trips, so they cannot be classes. Geometric features of a candidate transfer across trips. "
  "The predicted probability per candidate is also the emission score of the HMM.")
p("Exploration shows GPS noise of a few to tens of metres. Two-way streets and parallel roads create near ties in distance. "
  "Heading is noisy when the vehicle is slow. The thirteen features below follow from these patterns.")
table([
    ["Feature", "Reason", "Gain share in final LightGBM"],
    ["heading_diff", "GPS bearing against arc bearing, 0 to 180 degrees. It separates the two directions of a street and parallel roads.", "26.6%"],
    ["dist_rank", "Rank of the candidate by distance. Relative closeness matters in dense areas.", "19.9%"],
    ["dist_gap_nearest", "Extra distance compared with the nearest candidate, in metres", "19.6%"],
    ["off_end", "The projection falls beyond the arc ends, so the point likely belongs to a neighbour arc", "11.5%"],
    ["arc_length", "Short arcs at junctions are easy to confuse", "6.0%"],
    ["proj_frac", "Position of the projection along the arc, 0 to 1", "4.7%"],
    ["dist", "Perpendicular distance from the GPS point to the arc, in metres", "3.3%"],
    ["speed", "Distance from the previous fix divided by the time step. Low speed makes heading unreliable.", "2.5%"],
    ["conn_prev_nearest", "The arc continues from the nearest arc of the previous point", "2.3%"],
    ["in_deg, out_deg", "Number of arcs entering and leaving the arc's end nodes", "1.4%, 1.2%"],
    ["n_cand", "Number of candidate arcs within 50 m", "0.9%"],
    ["dt", "Seconds since the previous fix", "0.2%"],
])
p("Raw longitude, latitude and timestamp are not features, because they describe where and when a trip happens. Trip, point and arc ids are not features either. "
  "No engineered feature is dropped after training. The value dt carries little gain because 94% of time steps equal 1 s.")

# 3
h("3. Model Experimentation and Comparison", 1)
p("Five-fold GroupKFold by trip runs on the 78 clean train trips. Grouping stops near-duplicate points of one trip from appearing in training and validation together. "
  "Each model is then refitted on all train data and scored on student_test. Class weights handle the 16.9% positive rate. The random seed is 42.")
p("The primary metric is observation-level Top-1 accuracy on student_test. For each GPS point, the candidate with the highest probability is compared with the reference arc. "
  "The route depends on this choice. Macro-F1 at threshold 0.5 and log loss are secondary metrics. Log loss matters because the HMM uses the probabilities.")

h("3.1 Overall Model Comparison", 2)
table([
    ["Model", "Algorithm family", "Key configuration", "CV macro-F1 (std)", "Top-1 accuracy", "Route continuity", "Student-test macro-F1", "Log loss", "Inference time (ms per 1k rows)"],
    ["Logistic Regression", "Linear, probabilistic", "C=1, L2, balanced, scaled", "0.900 (0.011)", "0.8695", "0.932", "0.891", "0.190", "0.10"],
    ["KNN", "Instance-based", "k=25, distance-weighted, 150k-row sample", "0.923 (0.012)", "0.8639", "0.931", "0.918", "0.300", "314.6"],
    ["Random Forest", "Bagging ensemble", "200 trees, depth 16, min leaf 5, balanced", "0.921 (0.013)", "0.8694", "0.930", "0.914", "0.144", "3.06"],
    ["LightGBM", "Boosting", "400 trees, lr 0.05, 63 leaves, subsample 0.8", "0.917 (0.012)", "0.8703", "0.932", "0.908", "0.150", "6.16"],
    ["MLP", "Neural network", "Layers 64 and 32, ReLU, Adam, early stopping, 400k-row sample", "0.924 (0.012)", "0.8633", "0.933", "0.918", "0.124", "0.55"],
])
p("Route continuity is the share of consecutive per-point predictions that stay on one arc or move to a connected arc. It uses each model's own Top-1 picks without the HMM. "
  "All five models output probabilities, so log loss applies to each. For KNN the probability is the distance-weighted share of neighbour votes.")

h("3.2 Questions for Every Model", 2)

h("Model 1: Logistic Regression", 3)
table([
    ["Question", "Answer"],
    ["1. Suitability", "A transparent baseline that outputs probabilities through the sigmoid. It tests how far a linear mix of distance and heading goes."],
    ["2. Strengths and limitations", "It has the fastest inference and well-behaved probabilities. It cannot model interactions, such as a close arc with the wrong heading."],
    ["5. Hyperparameters", "C = 1.0, L2 penalty, lbfgs solver, max_iter 1000, balanced class weights, standardised features"],
    ["6. Loss", "Binary cross-entropy with an L2 penalty"],
    ["7. Student-test performance", "Precision 0.739 and recall 0.930. Balanced weights push the model towards recall."],
    ["8. Top-1, macro-F1, log loss", "0.8695, 0.891, 0.190"],
    ["9. Fit", "Slight underfitting. CV macro-F1 is the lowest of all models at 0.900, with folds from 0.882 to 0.913. CV and test scores stay close, so capacity limits the model."],
])

h("Model 2: K-Nearest Neighbours", 3)
table([
    ["Question", "Answer"],
    ["1. Suitability", "Map-matching situations repeat. Similar distance, heading and junction structure should give similar labels."],
    ["2. Strengths and limitations", "It makes no distribution assumption and draws non-linear boundaries. Prediction cost grows with the training set, and the probabilities are coarse vote shares."],
    ["5. Hyperparameters", "k = 25, distance weights, Euclidean metric on standardised features, random sample of 150,000 training rows"],
    ["6. Loss", "None. KNN is a lazy learner and predicts by a distance-weighted vote."],
    ["7. Student-test performance", "Precision 0.858 and recall 0.869. Inference takes 314.6 ms per 1,000 rows, about 50 times slower than LightGBM."],
    ["8. Top-1, macro-F1, log loss", "0.8639, 0.918, 0.300"],
    ["9. Fit", "No overfitting. CV macro-F1 is 0.923 and test macro-F1 is 0.918. Its hard decisions are good, but its ranking of candidates within one point is the weakest."],
])

h("Model 3: Random Forest", 3)
table([
    ["Question", "Answer"],
    ["1. Suitability", "It learns non-linear interactions between distance, heading and topology without scaling. Averaging many trees gives stable probabilities."],
    ["2. Strengths and limitations", "It needs little tuning and gives a good log loss. Training takes 91 s, and the saved model is about 195 MB."],
    ["5. Hyperparameters", "200 trees, max_depth 16, min_samples_leaf 5, square-root feature sampling, balanced_subsample class weights"],
    ["6. Loss", "Each split minimises Gini impurity. Trees train on bootstrap samples."],
    ["7. Student-test performance", "Precision 0.802 and recall 0.925"],
    ["8. Top-1, macro-F1, log loss", "0.8694, 0.914, 0.144"],
    ["9. Fit", "No overfitting. Depth and leaf limits keep CV macro-F1 at 0.921, folds 0.902 to 0.936, and test macro-F1 at 0.914."],
])

h("Model 4: LightGBM (selected)", 3)
table([
    ["Question", "Answer"],
    ["1. Suitability", "Gradient boosting builds trees in sequence, and each tree corrects earlier errors. Histogram training stays fast on one million rows."],
    ["2. Strengths and limitations", "It has the best Top-1, a good log loss, 8 s training and a small model. Many hyperparameters interact, so it gains from tuning."],
    ["5. Hyperparameters", "n_estimators 400, learning_rate 0.05, num_leaves 63, min_child_samples 20, subsample 0.8, colsample_bytree 0.8, reg_lambda 0, is_unbalance"],
    ["6. Loss", "Binary log loss, minimised by gradient boosting"],
    ["7. Student-test performance", "Precision 0.782 and recall 0.930"],
    ["8. Top-1, macro-F1, log loss", "0.8703 (best), 0.908, 0.150"],
    ["9. Fit", "No overfitting. CV macro-F1 is 0.917, folds 0.897 to 0.930, and test macro-F1 is 0.908. Row and column sampling act as regularisation."],
])

h("Model 5: Multi-Layer Perceptron", 3)
table([
    ["Question", "Answer"],
    ["1. Suitability", "A flexible non-linear function of the 13 features with smooth probabilities. It tests a learned representation against trees."],
    ["2. Strengths and limitations", "It has the best log loss and CV macro-F1. It needs scaling, trains in 44 s on a sample and is hard to interpret."],
    ["5. Hyperparameters", "Two hidden layers with 64 and 32 neurons, ReLU, Adam with learning rate 0.001, L2 alpha 0.0001, batch 512, at most 60 epochs, early stopping on 10% validation with patience 10, 400,000-row sample"],
    ["6. Loss", "Binary cross-entropy"],
    ["7. Student-test performance", "Precision 0.870 and recall 0.856"],
    ["8. Top-1, macro-F1, log loss", "0.8633, 0.918, 0.124"],
    ["9. Fit", "No overfitting. Training loss falls from 0.150 to 0.099 over 53 epochs. Validation accuracy levels at 0.963, and early stopping ends training."],
])

h("3.3 Model Visualisations and Selection", 2)
figure(P / "cv_macro_f1.png", "CV macro-F1 per model, 5-fold GroupKFold (train)", 3, 12)
figure(P / "model_comparison.png", "Top-1, macro-F1, log loss and inference time per model (student_test)", 3)
figure(P / "confusion_matrices.png", "Confusion matrices at threshold 0.5 (student_test)", 3)
figure(P / "mlp_curve.png", "MLP training loss and validation accuracy per epoch (train)", 3, 11)
p("LightGBM is the final model. It has the highest Top-1, which is the primary metric. Its log loss of 0.150 is close to the best, so its probabilities suit the HMM. "
  "Its fold scores are stable, with a standard deviation of 0.012. It trains 11 times faster than Random Forest and is far smaller. "
  "All five models lie within 0.7 points of Top-1, so held-out results and deployment cost decide the choice. Training accuracy plays no part.")
p("MLP is second, with the best log loss and macro-F1 but a lower Top-1. KNN has slow inference and coarse probabilities, so it does not suit the decoder.")

# 4
h("4. Bayesian Hyperparameter Optimisation", 1)
p("Optuna runs the search with the Tree-structured Parzen Estimator (TPE) sampler. TPE is a Bayesian method. Grid search and random search are not used. "
  "TPE splits finished trials into a good group and a bad group by objective value. It fits a density to each group and samples where the good density is high and the bad density is low. "
  "Each new trial therefore uses the results of all earlier trials. The first 10 trials are the TPE startup phase. Trial 0 holds the untuned configuration.")
h("1 and 2. Search space", 2)
table([
    ["Hyperparameter", "Range", "Reason", "Untuned", "Tuned"],
    ["n_estimators", "100 to 1000, step 50", "Number of boosting rounds", "400", "550"],
    ["learning_rate", "0.01 to 0.2, log scale", "Step size, linked to the number of rounds", "0.05", "0.0317"],
    ["num_leaves", "15 to 255, log scale", "Tree complexity", "63", "229"],
    ["min_child_samples", "10 to 300, log scale", "Minimum rows per leaf", "20", "59"],
    ["subsample", "0.5 to 1.0", "Row sampling per tree", "0.8", "0.597"],
    ["colsample_bytree", "0.4 to 1.0", "Feature sampling per tree", "0.8", "0.524"],
    ["reg_lambda", "0.001 to 10, log scale", "L2 penalty on leaf weights", "0.001", "0.683"],
])
h("3 to 5. Objective, trials and best values", 2)
p("The objective is the mean observation-level Top-1 accuracy over 3-fold GroupKFold on the clean train trips. Student_test only serves the final comparison. Evaluator data never enters the search. "
  "The search runs 40 trials in 1,012 s on 12 CPU cores, which is 120 model fits. Trial 25 is the best, with CV Top-1 0.8844 against 0.8836 for trial 0. "
  "Optuna fANOVA gives the importance: n_estimators 35%, num_leaves 34%, learning_rate 17%, and below 6% for the rest.")
p("The search ran before the change to per-trip UTM projection. The best configuration is kept and refitted on the corrected features. "
  "Both configurations are then scored again on the corrected features with the same folds and partitions. The next table shows these scores.")
h("6. Tuned against untuned", 2)
table([
    ["Model", "CV Top-1", "CV macro-F1", "CV log loss", "Student-test Top-1", "Student-test macro-F1", "Student-test log loss", "Inference (ms per 1k rows)"],
    ["LightGBM untuned", "0.8830", "0.9174", "0.1397", "0.8707", "0.9082", "0.1501", "5.0"],
    ["LightGBM tuned", "0.8829", "0.9210", "0.1341", "0.8703", "0.9111", "0.1456", "13.6"],
])
p("Tuning gives a partial gain. Top-1 does not change beyond noise. Macro-F1 rises by 0.3 to 0.4 points, and log loss falls by about 4% on CV and student_test. "
  "The default configuration already ranks candidates well, and tuning mainly improves probability quality for the HMM. "
  "Inference becomes 2.7 times slower but stays below one second per trip. The tuned trees have more leaves but use stronger sampling and an L2 penalty. "
  "Log loss improves on held-out data, so tuning does not overfit. The deployed model is the tuned model.")
figure(P / "optuna_history.png", "Optuna TPE history, CV Top-1 per trial and best value so far (train)", 2, 12)

# 5
h("5. HMM and Viterbi Route Reconstruction", 1)
h("1. Hidden states", 2)
p("At each GPS point the hidden state is one of its candidate arcs, at most 8. Each state keeps its arc id, so the decoded path maps straight to arcs. The GPS points are the observations.")
h("2. Emission probabilities", 2)
p("The emission score of candidate j at point t is the log of the tuned LightGBM probability that j is the reference arc. Probabilities below 0.000001 are clipped.")
h("3. Transition probabilities", 2)
p("The network distance d_route runs from the projection on arc i to the projection on arc j. It uses Dijkstra shortest paths along directed arcs, with a cut-off of 1.5 km. "
  "Staying on one arc costs the movement along it, with up to 5 m of backward jitter allowed. The GPS distance d_gps is the straight line between the two fixes.")
p("The transition score is log A(i, j) = -|d_route - d_gps| / (beta x max(1, dt)). An unreachable arc gets minus infinity. "
  "The Viterbi recurrence is delta_t(j) = max over i of [delta_t-1(i) + log A(i, j)] + log B_t(j). It stores the best predecessor of each state. "
  "Backtracking from the best final state gives the route. Log space prevents numeric underflow.")
h("4. Missing candidates and GPS gaps", 2)
p("A point with no arc within 50 m is skipped. The chain links its neighbours with their true GPS distance. Beta scales with dt, so a long gap allows a longer detour.")
h("5. Disconnected road segments", 2)
p("When every transition into point t is minus infinity, the chain breaks. The decoder counts the break and restarts at t from the emission scores alone. "
  "Backtracking joins the parts at the best final state of the earlier part. Every point still receives an arc, and each part stays connected. "
  "Breaks occur 0.3 times per student_test trip on average.")
h("6. Effect of HMM decoding", 2)
table([
    ["Method (student_test, mean of 10 trips)", "Top-1", "Route continuity", "Route F1", "Breaks per trip", "Decode time per trip"],
    ["Classifier only (tuned LightGBM)", "0.8875", "0.9311", "0.8729", "-", "-"],
    ["HMM, beta 2", "0.8833", "0.9745", "0.9115", "0.3", "1.3 s"],
    ["HMM, beta 10", "0.9019", "0.9737", "0.9063", "0.3", "0.8 s"],
    ["HMM, beta 20", "0.9050", "0.9715", "0.9027", "0.3", "0.8 s"],
    ["HMM, beta 50 (deployed)", "0.9049", "0.9666", "0.8990", "0.3", "0.9 s"],
])
p("Beta 50 comes from the student_test sweep of the first pipeline version. On the corrected features, beta 20 and beta 50 tie. "
  "Top-1 here is a mean over trips, so the classifier value differs slightly from the pooled 0.8703 in Section 3.")
table([
    ["Trip", "00", "07", "09", "40", "52", "56", "65", "76", "77", "91"],
    ["Top-1, classifier", "0.945", "0.926", "0.868", "0.936", "0.966", "0.897", "0.842", "0.906", "0.710", "0.879"],
    ["Top-1, HMM", "0.956", "0.933", "0.909", "0.932", "0.996", "0.931", "0.853", "0.930", "0.717", "0.891"],
    ["Continuity, classifier", "0.965", "0.996", "0.893", "0.968", "0.946", "0.904", "0.940", "0.893", "0.919", "0.887"],
    ["Continuity, HMM", "0.993", "0.999", "0.938", "0.987", "0.962", "0.958", "0.970", "0.947", "0.964", "0.947"],
])
figure(P / "hmm_vs_classifier.png", "Top-1 and route continuity per trip, classifier against HMM (student_test)", 3)
p("Continuity improves on all 10 trips, and Top-1 improves on 9 of 10. Trip 52 rises from 96.6% to 99.6%, and trip 9 rises from 86.8% to 90.9%. "
  "The HMM removes isolated jumps onto side streets and reverse arcs, because these need impossible or long detours.")
p("The HMM does not help when the classifier is wrong over a long stretch. Trip 77 stays near 71%. Its dense grid lets the GPS run between parallel streets, so a connected wrong street fits the transitions. "
  "On trip 40, Top-1 drops from 93.6% to 93.2%, because the decoder keeps a wrong branch at one junction. A small beta favours smoothness, and a large beta trusts the classifier more.")

# 6
h("6. Geographical Visualisation Using Folium", 1)
p("Each Folium map has four layers. Raw GPS points are blue markers. The nearby road network is grey and covers arcs within 150 m of the trajectory. "
  "The reference route is a wide green line. The predicted route is a narrow red line on top. Each map has a legend, a layer control and a view centred on the trip. "
  "The repository stores interactive maps for every student_test trip, for the classifier alone and with the HMM.")
figure(P / "folium_hmm_00000052.png", "Trip 00000052 with HMM, Top-1 99.6% (student_test)", 2, 14.5)
figure(P / "folium_clf_00000077.png", "Trip 00000077, classifier only (student_test)", 2, 14)
figure(P / "folium_hmm_00000077.png", "Trip 00000077 with HMM, dense grid case (student_test)", 2, 14)

# 7
h("7. API, UI and Deployment", 1)
p("Laptop 1 runs the FastAPI backend with uvicorn on host 0.0.0.0 and port 8000. The backend loads the tuned LightGBM model once at startup. "
  "It reads trips from the folder in the PATHSNAP_EVAL_DIR variable. Laptop 2 serves the static UI page with the Python HTTP server on port 5500. "
  "The page calls the API over the local network. CORS is on, because the page and the API have different origins.")
table([
    ["Endpoint", "Purpose"],
    ["GET /health", "Status, loaded model and evaluation folder"],
    ["GET /trips", "Trip folders found in the evaluation folder. No trip list is hardcoded."],
    ["POST /predict", "Runs cleaning, candidates, features, LightGBM and Viterbi for one trip id. Returns GPS points, both routes, arc ids, timing and metrics."],
])
p("Ground truth stays on the server. The trip loader never reads the route file, and the prediction function checks that no route is present. "
  "After prediction, the server reads the route file and computes Top-1, continuity and route precision, recall and F1. The UI receives numbers only.")
p("The UI lists the trips and shows Top-1 after the HMM with its change against the classifier. Bars compare both methods on each metric. "
  "Map layers show raw GPS, the classifier route and the HMM route. Screenshot 1 and Screenshot 2 in Section 8 show the running system.")
h("Results on the evaluator trips", 2)
table([
    ["Trip", "15", "17", "23", "30", "49", "59", "67", "73", "87", "94", "Mean"],
    ["GPS points", "6854", "1346", "458", "654", "2642", "1031", "916", "1144", "1399", "3230", "1967"],
    ["Top-1, classifier", "0.735", "0.918", "0.948", "0.786", "0.986", "0.957", "0.954", "0.976", "0.942", "0.927", "0.913"],
    ["Top-1, HMM", "0.707", "0.925", "0.943", "0.841", "0.992", "0.971", "0.926", "0.979", "0.995", "0.905", "0.918"],
    ["Continuity, HMM", "0.975", "0.868", "0.974", "0.893", "0.994", "0.924", "0.987", "0.988", "0.989", "0.959", "0.955"],
    ["Server time (s)", "6.6", "3.6", "0.6", "1.4", "2.8", "2.0", "0.9", "1.6", "2.5", "3.9", "2.6"],
])
p("These results come from the API at demonstration time. No modelling decision uses them. "
  "Mean Top-1 on the evaluator trips is 91.8%, close to 90.5% on student_test, so the pipeline generalises. Continuity improves on every trip, and Top-1 improves on 6 of 10.")

# 8
h("8. Output Screenshots", 1)
screenshot(S / "api_terminal.png", "Uvicorn log on laptop 1 serving evaluator trips", 2)
screenshot(P / "ui_screenshot.png", "Evaluator UI with trip 00000067 matched", 2, 13)

# 9
h("9. Limitations and AI Assistance", 1)
p("Labels come from aligning GPS to an untimed route, so they are approximate near junctions. Arcs are straight segments. "
  "Trips 33 and 54 have inconsistent reference routes. The Bayesian search ran once, before the projection correction.")
p("An AI coding assistant (Claude) helped to write code and draft documentation. The team ran all experiments. Every number in this report comes from the files in the results folder of the repository.")

doc.save(R / "5_Report.docx")
print("saved", R / "5_Report.docx")
