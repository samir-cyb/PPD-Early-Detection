"""
07b_advanced_modeling.py
==========================
Directly answers Samir's question: "can hyperparameter tuning, a stacking/
hybrid ensemble, or a deep-learning (MLP) model meaningfully beat the
Step 7 baseline results?"

For each of the 3 targets (EPDS Result / PHQ9 Result / PPD_binary), this
script:

  1. TUNES Random Forest, XGBoost, LightGBM, CatBoost with
     RandomizedSearchCV (stratified 5-fold CV, scoring=f1_macro, n_iter=20
     -- deliberately modest so this finishes in minutes, not hours, on a
     n=800 dataset).
  2. Trains a Multi-Layer Perceptron (sklearn MLPClassifier) as the deep-
     learning comparison point, trying 3 small architectures x 3
     regularization strengths and keeping the best by CV. NOTE: with only
     n=800 samples, a large/deep network is NOT expected to beat gradient
     boosting -- this is well-established for small tabular datasets, and
     was flagged back at the very start of this project. This step exists
     so the thesis can report an honest "we tried DL, here's what
     happened" comparison, not because we expect it to win.
  3. Builds a STACKING ensemble (a genuine hybrid model) combining the 4
     tuned tree models, with Logistic Regression as the meta-learner.
  4. Reloads the Step 7 baseline results (untuned models) from
     model_performance_<target>.csv and merges them into the SAME
     comparison table, so baseline vs tuned vs MLP vs stacking sit side by
     side.

Methodological note: 08_ablation_study.py deliberately keeps modeling
simple (untuned defaults) so it isolates ONE variable -- the FEATURE SET
(raw vs raw+composite vs selected) -- without mixing in hyperparameter-
tuning effects. "Best absolute model" (this script) and "does feature
engineering help" (the ablation study) are two different questions --
keep them separate when writing up results.

Output (./outputs/):
  - best_hyperparameters.json         -- winning params per model per target
  - model_performance_advanced_<target>.csv -- baseline + tuned + MLP + stacking

Usage
-----
    pip install scikit-learn xgboost lightgbm catboost --break-system-packages
    python 07b_advanced_modeling.py

Expect this to take several minutes (not seconds) -- it's doing real
hyperparameter search across multiple model families and 3 targets.
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_validate, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")

if not os.path.exists(MODEL_READY_PATH):
    raise SystemExit(f"{MODEL_READY_PATH} not found. Run 06_encoding.py first.")

df = pd.read_csv(MODEL_READY_PATH)
print(f"Loaded model-ready dataset: {df.shape}")

TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
RANDOM_STATE = 42
N_ITER_SEARCH = 20  # RandomizedSearchCV iterations per model -- keep modest

# ---------------------------------------------------------------------------
# Optional model imports
# ---------------------------------------------------------------------------
HAS_XGB = HAS_LGBM = HAS_CAT = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("[skip] xgboost not installed -- pip install xgboost --break-system-packages")
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    print("[skip] lightgbm not installed -- pip install lightgbm --break-system-packages")
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    print("[skip] catboost not installed -- pip install catboost --break-system-packages")

# ---------------------------------------------------------------------------
# Hyperparameter search spaces (kept modest on purpose -- see module docstring)
# ---------------------------------------------------------------------------
RF_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}
XGB_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}
LGBM_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
}
CAT_PARAM_DIST = {
    "iterations": [100, 200, 300],
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
}
MLP_ARCHITECTURES = [(32,), (64, 32), (100, 50, 25)]
MLP_ALPHAS = [0.0001, 0.001, 0.01]


def tune_model(name, base_model, param_dist, X, y, skf):
    print(f"  Tuning {name} ({N_ITER_SEARCH} iterations x 5-fold CV)...")
    t0 = time.time()
    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist, n_iter=N_ITER_SEARCH,
        scoring="f1_macro", cv=skf, random_state=RANDOM_STATE, n_jobs=-1,
        refit=True)
    search.fit(X, y)
    elapsed = time.time() - t0
    print(f"  {name} best CV f1_macro={search.best_score_:.3f} "
          f"({elapsed:.1f}s) -- best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_model(name, model, X, y, X_train, X_test, y_train, y_test, skf):
    cv = cross_validate(model, X, y, cv=skf, scoring=["accuracy", "f1_macro"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc_auc = np.nan
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if len(np.unique(y)) == 2:
                roc_auc = roc_auc_score(y_test, proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, proba, multi_class="ovr",
                                         average="weighted")
    except Exception:
        pass

    # Epoch/iteration count actually used by the fitted model. Only
    # meaningful for iterative models like MLP (tree models such as Random
    # Forest/XGBoost/LightGBM/CatBoost have no notion of "epoch" -- they
    # build a fixed number of trees, n_estimators/iterations, not something
    # that "converges" over epochs -- so this stays NaN for them).
    holdout_epochs = np.nan
    try:
        clf_step = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        if hasattr(clf_step, "n_iter_"):
            holdout_epochs = int(clf_step.n_iter_)
    except Exception:
        pass

    return {
        "model": name,
        "cv_accuracy_mean": cv["test_accuracy"].mean(),
        "cv_f1_macro_mean": cv["test_f1_macro"].mean(),
        "holdout_accuracy": acc,
        "holdout_f1_macro": f1,
        "holdout_roc_auc": roc_auc,
        "holdout_epochs_used": holdout_epochs,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_best_params = {}

for target_col in TARGETS:
    print(f"\n{'=' * 90}\nTARGET: {target_col}\n{'=' * 90}")
    X = df.drop(columns=TARGETS)
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    target_best_params = {}

    # --- 1. Hyperparameter tuning ---
    print("\n--- Hyperparameter tuning ---")
    rf_best, rf_params, _ = tune_model(
        "Random Forest",
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        RF_PARAM_DIST, X, y, skf)
    target_best_params["Random Forest"] = rf_params
    rows.append(evaluate_model("Random Forest (tuned)", rf_best, X, y,
                                X_train, X_test, y_train, y_test, skf))

    tuned_tree_models = [("rf", rf_best)]

    if HAS_XGB:
        xgb_best, xgb_params, _ = tune_model(
            "XGBoost",
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            XGB_PARAM_DIST, X, y, skf)
        target_best_params["XGBoost"] = xgb_params
        rows.append(evaluate_model("XGBoost (tuned)", xgb_best, X, y,
                                    X_train, X_test, y_train, y_test, skf))
        tuned_tree_models.append(("xgb", xgb_best))

    if HAS_LGBM:
        lgbm_best, lgbm_params, _ = tune_model(
            "LightGBM",
            LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            LGBM_PARAM_DIST, X, y, skf)
        target_best_params["LightGBM"] = lgbm_params
        rows.append(evaluate_model("LightGBM (tuned)", lgbm_best, X, y,
                                    X_train, X_test, y_train, y_test, skf))
        tuned_tree_models.append(("lgbm", lgbm_best))

    if HAS_CAT:
        cat_best, cat_params, _ = tune_model(
            "CatBoost",
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=False),
            CAT_PARAM_DIST, X, y, skf)
        target_best_params["CatBoost"] = cat_params
        rows.append(evaluate_model("CatBoost (tuned)", cat_best, X, y,
                                    X_train, X_test, y_train, y_test, skf))
        tuned_tree_models.append(("cat", cat_best))

    all_best_params[target_col] = target_best_params

    # --- 2. MLP (deep learning comparison) ---
    print("\n--- MLP (deep learning comparison) ---")
    MLP_MAX_ITER = 1000  # max epochs allowed; early_stopping=True lets it
                          # stop sooner once validation score stops improving
    best_mlp = None
    best_mlp_score = -np.inf
    best_mlp_config = None
    best_mlp_epochs_per_fold = None
    best_mlp_mean_epochs = None
    for arch in MLP_ARCHITECTURES:
        for alpha in MLP_ALPHAS:
            mlp = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=arch, alpha=alpha,
                                       activation="relu", solver="adam",
                                       max_iter=MLP_MAX_ITER, early_stopping=True,
                                       random_state=RANDOM_STATE)),
            ])
            # return_estimator=True lets us read each fold's fitted MLP so we
            # can report how many epochs it actually ran before stopping
            # (either hitting MLP_MAX_ITER or triggering early_stopping).
            scores = cross_validate(mlp, X, y, cv=skf, scoring=["f1_macro"],
                                     return_estimator=True)
            mean_score = scores["test_f1_macro"].mean()
            fold_epochs = [int(est.named_steps["clf"].n_iter_)
                           for est in scores["estimator"]]
            mean_epochs = float(np.mean(fold_epochs))
            if mean_score > best_mlp_score:
                best_mlp_score = mean_score
                best_mlp = mlp
                best_mlp_config = {"hidden_layer_sizes": arch, "alpha": alpha}
                best_mlp_epochs_per_fold = fold_epochs
                best_mlp_mean_epochs = mean_epochs
    print(f"  Best MLP config: {best_mlp_config} "
          f"(CV f1_macro={best_mlp_score:.3f})")
    print(f"  Epochs actually used (5-fold CV, max allowed={MLP_MAX_ITER}): "
          f"{best_mlp_epochs_per_fold}, mean={best_mlp_mean_epochs:.1f}")
    target_best_params["MLP"] = {k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in best_mlp_config.items()}
    target_best_params["MLP"]["max_iter_allowed"] = MLP_MAX_ITER
    target_best_params["MLP"]["epochs_used_per_cv_fold"] = best_mlp_epochs_per_fold
    target_best_params["MLP"]["mean_epochs_used_cv"] = best_mlp_mean_epochs

    mlp_result = evaluate_model("MLP (Deep Learning)", best_mlp, X, y,
                                 X_train, X_test, y_train, y_test, skf)
    print(f"  Epochs used on the final 80% holdout-training fit: "
          f"{mlp_result['holdout_epochs_used']}")
    rows.append(mlp_result)

    # --- 3. Stacking ensemble (hybrid model) ---
    if len(tuned_tree_models) >= 2:
        print("\n--- Stacking ensemble (hybrid model) ---")
        stack = StackingClassifier(
            estimators=tuned_tree_models,
            final_estimator=LogisticRegression(max_iter=2000,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE),
            cv=5, n_jobs=-1)
        rows.append(evaluate_model("Stacking Ensemble (hybrid)", stack, X, y,
                                    X_train, X_test, y_train, y_test, skf))
    else:
        print("\n[skip] Stacking needs >= 2 tree models; only Random Forest "
              "is available (install xgboost/lightgbm/catboost for a richer "
              "ensemble).")

    # --- 4. Merge in the Step 7 baseline (untuned) results ---
    baseline_path = os.path.join(
        OUT_DIR, f"model_performance_{target_col.replace(' ', '_')}.csv")
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        for _, r in baseline_df.iterrows():
            rows.append({
                "model": f"{r['model']} (baseline, untuned)",
                "cv_accuracy_mean": r["cv_accuracy_mean"],
                "cv_f1_macro_mean": r["cv_f1_macro_mean"],
                "holdout_accuracy": r["holdout_accuracy"],
                "holdout_f1_macro": r["holdout_f1_macro"],
                "holdout_roc_auc": r["holdout_roc_auc"],
            })
    else:
        print(f"  [note] {baseline_path} not found -- run 07_modeling.py "
              f"first for a full baseline comparison.")

    results_df = pd.DataFrame(rows).sort_values("cv_f1_macro_mean",
                                                  ascending=False)
    out_path = os.path.join(
        OUT_DIR, f"model_performance_advanced_{target_col.replace(' ', '_')}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"\nFull ranking for {target_col} (best first, by CV F1-macro):")
    print(results_df.to_string(index=False))

with open(os.path.join(OUT_DIR, "best_hyperparameters.json"), "w") as f:
    json.dump(all_best_params, f, indent=2, default=str)
print(f"\nSaved -> {os.path.join(OUT_DIR, 'best_hyperparameters.json')}")

print("\n\nDone. Compare each target's 'baseline, untuned' rows against the "
      "'(tuned)' / 'MLP' / 'Stacking' rows above to see exactly how much "
      "tuning, deep learning, and ensembling bought you. Next: "
      "08_ablation_study.py (feature-set comparison, kept deliberately "
      "simple/untuned by design).")
