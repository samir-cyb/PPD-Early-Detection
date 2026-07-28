"""
20_feature_augmentation_test.py
==================================
Closing-the-loop test: do the DISCOVERIES from script 17 (SHAP interaction
values) and script 18 (unsupervised clustering) actually IMPROVE the model,
or were they purely explanatory findings?

This script tests four feature configurations against each target's own
CURRENT BEST/adopted model (not the same reference model for every target --
see PROJECT_STATUS.md's "Consolidated Best Model Per Target" section):

  - EPDS Result:  Direct Random Forest classifier on Arm B (unchanged winner
                  since step 9 -- no Tier 1/2 experiment beat it).
  - PHQ9 Result:  Regression (Linear Regression) on the continuous PHQ9
                  Score, thresholded into the official clinical bands --
                  script 10's adopted winner, NOT the direct classifier.
  - PPD_binary:   Direct Random Forest classifier on Arm B (same as EPDS;
                  threshold retuning from script 11 is an orthogonal,
                  post-hoc decision-boundary choice that doesn't depend on
                  which features are used, so it's left out of this
                  specific feature-comparison test).

The four configurations, each evaluated with 5-fold CV for stability
(averaging out noise is important here since we are looking for a SMALL
effect, if any):
  1. baseline           -- Arm B features only (current adopted model)
  2. plus_interactions   -- Arm B + 3 engineered interaction (product) terms,
                            chosen directly from script 17's cross-target
                            top findings: Maternal_MentalHealth_Risk x
                            Social_Support, Maternal_MentalHealth_Risk x
                            Economic_Stability, "Angry after latest child
                            birth" x Social_Support.
  3. plus_cluster         -- Arm B + a fold-safe KMeans Cluster-ID feature
                            (K=3, same 6-feature clustering space as script
                            18: the 4 weighted composites + Age + pregnancy
                            count). Fold-safe means KMeans is FIT ONLY on
                            each fold's training portion and then used to
                            PREDICT cluster labels for that fold's test
                            portion -- never fit on the full dataset -- to
                            avoid any leakage of test-set distribution
                            information into training.
  4. plus_both            -- both of the above together.

Output (./outputs/):
  - feature_augmentation_results.csv
  - feature_augmentation_summary.txt

Usage
-----
    python 20_feature_augmentation_test.py
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")
ENGINEERED_PATH = os.path.join(OUT_DIR, "PPD_dataset_with_composite_features.csv")

for p in (MODEL_READY_PATH, ENGINEERED_PATH):
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found. Run 06_encoding.py / "
                          f"04_feature_engineering.py first.")

RANDOM_STATE = 42
N_SPLITS = 5
CONFIGS = ["baseline", "plus_interactions", "plus_cluster", "plus_both"]

EQUAL_WEIGHT_COMPOSITES = [
    "Economic_Stability_Index", "Social_Support_Index",
    "Maternal_MentalHealth_Risk_Index", "Neonatal_Delivery_Stress_Index",
]
WEIGHTED_COMPOSITES = [
    "Economic_Stability_Index_Weighted", "Social_Support_Index_Weighted",
    "Maternal_MentalHealth_Risk_Index_Weighted",
    "Neonatal_Delivery_Stress_Index_Weighted",
]
CLUSTER_FEATURES = WEIGHTED_COMPOSITES + ["Age", "Number of the latest pregnancy"]

results = []
summary_lines = []


# ===========================================================================
# Shared helpers
# ===========================================================================
def add_interaction_features(X):
    """Add the 3 product-interaction terms flagged as consistently strongest
    across all 3 targets in script 17's SHAP interaction analysis."""
    X = X.copy()
    X["Interact_MHRI_x_SSI"] = (
        X["Maternal_MentalHealth_Risk_Index_Weighted"]
        * X["Social_Support_Index_Weighted"])
    X["Interact_MHRI_x_ESI"] = (
        X["Maternal_MentalHealth_Risk_Index_Weighted"]
        * X["Economic_Stability_Index_Weighted"])
    X["Interact_Angry_x_SSI"] = (
        X["Angry after latest child birth_Yes"]
        * X["Social_Support_Index_Weighted"])
    return X


def add_cluster_features(X_train, X_test):
    """Fold-safe cluster-ID feature: KMeans fit on TRAIN only, applied to
    both train and test via the fitted scaler+model (no leakage)."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train[CLUSTER_FEATURES])
    test_scaled = scaler.transform(X_test[CLUSTER_FEATURES])
    km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    train_cluster = km.fit_predict(train_scaled)
    test_cluster = km.predict(test_scaled)
    for c in range(3):
        X_train[f"Cluster_{c}"] = (train_cluster == c).astype(int)
        X_test[f"Cluster_{c}"] = (test_cluster == c).astype(int)
    return X_train, X_test


def apply_config(X_tr, X_te, config):
    if config in ("plus_interactions", "plus_both"):
        X_tr = add_interaction_features(X_tr)
        X_te = add_interaction_features(X_te)
    if config in ("plus_cluster", "plus_both"):
        X_tr, X_te = add_cluster_features(X_tr, X_te)
    return X_tr, X_te


# ===========================================================================
# PART 1: EPDS Result & PPD_binary -- direct Random Forest classifier
# (each target's own adopted/current-best approach, per step 9 / Tier 1-2)
# ===========================================================================
df = pd.read_csv(MODEL_READY_PATH)
print(f"Loaded model-ready dataset: {df.shape}")

TARGETS_CLF = ["EPDS Result", "PPD_binary"]
ALL_FEATURE_COLS = [c for c in df.columns
                    if c not in ["EPDS Result", "PHQ9 Result", "PPD_binary"]]
RAW_ONLY_COLS = [c for c in ALL_FEATURE_COLS
                 if c not in EQUAL_WEIGHT_COMPOSITES + WEIGHTED_COMPOSITES]
ARM_B_COLS = RAW_ONLY_COLS + WEIGHTED_COMPOSITES

EPDS_ORDER = {"Low": 0, "Medium": 1, "High": 2}

for target_col in TARGETS_CLF:
    print(f"\n{'#' * 90}\nTARGET: {target_col} (direct Random Forest classifier"
          f" -- current adopted model)\n{'#' * 90}")
    X_base = df[ARM_B_COLS].copy()

    if target_col == "EPDS Result":
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
        class_names = list(le.classes_)
        order_codes = {i: EPDS_ORDER[name] for i, name in enumerate(class_names)}
    else:
        y = df[target_col].astype(int).values
        order_codes = None

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for config in CONFIGS:
        fold_f1, fold_acc, fold_qwk = [], [], []
        for tr_idx, te_idx in skf.split(X_base, y):
            X_tr, X_te = X_base.iloc[tr_idx].copy(), X_base.iloc[te_idx].copy()
            y_tr, y_te = y[tr_idx], y[te_idx]
            X_tr, X_te = apply_config(X_tr, X_te, config)

            model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=RANDOM_STATE)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)

            fold_f1.append(f1_score(y_te, pred, average="macro", zero_division=0))
            fold_acc.append(accuracy_score(y_te, pred))
            if order_codes is not None:
                true_codes = np.array([order_codes[v] for v in y_te])
                pred_codes = np.array([order_codes[v] for v in pred])
                fold_qwk.append(cohen_kappa_score(true_codes, pred_codes,
                                                    weights="quadratic"))

        row = {
            "target": target_col, "config": config,
            "mean_accuracy": np.mean(fold_acc), "std_accuracy": np.std(fold_acc),
            "mean_f1_macro": np.mean(fold_f1), "std_f1_macro": np.std(fold_f1),
        }
        if fold_qwk:
            row["mean_qwk"] = np.mean(fold_qwk)
            row["std_qwk"] = np.std(fold_qwk)
        results.append(row)
        qwk_str = f", QWK={row.get('mean_qwk', float('nan')):.4f}" if fold_qwk else ""
        print(f"  {config:20s}: F1-macro={row['mean_f1_macro']:.4f} "
              f"+/- {row['std_f1_macro']:.4f}, accuracy={row['mean_accuracy']:.4f}"
              f"{qwk_str}")

    base_f1 = next(r["mean_f1_macro"] for r in results
                    if r["target"] == target_col and r["config"] == "baseline")
    best = max((r for r in results if r["target"] == target_col),
               key=lambda r: r["mean_f1_macro"])
    summary_lines.append(
        f"{target_col}: baseline F1-macro={base_f1:.4f}; best config = "
        f"'{best['config']}' (F1-macro={best['mean_f1_macro']:.4f}, "
        f"delta={best['mean_f1_macro'] - base_f1:+.4f})")


# ===========================================================================
# PART 2: PHQ9 Result -- Regression + Threshold (Linear Regression), the
# ADOPTED winner from script 10, NOT the direct classifier.
# ===========================================================================
print(f"\n{'#' * 90}\nTARGET: PHQ9 Result (Linear Regression + threshold -- "
      f"script 10's adopted model)\n{'#' * 90}")

df_eng = pd.read_csv(ENGINEERED_PATH)
print(f"Loaded engineered dataset: {df_eng.shape}")

# The 19 raw EPDS/PHQ9 item-level columns are used ONLY to derive the score/
# category targets -- they must NEVER be model inputs (that would be direct
# leakage). Same exclusion list as script 10.
EPDS_ITEMS = [
    "You have been able to laugh and see the funny side of things",
    "You have looked forward with enjoyment to things",
    "You have blamed myself unnecessarily when things went wrong",
    "You have been anxious or worried for no good reason",
    "You have felt scared or panicky for no good reason",
    "Things have been getting to you",
    "You have been so unhappy that you have had difficulty sleeping",
    "You have felt sad or miserable",
    "You have been so unhappy that you have been crying",
    "The thought of harming yourself has occurred",
]
PHQ9_ITEMS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself or that you are a failure or have let "
    "yourself or your family down",
    "Trouble concentrating on things",
    "Moving or speaking or restlessness",
    "Thoughts that you would be better off dead, or of hurting yourself",
]
CATEGORY_TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
NON_FEATURE_COLS = ["EPDS Score", "PHQ9 Score"] + CATEGORY_TARGETS + EPDS_ITEMS + PHQ9_ITEMS
features_df = df_eng.drop(columns=[c for c in NON_FEATURE_COLS
                                    if c in df_eng.columns])

categorical_cols = features_df.select_dtypes(include="object").columns.tolist()
encoded = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True,
                          dtype=int)

ALL_COLS_ENG = encoded.columns.tolist()
RAW_ONLY_COLS_ENG = [c for c in ALL_COLS_ENG
                     if c not in EQUAL_WEIGHT_COMPOSITES + WEIGHTED_COMPOSITES]
ARM_B_COLS_ENG = [c for c in RAW_ONLY_COLS_ENG + WEIGHTED_COMPOSITES
                  if c in encoded.columns]

X_base_phq9 = encoded[ARM_B_COLS_ENG].copy()
y_cont = df_eng["PHQ9 Score"].values
print(f"PHQ9 regression feature set: {len(ARM_B_COLS_ENG)} features "
      f"(Arm B, rebuilt from the engineered file, matches script 10).")


def phq9_band(score):
    if score <= 4:
        return "Minimal"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    elif score <= 19:
        return "Moderately Severe"
    else:
        return "Severe"


PHQ9_ORDER = {"Minimal": 0, "Mild": 1, "Moderate": 2,
              "Moderately Severe": 3, "Severe": 4}

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for config in CONFIGS:
    fold_acc, fold_f1, fold_qwk = [], [], []
    for tr_idx, te_idx in kf.split(X_base_phq9):
        X_tr, X_te = X_base_phq9.iloc[tr_idx].copy(), X_base_phq9.iloc[te_idx].copy()
        y_tr, y_te = y_cont[tr_idx], y_cont[te_idx]
        X_tr, X_te = apply_config(X_tr, X_te, config)

        reg = LinearRegression()
        reg.fit(X_tr, y_tr)
        pred_cont = reg.predict(X_te)
        pred_cont = np.clip(pred_cont, y_tr.min(), y_tr.max())

        pred_band = np.array([phq9_band(s) for s in pred_cont])
        true_band = np.array([phq9_band(s) for s in y_te])
        pred_code = np.array([PHQ9_ORDER[b] for b in pred_band])
        true_code = np.array([PHQ9_ORDER[b] for b in true_band])

        fold_acc.append(accuracy_score(true_band, pred_band))
        fold_f1.append(f1_score(true_band, pred_band, average="macro", zero_division=0))
        fold_qwk.append(cohen_kappa_score(true_code, pred_code, weights="quadratic"))

    row = {
        "target": "PHQ9 Result", "config": config,
        "mean_accuracy": np.mean(fold_acc), "std_accuracy": np.std(fold_acc),
        "mean_f1_macro": np.mean(fold_f1), "std_f1_macro": np.std(fold_f1),
        "mean_qwk": np.mean(fold_qwk), "std_qwk": np.std(fold_qwk),
    }
    results.append(row)
    print(f"  {config:20s}: F1-macro={row['mean_f1_macro']:.4f} "
          f"+/- {row['std_f1_macro']:.4f}, accuracy={row['mean_accuracy']:.4f}, "
          f"QWK={row['mean_qwk']:.4f}")

base_f1_phq9 = next(r["mean_f1_macro"] for r in results
                     if r["target"] == "PHQ9 Result" and r["config"] == "baseline")
best_phq9 = max((r for r in results if r["target"] == "PHQ9 Result"),
                key=lambda r: r["mean_f1_macro"])
summary_lines.append(
    f"PHQ9 Result: baseline F1-macro={base_f1_phq9:.4f}; best config = "
    f"'{best_phq9['config']}' (F1-macro={best_phq9['mean_f1_macro']:.4f}, "
    f"delta={best_phq9['mean_f1_macro'] - base_f1_phq9:+.4f})")


# ===========================================================================
# Save + summarize
# ===========================================================================
results_df = pd.DataFrame(results)
out_path = os.path.join(OUT_DIR, "feature_augmentation_results.csv")
results_df.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")

summary_path = os.path.join(OUT_DIR, "feature_augmentation_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
print("\n".join(summary_lines))
print(f"Saved -> {summary_path}")
print("\nDone. If 'baseline' wins (or the deltas are tiny, within noise) for "
      "all 3 targets, that's an honest, reportable finding too: it confirms "
      "the composite indices and current models already capture what these "
      "interaction/cluster features would add -- exactly what you'd expect "
      "from a tree-based ensemble that already learns interactions "
      "internally (see script 17's discussion). If any config genuinely "
      "wins by more than its std, that's a concrete, adoptable improvement.")
