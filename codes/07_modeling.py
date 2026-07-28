"""
07_modeling.py
===============
Trains baseline ML models for all 3 possible targets, using the FULL
feature set (raw + composite indices together) from PPD_model_ready.csv.
This is the "everything included" baseline -- script 08 (ablation study)
is what isolates whether the composite indices actually add value on top
of raw features.

For each target (EPDS Result / PHQ9 Result / PPD_binary):
  - 80/20 stratified train/test split (random_state=42, reproducible)
  - Stratified 5-fold cross-validation on the full data (robustness check)
  - Models: Logistic Regression + Random Forest always (both with
    class_weight="balanced" to handle the mild class imbalance); XGBoost,
    LightGBM, CatBoost if installed (skipped gracefully otherwise)
  - Metrics: Accuracy, Precision, Recall, F1 (macro-averaged for the
    multi-class targets), ROC-AUC (binary: standard; multi-class: one-vs-
    rest, weighted)

Output (./outputs/):
  - model_performance_EPDS_Result.csv
  - model_performance_PHQ9_Result.csv
  - model_performance_PPD_binary.csv
  (each row = one model, columns = CV and holdout metrics)

Usage
-----
    pip install scikit-learn xgboost lightgbm catboost --break-system-packages
    python 07_modeling.py

xgboost / lightgbm / catboost are optional -- if any aren't installed, this
script just skips that model and tells you what to pip install, it will
NOT crash.
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")

if not os.path.exists(MODEL_READY_PATH):
    raise SystemExit(f"{MODEL_READY_PATH} not found. Run 06_encoding.py first.")

df = pd.read_csv(MODEL_READY_PATH)
print(f"Loaded model-ready dataset: {df.shape}")

TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Optional model imports
# ---------------------------------------------------------------------------
EXTRA_MODELS = {}

try:
    from xgboost import XGBClassifier
    # NOTE: use_label_encoder is a removed/deprecated xgboost parameter in
    # recent versions -- passing it just spams a warning on every fold, so
    # it's intentionally left out here.
    EXTRA_MODELS["XGBoost"] = lambda: XGBClassifier(
        n_estimators=300, random_state=RANDOM_STATE, eval_metric="logloss")
except ImportError:
    print("[skip] xgboost not installed -- "
          "pip install xgboost --break-system-packages to include it.")

try:
    from lightgbm import LGBMClassifier
    EXTRA_MODELS["LightGBM"] = lambda: LGBMClassifier(
        n_estimators=300, random_state=RANDOM_STATE, verbose=-1)
except ImportError:
    print("[skip] lightgbm not installed -- "
          "pip install lightgbm --break-system-packages to include it.")

try:
    from catboost import CatBoostClassifier
    EXTRA_MODELS["CatBoost"] = lambda: CatBoostClassifier(
        iterations=300, random_state=RANDOM_STATE, verbose=False)
except ImportError:
    print("[skip] catboost not installed -- "
          "pip install catboost --break-system-packages to include it.")


def get_models():
    """Fresh model instances every call (avoids state leaking across targets)."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE),
    }
    for name, factory in EXTRA_MODELS.items():
        models[name] = factory()
    return models


# ---------------------------------------------------------------------------
# 2. Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_target(target_col):
    print(f"\n{'=' * 90}\nTARGET: {target_col}\n{'=' * 90}")

    X = df.drop(columns=TARGETS)
    y_raw = df[target_col]
    is_binary = target_col == "PPD_binary"

    # Encode string targets to integers for sklearn/xgboost compatibility;
    # keep the encoder so we can decode class names in reports.
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    # classification_report requires string labels -- PPD_binary's classes
    # are numpy.int64 (0/1), which crashes classification_report's internal
    # `len(cn) for cn in target_names` check. Cast to plain strings for
    # display purposes only (y itself stays numeric for the models).
    class_names_str = [str(c) for c in class_names]
    print(f"Classes ({len(class_names)}): {list(class_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for name, model in get_models().items():
        print(f"\n--- {name} ---")

        # 5-fold CV on the FULL dataset (robustness check, matches the
        # methodology style of the reference papers)
        cv_scores = cross_validate(
            model, X, y, cv=skf,
            scoring=["accuracy", "f1_macro"], n_jobs=None)
        cv_acc_mean, cv_acc_std = cv_scores["test_accuracy"].mean(), cv_scores["test_accuracy"].std()
        cv_f1_mean, cv_f1_std = cv_scores["test_f1_macro"].mean(), cv_scores["test_f1_macro"].std()
        print(f"  CV accuracy: {cv_acc_mean:.3f} +/- {cv_acc_std:.3f}")
        print(f"  CV F1 (macro): {cv_f1_mean:.3f} +/- {cv_f1_std:.3f}")

        # Single 80/20 holdout fit (for a confusion matrix / classification
        # report you can paste directly into the thesis)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        roc_auc = np.nan
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if is_binary:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_proba,
                                             multi_class="ovr",
                                             average="weighted")
        except Exception as e:
            print(f"  (ROC-AUC could not be computed: {e})")

        print(f"  Holdout: acc={acc:.3f}, precision={prec:.3f}, "
              f"recall={rec:.3f}, f1_macro={f1:.3f}, roc_auc={roc_auc:.3f}")
        print(classification_report(y_test, y_pred,
                                     target_names=class_names_str,
                                     zero_division=0))
        print("Confusion matrix (rows=actual, cols=predicted):")
        print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                            index=class_names_str,
                            columns=class_names_str).to_string())

        results.append({
            "target": target_col,
            "model": name,
            "cv_accuracy_mean": cv_acc_mean,
            "cv_accuracy_std": cv_acc_std,
            "cv_f1_macro_mean": cv_f1_mean,
            "cv_f1_macro_std": cv_f1_std,
            "holdout_accuracy": acc,
            "holdout_precision_macro": prec,
            "holdout_recall_macro": rec,
            "holdout_f1_macro": f1,
            "holdout_roc_auc": roc_auc,
        })

    results_df = pd.DataFrame(results).sort_values("cv_f1_macro_mean",
                                                     ascending=False)
    out_path = os.path.join(
        OUT_DIR, f"model_performance_{target_col.replace(' ', '_')}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"\nRanked by CV F1 (macro), best first:")
    print(results_df[["model", "cv_accuracy_mean", "cv_f1_macro_mean",
                       "holdout_accuracy", "holdout_f1_macro",
                       "holdout_roc_auc"]].to_string(index=False))
    return results_df


# ---------------------------------------------------------------------------
# 3. Run for all 3 targets
# ---------------------------------------------------------------------------
all_results = {}
for target in TARGETS:
    all_results[target] = evaluate_target(target)

print(f"\n\n{'=' * 90}\nOVERALL BEST MODEL PER TARGET (by CV F1 macro)\n{'=' * 90}")
for target, res in all_results.items():
    best = res.iloc[0]
    print(f"{target}: {best['model']} "
          f"(CV F1={best['cv_f1_macro_mean']:.3f}, "
          f"holdout accuracy={best['holdout_accuracy']:.3f})")

print("\nDone. These are FULL-feature-set baselines (raw + composite "
      "together). Next: 08_ablation_study.py compares Raw-only vs "
      "Raw+Composite vs Selected+Composite to isolate the composite "
      "indices' actual contribution.")
