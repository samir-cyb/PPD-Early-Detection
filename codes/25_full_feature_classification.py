"""
25_full_feature_classification.py
===================================
Binary PPD classification using ALL features (including leakage).
Target: EPDS Score >= 13 -> PPD=1, EPDS Score < 13 -> PPD=0
Model: Random Forest with ALL columns from cleaned dataset.

This script intentionally includes leakage features (EPDS items, PHQ-9 items,
EPDS Score, PHQ9 Score) to show the upper-bound performance when the model
has access to everything.

Output (./outputs/):
  - 25_full_feature_metrics.csv
  - 25_full_feature_confusion_matrix.csv
  - 25_full_feature_importance.csv
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
CLEANED_PATH = os.path.join(OUT_DIR, "PPD_dataset_cleaned_v2.csv")

if not os.path.exists(CLEANED_PATH):
    raise SystemExit(f"Cleaned dataset not found at: {CLEANED_PATH}\nRun 02_data_cleaning.py first.")

# ---------------------------------------------------------------------------
# 1. Load cleaned dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(CLEANED_PATH)
print(f"Loaded cleaned dataset: {df.shape}")

# ---------------------------------------------------------------------------
# 2. Create binary target from EPDS Score
#    EPDS Score >= 13 -> PPD (1), EPDS Score < 13 -> Non-PPD (0)
# ---------------------------------------------------------------------------
if "EPDS Score" not in df.columns:
    raise SystemExit("EPDS Score column not found in cleaned dataset.")

df["PPD_binary_from_EPDS"] = (df["EPDS Score"] >= 13).astype(int)
print(f"\nBinary target distribution (EPDS Score >= 13):")
print(df["PPD_binary_from_EPDS"].value_counts().to_string())
print(f"PPD rate: {df['PPD_binary_from_EPDS'].mean():.1%}")

# ---------------------------------------------------------------------------
# 3. Define features: USE EVERYTHING (including leakage)
#    Drop only: the new target we just created, and the 3 old targets
#    Keep: EPDS items, PHQ-9 items, EPDS Score, PHQ9 Score, everything else
# ---------------------------------------------------------------------------
TARGETS_TO_DROP = ["EPDS Result", "PHQ9 Result", "PPD_binary", "PPD_binary_from_EPDS"]
feature_cols = [c for c in df.columns if c not in TARGETS_TO_DROP]

print(f"\nTotal features used (including leakage): {len(feature_cols)}")
print("Features list:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

# ---------------------------------------------------------------------------
# 4. Separate X and y
# ---------------------------------------------------------------------------
X_raw = df[feature_cols].copy()
y = df["PPD_binary_from_EPDS"].values

# ---------------------------------------------------------------------------
# 5. One-hot encode ALL categorical columns
#    (Numeric columns including EPDS Score, PHQ9 Score, Age, etc. stay as-is)
# ---------------------------------------------------------------------------
cat_cols = X_raw.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X_raw.columns if c not in cat_cols]

print(f"\nCategorical columns to encode: {len(cat_cols)}")
print(f"Numeric columns (kept as-is): {len(num_cols)}")

X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=int)
print(f"Shape after one-hot encoding: {X.shape}")

# ---------------------------------------------------------------------------
# 6. Train/Test split (80/20, stratified)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# ---------------------------------------------------------------------------
# 7. Random Forest Model
# ---------------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# ---------------------------------------------------------------------------
# 8. 5-Fold Stratified Cross-Validation
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("10-FOLD CROSS-VALIDATION RESULTS")
print(f"{'='*60}")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

cv_scores = cross_validate(
    model, X, y, cv=skf,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    return_train_score=False
)

cv_results = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "CV_Mean": [
        cv_scores["test_accuracy"].mean(),
        cv_scores["test_precision"].mean(),
        cv_scores["test_recall"].mean(),
        cv_scores["test_f1"].mean(),
        cv_scores["test_roc_auc"].mean(),
    ],
    "CV_Std": [
        cv_scores["test_accuracy"].std(),
        cv_scores["test_precision"].std(),
        cv_scores["test_recall"].std(),
        cv_scores["test_f1"].std(),
        cv_scores["test_roc_auc"].std(),
    ]
}

cv_df = pd.DataFrame(cv_results)
print(cv_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 9. Holdout Test Set Evaluation
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("HOLDOUT TEST SET RESULTS (20% unseen data)")
print(f"{'='*60}")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

holdout_acc = accuracy_score(y_test, y_pred)
holdout_prec = precision_score(y_test, y_pred, zero_division=0)
holdout_rec = recall_score(y_test, y_pred, zero_division=0)
holdout_f1 = f1_score(y_test, y_pred, zero_division=0)
holdout_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy : {holdout_acc:.4f}")
print(f"Precision: {holdout_prec:.4f}")
print(f"Recall   : {holdout_rec:.4f}")
print(f"F1-Score : {holdout_f1:.4f}")
print(f"ROC-AUC  : {holdout_auc:.4f}")

print(f"\n{'='*60}")
print("CLASSIFICATION REPORT")
print(f"{'='*60}")
print(classification_report(y_test, y_pred, target_names=["Non-PPD", "PPD"], zero_division=0))

# ---------------------------------------------------------------------------
# 10. Confusion Matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual Non-PPD", "Actual PPD"],
    columns=["Predicted Non-PPD", "Predicted PPD"]
)
print(f"\n{'='*60}")
print("CONFUSION MATRIX")
print(f"{'='*60}")
print(cm_df.to_string())

# ---------------------------------------------------------------------------
# 11. Feature Importance (Top 30)
# ---------------------------------------------------------------------------
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print(f"\n{'='*60}")
print("TOP 30 FEATURE IMPORTANCE (Random Forest)")
print(f"{'='*60}")
for i, (feat, score) in enumerate(importance.head(30).items(), 1):
    print(f"  {i:2d}. {feat:<50s} {score:.4f}")

# ---------------------------------------------------------------------------
# 12. Save all results to outputs/
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# Save CV metrics
cv_df.to_csv(os.path.join(OUT_DIR, "25_full_feature_metrics.csv"), index=False)
print(f"\nSaved -> {os.path.join(OUT_DIR, '25_full_feature_metrics.csv')}")

# Save confusion matrix
cm_df.to_csv(os.path.join(OUT_DIR, "25_full_feature_confusion_matrix.csv"))
print(f"Saved -> {os.path.join(OUT_DIR, '25_full_feature_confusion_matrix.csv')}")

# Save feature importance
imp_df = importance.reset_index()
imp_df.columns = ["Feature", "Importance"]
imp_df.to_csv(os.path.join(OUT_DIR, "25_full_feature_importance.csv"), index=False)
print(f"Saved -> {os.path.join(OUT_DIR, '25_full_feature_importance.csv')}")

print(f"\n{'='*60}")
print("DONE! All results saved to outputs/ folder.")
print(f"{'='*60}")