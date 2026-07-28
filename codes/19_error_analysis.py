"""
19_error_analysis.py
=======================
Improvement idea: ERROR ANALYSIS -- characterizing exactly which patients
the model gets WRONG, and what they have in common.

Every prior script reported aggregate metrics (accuracy, F1, ROC-AUC) or a
confusion matrix, but never asked "who, specifically, are the ~35 patients
this model misclassifies on the test set, and is there a common pattern
among them?" That pattern, if one exists, tells you exactly where the
model's blind spot is -- valuable for a thesis discussion and for knowing
which patients a clinician should NOT fully trust the model's output for.

PRIMARY TARGET: PPD_binary (clean binary framework). On the held-out test
set (same split as steps 9/11/16 for consistency):
  - False Negatives (FN): truly PPD-positive, model said negative --
    compared against True Positives (TP) -- i.e. "among mothers who really
    have PPD, what distinguishes the ones the model catches from the ones
    it misses?"
  - False Positives (FP): truly PPD-negative, model said positive --
    compared against True Negatives (TN) -- "among mothers who don't have
    PPD, what distinguishes the ones incorrectly flagged?"
Both comparisons use the 4 composite indices + Age via a Mann-Whitney U
test (robust to non-normal small samples), plus a look at a few of the
strongest raw categorical predictors (Abuse, Received Support, Angry after
childbirth).

SECONDARY: a lighter-weight look at EPDS Result / PHQ9 Result errors --
since these are multi-class, "error" is graded by ordinal distance (how
many severity bands off the prediction was), reusing the QWK ordinal
ordering from script 10.

Output (./outputs/):
  - error_analysis_PPD_binary_FN_vs_TP.csv
  - error_analysis_PPD_binary_FP_vs_TN.csv
  - error_analysis_ordinal_distance_<target>.csv
  - error_analysis_summary.txt

Usage
-----
    python 19_error_analysis.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")

if not os.path.exists(MODEL_READY_PATH):
    raise SystemExit(f"{MODEL_READY_PATH} not found. Run 06_encoding.py first.")

df = pd.read_csv(MODEL_READY_PATH).reset_index(drop=True)
print(f"Loaded model-ready dataset: {df.shape}")

RANDOM_STATE = 42
TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]

EQUAL_WEIGHT_COMPOSITES = [
    "Economic_Stability_Index", "Social_Support_Index",
    "Maternal_MentalHealth_Risk_Index", "Neonatal_Delivery_Stress_Index",
]
WEIGHTED_COMPOSITES = [
    "Economic_Stability_Index_Weighted", "Social_Support_Index_Weighted",
    "Maternal_MentalHealth_Risk_Index_Weighted",
    "Neonatal_Delivery_Stress_Index_Weighted",
]
ALL_FEATURE_COLS = [c for c in df.columns if c not in TARGETS]
RAW_ONLY_COLS = [c for c in ALL_FEATURE_COLS
                 if c not in EQUAL_WEIGHT_COMPOSITES + WEIGHTED_COMPOSITES]
FEATURE_COLS = RAW_ONLY_COLS + WEIGHTED_COMPOSITES  # Arm B

PROFILE_COLS = WEIGHTED_COMPOSITES + ["Age", "Number of the latest pregnancy"]
RAW_FLAG_COLS_OF_INTEREST = ["Abuse_Yes", "Angry after latest child birth_Yes",
                              "Received Support_Medium", "Need for Support_Not Reported"]

summary_lines = []

# ===========================================================================
# PRIMARY: PPD_binary FN-vs-TP and FP-vs-TN
# ===========================================================================
print(f"\n{'#' * 90}\nPPD_binary ERROR ANALYSIS\n{'#' * 90}")
y = df["PPD_binary"].astype(int).values
X = df[FEATURE_COLS]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
test_idx = X_test.index  # original row indices for the held-out set

model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RANDOM_STATE)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

test_df = df.loc[test_idx].copy()
test_df["y_true"] = y_test
test_df["y_pred"] = y_pred
test_df["outcome"] = np.select(
    [(test_df["y_true"] == 1) & (test_df["y_pred"] == 1),
     (test_df["y_true"] == 1) & (test_df["y_pred"] == 0),
     (test_df["y_true"] == 0) & (test_df["y_pred"] == 1),
     (test_df["y_true"] == 0) & (test_df["y_pred"] == 0)],
    ["TP", "FN", "FP", "TN"],
    default="UNKNOWN")

counts = test_df["outcome"].value_counts()
print(f"Test set outcome breakdown: {counts.to_dict()}")


def compare_groups(df_a, label_a, df_b, label_b, cols, cat_cols):
    rows = []
    for col in cols:
        a_vals, b_vals = df_a[col].dropna(), df_b[col].dropna()
        if len(a_vals) < 2 or len(b_vals) < 2:
            continue
        try:
            stat, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append({
            "feature": col, f"mean_{label_a}": a_vals.mean(),
            f"mean_{label_b}": b_vals.mean(),
            "mean_diff": a_vals.mean() - b_vals.mean(),
            "mannwhitney_pvalue": p,
            "significant_at_0.05": (p < 0.05) if not np.isnan(p) else False,
        })
    for col in cat_cols:
        if col not in df_a.columns:
            continue
        rate_a = df_a[col].mean()
        rate_b = df_b[col].mean()
        rows.append({
            "feature": col, f"mean_{label_a}": rate_a, f"mean_{label_b}": rate_b,
            "mean_diff": rate_a - rate_b,
            "mannwhitney_pvalue": np.nan, "significant_at_0.05": False,
        })
    return pd.DataFrame(rows).sort_values("mean_diff", key=abs, ascending=False)


fn_df = test_df[test_df["outcome"] == "FN"]
tp_df = test_df[test_df["outcome"] == "TP"]
fp_df = test_df[test_df["outcome"] == "FP"]
tn_df = test_df[test_df["outcome"] == "TN"]

print(f"\n--- FN (missed true PPD cases, n={len(fn_df)}) vs "
      f"TP (correctly caught, n={len(tp_df)}) ---")
fn_vs_tp = compare_groups(fn_df, "FN", tp_df, "TP", PROFILE_COLS,
                           RAW_FLAG_COLS_OF_INTEREST)
print(fn_vs_tp.to_string(index=False))
fn_vs_tp.to_csv(os.path.join(OUT_DIR, "error_analysis_PPD_binary_FN_vs_TP.csv"),
                 index=False)

print(f"\n--- FP (false alarms, n={len(fp_df)}) vs "
      f"TN (correctly cleared, n={len(tn_df)}) ---")
fp_vs_tn = compare_groups(fp_df, "FP", tn_df, "TN", PROFILE_COLS,
                           RAW_FLAG_COLS_OF_INTEREST)
print(fp_vs_tn.to_string(index=False))
fp_vs_tn.to_csv(os.path.join(OUT_DIR, "error_analysis_PPD_binary_FP_vs_TN.csv"),
                 index=False)

sig_fn = fn_vs_tp[fn_vs_tp["significant_at_0.05"]]["feature"].tolist()
sig_fp = fp_vs_tn[fp_vs_tn["significant_at_0.05"]]["feature"].tolist()
summary_lines.append(
    f"PPD_binary: {len(fn_df)} false negatives, {len(fp_df)} false positives "
    f"on the test set (n={len(test_df)}).\n"
    f"  Features significantly different between FN (missed) and TP (caught): "
    f"{sig_fn if sig_fn else 'none at p<0.05'}\n"
    f"  Features significantly different between FP (false alarm) and TN "
    f"(correctly cleared): {sig_fp if sig_fp else 'none at p<0.05'}")

# ===========================================================================
# SECONDARY: ordinal distance errors for EPDS Result / PHQ9 Result
# ===========================================================================
print(f"\n{'#' * 90}\nORDINAL ERROR DISTANCE: EPDS Result / PHQ9 Result\n{'#' * 90}")
EPDS_ORDER = {"Low": 0, "Medium": 1, "High": 2}
PHQ9_ORDER = {"Minimal": 0, "Mild": 1, "Moderate": 2,
              "Moderately Severe": 3, "Severe": 4}

for target_col, order_map in [("EPDS Result", EPDS_ORDER), ("PHQ9 Result", PHQ9_ORDER)]:
    print(f"\n--- TARGET: {target_col} ---")
    le = LabelEncoder()
    y_t = le.fit_transform(df[target_col])
    X_t = df[FEATURE_COLS]
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_t, y_t, test_size=0.2, stratify=y_t, random_state=RANDOM_STATE)
    model_t = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=RANDOM_STATE)
    model_t.fit(X_train_t, y_train_t)
    pred_t = model_t.predict(X_test_t)

    true_labels = le.inverse_transform(y_test_t)
    pred_labels = le.inverse_transform(pred_t)
    true_codes = np.array([order_map[c] for c in true_labels])
    pred_codes = np.array([order_map[c] for c in pred_labels])
    distance = np.abs(true_codes - pred_codes)

    dist_df = pd.DataFrame({"true": true_labels, "predicted": pred_labels,
                             "band_distance": distance})
    dist_counts = dist_df["band_distance"].value_counts().sort_index()
    print(f"  Error distance distribution (0 = correct, 1 = one band off, ...):")
    print(f"  {dist_counts.to_dict()}")
    pct_correct_or_adjacent = (distance <= 1).mean()
    print(f"  {pct_correct_or_adjacent:.1%} of predictions are correct or "
          f"only ONE band off (near-miss).")

    dist_path = os.path.join(
        OUT_DIR, f"error_analysis_ordinal_distance_{target_col.replace(' ', '_')}.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"  Saved -> {dist_path}")
    summary_lines.append(
        f"{target_col}: {pct_correct_or_adjacent:.1%} of test predictions are "
        f"correct or only one severity band off (max possible distance = "
        f"{max(order_map.values())}).")

summary_path = os.path.join(OUT_DIR, "error_analysis_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"\n\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
print("\n".join(summary_lines))
print(f"Saved -> {summary_path}")
print("\nDone. Check the FN-vs-TP and FP-vs-TN tables for any composite "
      "index or raw flag with significant mean_diff -- that's your model's "
      "specific blind spot (e.g. 'the model misses PPD cases with unusually "
      "high apparent social support' would be a very reportable finding).")
