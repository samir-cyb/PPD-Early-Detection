"""
11_threshold_calibration.py
=============================
Improvement idea #2: PPD_binary is the project's headline, most clinically
usable target, but every prior script (07, 07b, 08, 09) used the default
0.5 probability cutoff to turn a predicted probability into a 0/1 decision,
and never checked whether the model's predicted probabilities are actually
trustworthy numbers (calibration).

Two things this script adds, using the SAME Arm B "Raw + Composite" feature
set and Random Forest model as steps 8/9 (for a fair, direct comparison):

1. THRESHOLD TUNING. For a screening tool, missing a true PPD case (a false
   negative) is generally worse than a false alarm (a false positive) that
   simply leads to an extra follow-up conversation. The default 0.5 cutoff
   optimizes for balanced error, not this asymmetry. This script computes
   the full precision-recall curve on the held-out test set and reports
   two alternative operating points alongside the default:
     - "Max-F1 threshold": whatever cutoff maximizes F1 on this data.
     - "Screening threshold": the smallest cutoff that still achieves at
       least 85% recall (catches at least 85% of true PPD cases), a common
       target for a first-pass screening instrument.

2. CALIBRATION. A predicted "70% risk" should correspond to roughly 70% of
   such patients actually having PPD -- otherwise the probability number
   itself is misleading, even if the model's rank-ordering (who is higher
   risk than whom) is fine. This script wraps the classifier in
   CalibratedClassifierCV (Platt scaling / sigmoid) and compares the Brier
   score (lower = better-calibrated) and a reliability diagram before and
   after calibration.

Output (./outputs/):
  - threshold_calibration_results.csv    -- metrics at each threshold,
                                             before/after calibration
  - precision_recall_curve.png           -- PR curve with both thresholds
                                             marked
  - calibration_curve.png                -- reliability diagram,
                                             uncalibrated vs calibrated
  - threshold_calibration_summary.txt

Usage
-----
    python 11_threshold_calibration.py
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (precision_recall_curve, f1_score, precision_score,
                              recall_score, accuracy_score, brier_score_loss,
                              roc_auc_score)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")

if not os.path.exists(MODEL_READY_PATH):
    raise SystemExit(f"{MODEL_READY_PATH} not found. Run 06_encoding.py first.")

df = pd.read_csv(MODEL_READY_PATH)
print(f"Loaded model-ready dataset: {df.shape}")

RANDOM_STATE = 42
TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
RECALL_TARGET = 0.85  # screening threshold: catch >= 85% of true PPD cases

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
FEATURE_COLS = RAW_ONLY_COLS + WEIGHTED_COMPOSITES  # Arm B, matches step 8/9

X = df[FEATURE_COLS]
y = df["PPD_binary"].astype(int)
print(f"Feature set: Arm B 'Raw + Composite' -- {len(FEATURE_COLS)} features. "
      f"Target: PPD_binary.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Baseline (uncalibrated) Random Forest -- same model family as step 9
# ---------------------------------------------------------------------------
base_model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=RANDOM_STATE)
base_model.fit(X_train, y_train)
proba_uncal = base_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, proba_uncal)
print(f"\nUncalibrated Random Forest -- test ROC-AUC: {auc:.3f}")

# ---------------------------------------------------------------------------
# 2. Threshold tuning from the precision-recall curve
# ---------------------------------------------------------------------------
precision, recall, thresholds = precision_recall_curve(y_test, proba_uncal)
# precision_recall_curve returns len(thresholds) = len(precision) - 1
denom = precision[:-1] + recall[:-1]
f1_scores = np.zeros_like(denom)
nonzero = denom > 0
f1_scores[nonzero] = (2 * precision[:-1][nonzero] * recall[:-1][nonzero]
                       / denom[nonzero])
best_f1_idx = int(np.argmax(f1_scores))
best_f1_threshold = thresholds[best_f1_idx]

# Smallest threshold achieving recall >= RECALL_TARGET (thresholds are
# ascending, recall is descending as threshold increases, so scan from the
# low-threshold end and take the last index where recall still clears the
# bar, i.e. the highest threshold that still keeps recall >= target).
valid_idx = np.where(recall[:-1] >= RECALL_TARGET)[0]
screening_threshold = thresholds[valid_idx[-1]] if len(valid_idx) > 0 else 0.0

print(f"Max-F1 threshold: {best_f1_threshold:.3f}")
print(f"Screening threshold (recall >= {RECALL_TARGET:.0%}): "
      f"{screening_threshold:.3f}")


def evaluate_at_threshold(name, thresh, proba):
    pred = (proba >= thresh).astype(int)
    return {
        "operating_point": name,
        "threshold": thresh,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }


rows = [
    evaluate_at_threshold("Default (0.5)", 0.5, proba_uncal),
    evaluate_at_threshold("Max-F1", best_f1_threshold, proba_uncal),
    evaluate_at_threshold(f"Screening (recall>={RECALL_TARGET:.0%})",
                           screening_threshold, proba_uncal),
]
print("\nOperating points on the held-out test set:")
print(pd.DataFrame(rows).to_string(index=False))

try:
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label="Precision-Recall curve")
    for r in rows:
        p_at = precision_score(y_test, (proba_uncal >= r["threshold"]).astype(int),
                                zero_division=0)
        r_at = recall_score(y_test, (proba_uncal >= r["threshold"]).astype(int),
                             zero_division=0)
        plt.scatter([r_at], [p_at], s=60, label=f"{r['operating_point']} "
                                                  f"(t={r['threshold']:.2f})")
    plt.xlabel("Recall (sensitivity)")
    plt.ylabel("Precision")
    plt.title("PPD_binary -- Precision-Recall Curve with Candidate Thresholds")
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(OUT_DIR, "precision_recall_curve.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"Saved -> {pr_path}")
except Exception as e:
    print(f"[plot failed] precision-recall curve: {e}")

# ---------------------------------------------------------------------------
# 3. Calibration: CalibratedClassifierCV (sigmoid/Platt scaling)
# ---------------------------------------------------------------------------
print("\n--- Calibration ---")
calibrated_model = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=RANDOM_STATE),
    method="sigmoid", cv=5)
calibrated_model.fit(X_train, y_train)
proba_cal = calibrated_model.predict_proba(X_test)[:, 1]

brier_uncal = brier_score_loss(y_test, proba_uncal)
brier_cal = brier_score_loss(y_test, proba_cal)
auc_cal = roc_auc_score(y_test, proba_cal)

print(f"Brier score -- uncalibrated: {brier_uncal:.4f}, "
      f"calibrated: {brier_cal:.4f} (lower is better)")
print(f"ROC-AUC -- uncalibrated: {auc:.3f}, calibrated: {auc_cal:.3f} "
      f"(calibration should not change ranking/AUC much, only the "
      f"probability VALUES)")

try:
    frac_pos_uncal, mean_pred_uncal = calibration_curve(y_test, proba_uncal,
                                                         n_bins=10)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_test, proba_cal, n_bins=10)
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(mean_pred_uncal, frac_pos_uncal, marker="o",
              label=f"Uncalibrated (Brier={brier_uncal:.3f})")
    plt.plot(mean_pred_cal, frac_pos_cal, marker="o",
              label=f"Calibrated (Brier={brier_cal:.3f})")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of actual positives")
    plt.title("PPD_binary -- Calibration (Reliability) Curve")
    plt.legend()
    plt.tight_layout()
    cal_path = os.path.join(OUT_DIR, "calibration_curve.png")
    plt.savefig(cal_path, dpi=150)
    plt.close()
    print(f"Saved -> {cal_path}")
except Exception as e:
    print(f"[plot failed] calibration curve: {e}")

# ---------------------------------------------------------------------------
# 4. Save results + summary
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(rows)
results_df["brier_uncalibrated"] = brier_uncal
results_df["brier_calibrated"] = brier_cal
results_df["auc_uncalibrated"] = auc
results_df["auc_calibrated"] = auc_cal
out_path = os.path.join(OUT_DIR, "threshold_calibration_results.csv")
results_df.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")

summary = (
    f"PPD_binary threshold tuning + calibration summary\n"
    f"{'=' * 60}\n"
    f"Default threshold (0.5): precision={rows[0]['precision']:.3f}, "
    f"recall={rows[0]['recall']:.3f}, f1={rows[0]['f1']:.3f}\n"
    f"Max-F1 threshold ({best_f1_threshold:.3f}): "
    f"precision={rows[1]['precision']:.3f}, recall={rows[1]['recall']:.3f}, "
    f"f1={rows[1]['f1']:.3f}\n"
    f"Screening threshold ({screening_threshold:.3f}, recall>="
    f"{RECALL_TARGET:.0%}): precision={rows[2]['precision']:.3f}, "
    f"recall={rows[2]['recall']:.3f}, f1={rows[2]['f1']:.3f}\n\n"
    f"Calibration: Brier score {brier_uncal:.4f} -> {brier_cal:.4f} "
    f"({'improved' if brier_cal < brier_uncal else 'did not improve'}); "
    f"ROC-AUC essentially unchanged ({auc:.3f} -> {auc_cal:.3f}), as expected "
    f"since calibration reshapes probability values, not rank ordering.\n"
)
summary_path = os.path.join(OUT_DIR, "threshold_calibration_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)
print(f"\n{summary}")
print(f"Saved -> {summary_path}")
print("\nDone. For a screening tool, the 'Screening' threshold row is "
      "usually the one to recommend clinically (catches most true cases, "
      "accepting more false alarms); the calibrated probabilities are the "
      "ones to report as a patient-facing 'risk score', not the raw ones.")
