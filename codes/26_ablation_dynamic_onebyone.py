"""
26_ablation_dynamic_onebyone.py
================================
Dynamic ablation study: starting with ALL original features (excluding
technical flags), iteratively remove the LEAST important feature using
Chi-Square + Cramer's V (re-ranked dynamically after each removal).
At every step, one-hot encode the remaining features and run 10-fold
Stratified CV with Random Forest.

Target: PPD_binary (EPDS Score >= 13)
Model: Random Forest (n_estimators=300, class_weight="balanced")
Ranking: Chi-Square test of independence + Cramer's V effect size
Removal: One ORIGINAL feature at a time (all its dummies go together)

Output (./outputs/):
  - 26_ablation_dynamic_onebyone.csv
  - 26_ablation_dynamic_progress.csv
  - 26_ablation_dynamic_summary.txt
"""

import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
CLEANED_PATH = os.path.join(OUT_DIR, "PPD_dataset_cleaned_v2.csv")

if not os.path.exists(CLEANED_PATH):
    raise SystemExit(
        f"Cleaned dataset not found at: {CLEANED_PATH}\n"
        f"Run 02_data_cleaning.py first."
    )

# ---------------------------------------------------------------------------
# 1. Load dataset & create binary target from EPDS Score
# ---------------------------------------------------------------------------
df = pd.read_csv(CLEANED_PATH)
print(f"Loaded cleaned dataset: {df.shape}")

if "EPDS Score" not in df.columns:
    raise SystemExit("EPDS Score column not found.")

df["PPD_binary_ablation"] = (df["EPDS Score"] >= 13).astype(int)
y = df["PPD_binary_ablation"].values

print(f"\nBinary target (EPDS Score >= 13):")
print(pd.Series(y).value_counts().to_string())
print(f"PPD positive rate: {y.mean():.1%}")

# ---------------------------------------------------------------------------
# 2. Define leakage columns (for labeling only)
# ---------------------------------------------------------------------------
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
RAW_SCORES = ["EPDS Score", "PHQ9 Score"]
ALL_LEAKAGE = set(EPDS_ITEMS + PHQ9_ITEMS + RAW_SCORES)

def leakage_label(col):
    return "[LEAKAGE]" if col in ALL_LEAKAGE else "[SAFE]"

# ---------------------------------------------------------------------------
# 3. Prepare feature pool (same exclusions as script 27)
# ---------------------------------------------------------------------------
EXCLUDE_COLS = [
    "Abuse_was_missing",
    "Husband's monthly income_was_missing",
    "Husband's education level_was_missing",
    "Trust and share feelings_was_missing",
]
INDEX_LIKE = {"serial", "id", "sl", "sl.", "serial no", "serial no."}

OTHER_TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary", "PPD_binary_ablation"]
drop_cols = [c for c in df.columns
             if c.lower() in INDEX_LIKE
             or c in OTHER_TARGETS
             or c in EXCLUDE_COLS]

df_features = df.drop(columns=drop_cols, errors="ignore")
y_series = df["PPD_binary_ablation"].copy()

print(f"\nOriginal features to test: {df_features.shape[1]}")

# ---------------------------------------------------------------------------
# 4. Helper: Chi-Square + Cramer's V ranking
# ---------------------------------------------------------------------------
def cramers_v(chi2, n, r, k):
    if n == 0:
        return 0.0
    denom = n * (min(k - 1, r - 1))
    if denom == 0:
        return 0.0
    return np.sqrt(chi2 / denom)


def chi_square_ranking(features_df, y_ser, cols):
    """Rank original features by Chi-Square p-value (ascending),
    tie-broken by Cramer's V (descending)."""
    rows = []
    for col in cols:
        try:
            ct = pd.crosstab(features_df[col], y_ser)
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                rows.append({
                    "feature": col, "type": leakage_label(col),
                    "chi2": np.nan, "p_value": np.nan,
                    "neg_log10_p": 0.0, "cramers_v": 0.0,
                })
                continue

            chi2_stat, p_val, dof, expected = chi2_contingency(ct)
            r, k = ct.shape[0], ct.shape[1]
            cv = cramers_v(chi2_stat, ct.sum().sum(), r, k)

            rows.append({
                "feature": col, "type": leakage_label(col),
                "chi2": chi2_stat, "p_value": p_val,
                "neg_log10_p": -np.log10(p_val + 1e-300),
                "cramers_v": cv,
            })
        except Exception:
            rows.append({
                "feature": col, "type": leakage_label(col),
                "chi2": np.nan, "p_value": np.nan,
                "neg_log10_p": 0.0, "cramers_v": 0.0,
            })

    rank_df = pd.DataFrame(rows).sort_values(
        by=["p_value", "cramers_v"], ascending=[True, False]
    ).reset_index(drop=True)
    return rank_df


# ---------------------------------------------------------------------------
# 5. Setup Random Forest & 10-fold CV
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 10

MODEL = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# ---------------------------------------------------------------------------
# 6. Dynamic one-by-one ablation loop
# ---------------------------------------------------------------------------
remaining_features = list(df_features.columns)
results = []
step = 0

print(f"\n{'='*75}")
print("DYNAMIC ABLATION: Chi-Square ranking + one-by-one removal")
print(f"{'='*75}")
print(f"Total steps expected: ~{len(remaining_features)}")
print("Progress prints every step...\n")

while len(remaining_features) >= 1:
    step += 1
    n_orig = len(remaining_features)

    # --- A. Dynamic Chi-Square re-ranking on remaining original features ---
    rank_df = chi_square_ranking(df_features, y_series, remaining_features)

    worst = rank_df.iloc[-1]          # least associated with target
    best = rank_df.iloc[0]            # most associated with target

    worst_feature = worst["feature"]
    worst_type = worst["type"]
    worst_p = worst["p_value"]
    worst_cv = worst["cramers_v"]

    best_feature = best["feature"]
    best_type = best["type"]
    best_p = best["p_value"]
    best_cv = best["cramers_v"]

    # --- B. One-hot encode remaining features for RF ---
    X_raw = df_features[remaining_features].copy()
    cat_cols = X_raw.select_dtypes(include="object").columns.tolist()
    X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=int)

    # --- C. 10-fold CV ---
    try:
        cv = cross_validate(
            MODEL, X_enc, y, cv=skf, scoring=scoring, n_jobs=1,
        )

        row = {
            "step": step,
            "n_original_features": n_orig,
            "n_encoded_features": X_enc.shape[1],
            "removed_next": worst_feature if n_orig > 1 else "NONE (final)",
            "removed_type": worst_type,
            "removed_p_value": worst_p,
            "removed_cramers_v": worst_cv,
            "top_feature": best_feature,
            "top_type": best_type,
            "top_p_value": best_p,
            "top_cramers_v": best_cv,
            "cv_accuracy_mean": cv["test_accuracy"].mean(),
            "cv_accuracy_std": cv["test_accuracy"].std(),
            "cv_precision_mean": cv["test_precision"].mean(),
            "cv_precision_std": cv["test_precision"].std(),
            "cv_recall_mean": cv["test_recall"].mean(),
            "cv_recall_std": cv["test_recall"].std(),
            "cv_f1_mean": cv["test_f1"].mean(),
            "cv_f1_std": cv["test_f1"].std(),
            "cv_roc_auc_mean": cv["test_roc_auc"].mean(),
            "cv_roc_auc_std": cv["test_roc_auc"].std(),
        }

    except Exception as e:
        print(f"  [WARN] CV failed at {n_orig} features: {e}")
        row = {
            "step": step,
            "n_original_features": n_orig,
            "n_encoded_features": X_enc.shape[1],
            "removed_next": worst_feature if n_orig > 1 else "NONE (final)",
            "removed_type": worst_type,
            "removed_p_value": worst_p,
            "removed_cramers_v": worst_cv,
            "top_feature": best_feature,
            "top_type": best_type,
            "top_p_value": best_p,
            "top_cramers_v": best_cv,
            "cv_accuracy_mean": np.nan, "cv_accuracy_std": np.nan,
            "cv_precision_mean": np.nan, "cv_precision_std": np.nan,
            "cv_recall_mean": np.nan, "cv_recall_std": np.nan,
            "cv_f1_mean": np.nan, "cv_f1_std": np.nan,
            "cv_roc_auc_mean": np.nan, "cv_roc_auc_std": np.nan,
        }

    results.append(row)

    print(
        f"Step {step:3d} | {n_orig:3d} orig | {X_enc.shape[1]:3d} enc | "
        f"Acc={row['cv_accuracy_mean']:.4f} | F1={row['cv_f1_mean']:.4f} | "
        f"AUC={row['cv_roc_auc_mean']:.4f} | "
        f"Remove: {worst_feature[:32]:32s} {worst_type} p={worst_p:.2e}"
    )

    # Auto-save every 10 steps
    if step % 10 == 0 or n_orig <= 5:
        pd.DataFrame(results).to_csv(
            os.path.join(OUT_DIR, "26_ablation_dynamic_progress.csv"),
            index=False,
        )

    if n_orig == 1:
        print(f"\n{'='*75}")
        print("Reached final feature. Stopping ablation.")
        print(f"{'='*75}")
        break

    remaining_features.remove(worst_feature)

# ---------------------------------------------------------------------------
# 7. Save final results
# ---------------------------------------------------------------------------
res_df = pd.DataFrame(results)

out_csv = os.path.join(OUT_DIR, "26_ablation_dynamic_onebyone.csv")
res_df.to_csv(out_csv, index=False)
print(f"\nSaved final results -> {out_csv}")

# ---------------------------------------------------------------------------
# 8. Summary analysis
# ---------------------------------------------------------------------------
print(f"\n{'='*75}")
print("SUMMARY")
print(f"{'='*75}")

best_idx = res_df["cv_f1_mean"].idxmax()
best_row = res_df.loc[best_idx]
print(f"\nBest F1-Score: {best_row['cv_f1_mean']:.4f} (+/- {best_row['cv_f1_std']:.4f})")
print(f"  At: Step {best_row['step']} | {best_row['n_original_features']} original features "
      f"({best_row['n_encoded_features']} encoded)")
print(f"  Top feature there: {best_row['top_feature']} ({best_row['top_type']})")

print(f"\nTop 10 steps by F1-Score:")
top10 = res_df.nlargest(10, "cv_f1_mean")[
    ["step", "n_original_features", "n_encoded_features",
     "cv_f1_mean", "cv_accuracy_mean", "cv_roc_auc_mean",
     "top_feature", "top_type"]
]
print(top10.to_string(index=False))

print(f"\nLast 10 steps (collapse phase):")
last10 = res_df.tail(10)[
    ["step", "n_original_features", "n_encoded_features",
     "cv_f1_mean", "cv_accuracy_mean", "cv_roc_auc_mean",
     "top_feature", "top_type"]
]
print(last10.to_string(index=False))

threshold_f1 = best_row["cv_f1_mean"] * 0.95
drop_df = res_df[res_df["cv_f1_mean"] < threshold_f1]
if not drop_df.empty:
    dr = drop_df.iloc[0]
    print(f"\nF1 dropped below 95% of best at:")
    print(f"  Step {dr['step']} ({dr['n_original_features']} original) -> F1={dr['cv_f1_mean']:.4f}")

# Save text summary
summary_lines = [
    "Dynamic Ablation Summary (Chi-Square Ranking)",
    "=" * 75,
    f"Starting original features: {df_features.shape[1]}",
    f"CV method: {N_SPLITS}-fold Stratified",
    f"Model: Random Forest (n_estimators=300, class_weight=balanced)",
    f"Ranking: Chi-Square + Cramer's V (dynamic re-ranking each step)",
    f"Exclusions: 4 '_was_missing' flags + index-like columns",
    "",
    f"Best F1: {best_row['cv_f1_mean']:.4f}",
    f"  Step {best_row['step']} | {best_row['n_original_features']} original features",
    f"  Top feature: {best_row['top_feature']} ({best_row['top_type']})",
    "",
    "Top 10 steps by F1:",
    top10.to_string(index=False),
    "",
    "Last 10 steps:",
    last10.to_string(index=False),
]

summary_path = os.path.join(OUT_DIR, "26_ablation_dynamic_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"\nSaved summary -> {summary_path}")

print(f"\n{'='*75}")
print("DONE. Review these files:")
print(f"  1. {out_csv}")
print(f"  2. {summary_path}")
print(f"{'='*75}")