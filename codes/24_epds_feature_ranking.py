"""
24_epds_feature_ranking.py
==========================
Ranks ALL features (including EPDS & PHQ-9 question columns) by importance
for predicting EPDS_Result (3-class: Low / Medium / High).

⚠️  NOTE: This script intentionally includes EPDS and PHQ-9 items so you can
    SEE how they rank vs safe demographic features. In real modeling these
    columns are excluded (leakage). Here we include them for analysis only.

TWO VIEWS produced:
  VIEW A — all one-hot encoded columns  (detailed binary level)
  VIEW B — all original features        (grouped / readable level)

THREE RANKING METHODS:
  1. Chi-square test        → statistical association, no model needed
  2. RF Feature Importance  → how often the RF splits on each feature
  3. SHAP mean |value|      → most reliable: actual contribution per prediction

Each feature is labelled [LEAKAGE] or [SAFE] so you can clearly see which is which.

OUTPUT FILES (saved to codes/outputs/):
  - epds_ranking_all_encoded.csv    → all encoded columns ranked
  - epds_ranking_all_original.csv   → all original features ranked (grouped)
  - epds_ranking_all_summary.txt    → top-30 printed report

Usage:
    pip install shap --break-system-packages
    python 24_epds_feature_ranking.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(THIS_DIR, "outputs")
CLEANED_PATH = os.path.join(OUT_DIR, "PPD_dataset_cleaned_v2.csv")

if not os.path.exists(CLEANED_PATH):
    raise SystemExit(f"File not found: {CLEANED_PATH}\nRun 02_data_cleaning.py first.")

TARGET = "EPDS Result"

# These are the leakage columns (EPDS items + PHQ-9 items + raw scores)
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
RAW_SCORES   = ["EPDS Score", "PHQ9 Score"]
ALL_LEAKAGE  = set(EPDS_ITEMS + PHQ9_ITEMS + RAW_SCORES)

# The 3 targets — these are the labels, never used as features
ALL_TARGETS  = ["EPDS Result", "PHQ9 Result", "PPD_binary"]

# Index/serial column
INDEX_LIKE   = {"serial", "id", "sl", "sl.", "serial no", "serial no."}

print("=" * 65)
print("  EPDS Feature Ranking — ALL columns (Script 24)")
print("  Including EPDS & PHQ-9 items  [LEAKAGE marked clearly]")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Load cleaned dataset (has everything)
# ═══════════════════════════════════════════════════════════════════
df_raw = pd.read_csv(CLEANED_PATH)
print(f"\n[Step 1] Loaded cleaned dataset: {df_raw.shape}")

# Drop only: index/serial column and the 2 OTHER targets (keep EPDS Result as y)
drop_cols = [c for c in df_raw.columns
             if c.lower() in INDEX_LIKE
             or c in ["PHQ9 Result", "PPD_binary"]]
df_raw = df_raw.drop(columns=drop_cols, errors="ignore")

# Separate target
if TARGET not in df_raw.columns:
    raise SystemExit(f"Target '{TARGET}' not found in dataset.")

y_series = df_raw[TARGET].copy()
df_features = df_raw.drop(columns=[TARGET])

print(f"  Features (before encoding): {df_features.shape[1]}")
print(f"  Target: {TARGET}  |  Classes: {sorted(y_series.unique())}")

# Label-encode target
le = LabelEncoder()
y_enc = le.fit_transform(y_series)
class_names = le.classes_

# Build leakage label map for original columns
def leakage_label(col):
    return "[LEAKAGE]" if col in ALL_LEAKAGE else "[SAFE]"

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Chi-square on ALL original columns
# ═══════════════════════════════════════════════════════════════════
print("\n[Step 2] Chi-square test on all original columns...")

chi2_results = []
for col in df_features.columns:
    try:
        ct = pd.crosstab(df_features[col], y_series)
        chi2_stat, p_val, dof, _ = chi2_contingency(ct)
        chi2_results.append({
            "original_feature": col,
            "type": leakage_label(col),
            "chi2_statistic": round(chi2_stat, 4),
            "chi2_p_value": p_val,
            "chi2_neg_log10_p": round(-np.log10(p_val + 1e-300), 4),
            "significant_p05": "Yes" if p_val < 0.05 else "No",
        })
    except Exception:
        chi2_results.append({
            "original_feature": col,
            "type": leakage_label(col),
            "chi2_statistic": np.nan,
            "chi2_p_value": np.nan,
            "chi2_neg_log10_p": 0.0,
            "significant_p05": "Error",
        })

chisq_df = (pd.DataFrame(chi2_results)
            .sort_values("chi2_neg_log10_p", ascending=False)
            .reset_index(drop=True))
chisq_df.insert(0, "chi2_rank", range(1, len(chisq_df) + 1))
print(f"  Done for {len(chi2_results)} features.")

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — One-hot encode ALL features
# ═══════════════════════════════════════════════════════════════════
print("\n[Step 3] One-hot encoding all features...")

cat_cols = df_features.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in df_features.columns if c not in cat_cols]

print(f"  Categorical: {len(cat_cols)}  |  Numeric: {len(num_cols)}")

encoded = pd.get_dummies(df_features, columns=cat_cols, drop_first=True, dtype=int)
print(f"  Shape after encoding: {encoded.shape}")

# Build map: encoded column → original column
col_to_orig = {}
for col in num_cols:
    col_to_orig[col] = col
for col in cat_cols:
    for enc_col in encoded.columns:
        if enc_col.startswith(col + "_"):
            col_to_orig[enc_col] = col

feature_names = encoded.columns.tolist()
X = encoded.values

# ═══════════════════════════════════════════════════════════════════
# STEP 4 — Train Random Forest on ALL columns
# ═══════════════════════════════════════════════════════════════════
print("\n[Step 4] Training Random Forest on ALL columns...")

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X, y_enc)
rf_imp = rf.feature_importances_

rf_df = pd.DataFrame({
    "encoded_feature":   feature_names,
    "original_feature":  [col_to_orig.get(c, c) for c in feature_names],
    "rf_importance":     rf_imp,
}).sort_values("rf_importance", ascending=False).reset_index(drop=True)
rf_df.insert(0, "rf_rank", range(1, len(rf_df) + 1))
rf_df["type"] = rf_df["original_feature"].apply(leakage_label)

print(f"  Top feature: {rf_df.iloc[0]['encoded_feature']} "
      f"({rf_df.iloc[0]['type']}, imp={rf_df.iloc[0]['rf_importance']:.4f})")

# ═══════════════════════════════════════════════════════════════════
# STEP 5 — SHAP values
# ═══════════════════════════════════════════════════════════════════
print("\n[Step 5] Computing SHAP values (may take 30-90 seconds)...")

explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# Handle all SHAP output formats
if isinstance(shap_values, list):
    shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
elif isinstance(shap_values, np.ndarray):
    if shap_values.ndim == 3:          # (n_samples, n_features, n_classes)
        shap_abs_mean = np.abs(shap_values).mean(axis=0).mean(axis=1)
    else:                               # (n_samples, n_features)
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
else:
    shap_abs_mean = np.abs(np.array(shap_values)).mean(axis=0)

shap_abs_mean = np.array(shap_abs_mean).flatten()
assert len(shap_abs_mean) == len(feature_names), "SHAP length mismatch"

shap_df = pd.DataFrame({
    "encoded_feature": feature_names,
    "shap_mean_abs":   shap_abs_mean,
}).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
shap_df.insert(0, "shap_rank", range(1, len(shap_df) + 1))

print(f"  Top SHAP feature: {shap_df.iloc[0]['encoded_feature']} "
      f"(SHAP={shap_df.iloc[0]['shap_mean_abs']:.4f})")

# ═══════════════════════════════════════════════════════════════════
# STEP 6 — 108-column master (encoded level)
# ═══════════════════════════════════════════════════════════════════
print("\n[Step 6] Building encoded-level master ranking...")

master_enc = (rf_df
              .merge(shap_df[["encoded_feature", "shap_rank", "shap_mean_abs"]],
                     on="encoded_feature", how="left"))
master_enc["avg_rank"] = (master_enc["rf_rank"] + master_enc["shap_rank"]) / 2
master_enc = master_enc.sort_values("avg_rank").reset_index(drop=True)
master_enc.insert(0, "final_rank_encoded", range(1, len(master_enc) + 1))

out_enc_path = os.path.join(OUT_DIR, "epds_ranking_all_encoded.csv")
master_enc.to_csv(out_enc_path, index=False)
print(f"  Saved -> {out_enc_path}  ({len(master_enc)} rows)")

# ═══════════════════════════════════════════════════════════════════
# STEP 7 — Group back to original features
# ═══════════════════════════════════════════════════════════════════
print("\n[Step 7] Grouping → original feature level...")

grouped = (master_enc
           .groupby("original_feature", as_index=False)
           .agg(
               type               = ("type",          "first"),
               rf_importance_sum  = ("rf_importance",  "sum"),
               shap_mean_abs_sum  = ("shap_mean_abs",  "sum"),
               n_dummies          = ("encoded_feature", "count"),
           ))

# Merge chi-square
grouped = grouped.merge(
    chisq_df[["original_feature", "chi2_rank", "chi2_statistic",
               "chi2_p_value", "chi2_neg_log10_p", "significant_p05"]],
    on="original_feature", how="left",
)

grouped["rf_grouped_rank"]   = grouped["rf_importance_sum"].rank(ascending=False).astype(int)
grouped["shap_grouped_rank"] = grouped["shap_mean_abs_sum"].rank(ascending=False).astype(int)
grouped["chi2_rank"]         = grouped["chi2_rank"].fillna(999).astype(int)

grouped["avg_rank_3methods"] = (
    grouped["rf_grouped_rank"] +
    grouped["shap_grouped_rank"] +
    grouped["chi2_rank"]
) / 3

grouped = grouped.sort_values("avg_rank_3methods").reset_index(drop=True)
grouped.insert(0, "final_rank", range(1, len(grouped) + 1))

for c in ["rf_importance_sum", "shap_mean_abs_sum", "chi2_neg_log10_p"]:
    grouped[c] = grouped[c].round(5)

out_orig_path = os.path.join(OUT_DIR, "epds_ranking_all_original.csv")
grouped.to_csv(out_orig_path, index=False)
print(f"  Saved -> {out_orig_path}  ({len(grouped)} rows)")

# ═══════════════════════════════════════════════════════════════════
# STEP 8 — Print reports
# ═══════════════════════════════════════════════════════════════════
sep = "─" * 72

print(f"\n{sep}")
print(f"  TOP 30 — Original features (ALL columns, grouped)  |  Target: {TARGET}")
print(f"  [LEAKAGE] = EPDS/PHQ-9 items (excluded in real modeling)")
print(f"  [SAFE]    = demographic, social, economic features")
print(sep)
print(f"{'Rk':<4} {'Type':<11} {'Feature':<38} {'RF':>7} {'SHAP':>7} {'Chi2':>7}")
print(f"{'──':<4} {'─────────':<11} {'──────────────────────────────────────':<38} {'───────':>7} {'───────':>7} {'───────':>7}")
for _, row in grouped.head(30).iterrows():
    feat = str(row["original_feature"])[:37]
    print(f"{int(row['final_rank']):<4} "
          f"{str(row['type']):<11} "
          f"{feat:<38} "
          f"{row['rf_importance_sum']:>7.4f} "
          f"{row['shap_mean_abs_sum']:>7.4f} "
          f"{row['chi2_neg_log10_p']:>7.2f}")

print(f"\n{sep}")
print(f"  TOP 30 — Encoded columns (binary level)")
print(sep)
print(f"{'Rk':<4} {'Type':<11} {'Encoded Column':<40} {'RF':>7} {'SHAP':>7}")
print(f"{'──':<4} {'─────────':<11} {'────────────────────────────────────────':<40} {'───────':>7} {'───────':>7}")
for _, row in master_enc.head(30).iterrows():
    col  = str(row["encoded_feature"])[:39]
    print(f"{int(row['final_rank_encoded']):<4} "
          f"{str(row['type']):<11} "
          f"{col:<40} "
          f"{row['rf_importance']:>7.4f} "
          f"{row['shap_mean_abs']:>7.4f}")

print(f"\n{sep}")
print(f"  SAFE-only TOP 20 (leakage removed — real model view)")
print(sep)
safe_only = grouped[grouped["type"] == "[SAFE]"].head(20).reset_index(drop=True)
print(f"{'Rk':<4} {'Feature':<42} {'RF':>7} {'SHAP':>7} {'Chi2':>7}")
print(f"{'──':<4} {'──────────────────────────────────────────':<42} {'───────':>7} {'───────':>7} {'───────':>7}")
for i, row in safe_only.iterrows():
    feat = str(row["original_feature"])[:41]
    print(f"{i+1:<4} {feat:<42} "
          f"{row['rf_importance_sum']:>7.4f} "
          f"{row['shap_mean_abs_sum']:>7.4f} "
          f"{row['chi2_neg_log10_p']:>7.2f}")

# Summary counts
n_leakage = (grouped["type"] == "[LEAKAGE]").sum()
n_safe     = (grouped["type"] == "[SAFE]").sum()
print(f"\n  Total features ranked: {len(grouped)}")
print(f"  [LEAKAGE] columns: {n_leakage}  |  [SAFE] columns: {n_safe}")

# Save text report
report_path = os.path.join(OUT_DIR, "epds_ranking_all_summary.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"EPDS Feature Ranking — ALL columns (including EPDS/PHQ-9 items)\n")
    f.write(f"Target: {TARGET}\n{'=' * 72}\n\n")
    f.write("TOP 30 Original Features (all 3 methods combined)\n")
    f.write(f"{'Rk':<4} {'Type':<11} {'Feature':<38} {'RF':>7} {'SHAP':>7} {'Chi2':>7}\n")
    f.write(f"{'-' * 72}\n")
    for _, row in grouped.head(30).iterrows():
        feat = str(row["original_feature"])[:37]
        f.write(f"{int(row['final_rank']):<4} {str(row['type']):<11} {feat:<38} "
                f"{row['rf_importance_sum']:>7.4f} "
                f"{row['shap_mean_abs_sum']:>7.4f} "
                f"{row['chi2_neg_log10_p']:>7.2f}\n")
    f.write(f"\n\nSAFE-only TOP 20\n{'-' * 72}\n")
    for i, row in safe_only.iterrows():
        feat = str(row["original_feature"])[:41]
        f.write(f"{i+1:<4} {feat:<42} "
                f"{row['rf_importance_sum']:>7.4f} "
                f"{row['shap_mean_abs_sum']:>7.4f} "
                f"{row['chi2_neg_log10_p']:>7.2f}\n")

print(f"\n  Saved text report -> {report_path}")
print(f"\n{'=' * 65}")
print("  OUTPUT FILES")
print(f"{'=' * 65}")
print(f"  1. epds_ranking_all_encoded.csv   — every encoded column ranked")
print(f"  2. epds_ranking_all_original.csv  — every original feature ranked")
print(f"  3. epds_ranking_all_summary.txt   — top-30 text report")
print(f"{'=' * 65}")
print("  DONE\n")
