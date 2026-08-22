"""
27_chi_square_only_ranking.py
==============================
Feature ranking based ONLY on Chi-Square test + Cramer's V effect size.
NO Random Forest, NO SHAP, NO encoding — pure statistical association.
Much faster than script 24 (seconds instead of minutes).

Target: EPDS Result (3-class) by default. Change TARGET below for PPD_binary.
Output (./outputs/):
  - 27_chi_square_ranking.csv
  - 27_chi_square_ranking.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
CLEANED_PATH = os.path.join(OUT_DIR, "PPD_dataset_cleaned_v2.csv")

if not os.path.exists(CLEANED_PATH):
    raise SystemExit(f"File not found: {CLEANEN_PATH}\nRun 02_data_cleaning.py first.")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG: Change target here if you want PPD_binary instead of EPDS Result
# ═══════════════════════════════════════════════════════════════════════════
TARGET = "EPDS Result"          # Options: "EPDS Result"  or  "PPD_binary"
# TARGET = "PPD_binary"         # <-- Uncomment this line for binary target

# Leakage columns (same definitions as script 24)
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

# ═══════════════════════════════════════════════════════════════════════════
# NEW: Exclude technical imputation-flag columns (created in 02_data_cleaning)
# These are not clinical features — just indicators of missingness.
# ═══════════════════════════════════════════════════════════════════════════
EXCLUDE_COLS = [
    "Abuse_was_missing",
    "Husband's monthly income_was_missing",
    "Husband's education level_was_missing",
    "Trust and share feelings_was_missing",
]

# Index-like columns to drop
INDEX_LIKE = {"serial", "id", "sl", "sl.", "serial no", "serial no."}

print("=" * 75)
print("  Script 27: Chi-Square ONLY Feature Ranking")
print(f"  Target: {TARGET}")
print("  Methods: (1) Chi-Square test  (2) Cramer's V (effect size)")
print("  Excluded: 4 '_was_missing' technical flag columns")
print("=" * 75)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Load cleaned dataset
# ═══════════════════════════════════════════════════════════════════════════
df_raw = pd.read_csv(CLEANED_PATH)
print(f"\n[Step 1] Loaded cleaned dataset: {df_raw.shape}")

# Drop index columns, OTHER targets, and the 4 excluded _was_missing flags
OTHER_TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
drop_cols = [c for c in df_raw.columns
             if c.lower() in INDEX_LIKE
             or (c in OTHER_TARGETS and c != TARGET)
             or c in EXCLUDE_COLS]          # <-- NEW: exclude the 4 flags

df_raw = df_raw.drop(columns=drop_cols, errors="ignore")

if TARGET not in df_raw.columns:
    raise SystemExit(f"Target '{TARGET}' not found in dataset.")

y_series = df_raw[TARGET].copy()
df_features = df_raw.drop(columns=[TARGET])

print(f"  Original features to test: {df_features.shape[1]}")
print(f"  Target classes: {sorted(y_series.dropna().unique())}")

# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════
def cramers_v(chi2, n, r, k):
    """
    Cramer's V: international standard effect size for categorical association.
    Range: 0 (no association) to 1 (perfect association).
    """
    if n == 0:
        return 0.0
    denom = n * (min(k - 1, r - 1))
    if denom == 0:
        return 0.0
    return np.sqrt(chi2 / denom)

def leakage_label(col):
    return "[LEAKAGE]" if col in ALL_LEAKAGE else "[SAFE]"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Chi-Square test on ALL original columns (excluding the 4 flags)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Step 2] Running Chi-Square test on every feature...")

results = []
n_total = len(df_features.columns)

for idx, col in enumerate(df_features.columns, 1):
    # Progress indicator
    if idx % 10 == 0 or idx == 1 or idx == n_total:
        print(f"  ... {idx:3d}/{n_total}: {col[:45]}")

    try:
        # Build contingency table: feature vs target
        ct = pd.crosstab(df_features[col], y_series)

        # Skip degenerate tables (only 1 category)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            results.append({
                "original_feature": col,
                "type": leakage_label(col),
                "chi2_statistic": np.nan,
                "chi2_p_value": np.nan,
                "neg_log10_p": 0.0,
                "cramers_v": 0.0,
                "significant_p05": "No (single category)",
                "n_categories_feature": int(df_features[col].nunique(dropna=True)),
                "n_categories_target": int(y_series.nunique(dropna=True)),
            })
            continue

        # Chi-square test
        chi2_stat, p_val, dof, expected = chi2_contingency(ct)

        # Cramer's V effect size
        r = ct.shape[0]  # feature categories
        k = ct.shape[1]  # target categories
        cv = cramers_v(chi2_stat, ct.sum().sum(), r, k)

        results.append({
            "original_feature": col,
            "type": leakage_label(col),
            "chi2_statistic": chi2_stat,
            "chi2_p_value": p_val,
            "neg_log10_p": round(-np.log10(p_val + 1e-300), 4),
            "cramers_v": round(cv, 4),
            "significant_p05": "Yes" if p_val < 0.05 else "No",
            "n_categories_feature": r,
            "n_categories_target": k,
        })

    except Exception as e:
        results.append({
            "original_feature": col,
            "type": leakage_label(col),
            "chi2_statistic": np.nan,
            "chi2_p_value": np.nan,
            "neg_log10_p": 0.0,
            "cramers_v": 0.0,
            "significant_p05": f"Error",
            "n_categories_feature": int(df_features[col].nunique(dropna=True)),
            "n_categories_target": int(y_series.nunique(dropna=True)),
        })

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Rank by Chi-Square significance (p-value), then by Cramer's V
# ═══════════════════════════════════════════════════════════════════════════
print("\n[Step 3] Ranking features...")

rank_df = pd.DataFrame(results)

# Sort: most significant first (smallest p-value), tie-break by largest Cramer's V
rank_df = rank_df.sort_values(
    by=["chi2_p_value", "cramers_v"],
    ascending=[True, False]
).reset_index(drop=True)

rank_df.insert(0, "rank", range(1, len(rank_df) + 1))

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Save CSV
# ═══════════════════════════════════════════════════════════════════════════
csv_path = os.path.join(OUT_DIR, "27_chi_square_ranking.csv")
rank_df.to_csv(csv_path, index=False)
print(f"\n[Step 4] Saved CSV -> {csv_path}")
print(f"  Total features ranked: {len(rank_df)}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Print ALL features in console log
# ═══════════════════════════════════════════════════════════════════════════
sep = "=" * 95

print(f"\n{sep}")
print(f"  FULL RANKING: ALL {len(rank_df)} FEATURES (Chi-Square + Cramer's V)")
print(f"  Target: {TARGET}")
print(f"  [LEAKAGE] = EPDS/PHQ-9 items + raw scores (excluded in real modeling)")
print(f"  [SAFE]    = demographic, social, economic, clinical features")
print(f"  EXCLUDED  = 4 '_was_missing' technical flag columns")
print(sep)

# Header
hdr = (f"{'Rk':<5} {'Type':<11} {'Feature':<40} {'Chi2':>10} {'p-value':>12} "
       f"{'-log10(p)':>10} {'CramersV':>10} {'Sig?':>6}")
print(hdr)
print("-" * 95)

# Every single feature printed
for _, row in rank_df.iterrows():
    feat = str(row["original_feature"])[:38]
    chi2_str = f"{row['chi2_statistic']:>10.2f}" if pd.notna(row['chi2_statistic']) else f"{'N/A':>10}"
    p_str = f"{row['chi2_p_value']:>12.2e}" if pd.notna(row['chi2_p_value']) else f"{'N/A':>12}"
    neglog = f"{row['neg_log10_p']:>10.2f}"
    cram = f"{row['cramers_v']:>10.4f}"
    sig = str(row["significant_p05"])[:6]

    print(f"{int(row['rank']):<5} {str(row['type']):<11} {feat:<40} "
          f"{chi2_str} {p_str} {neglog} {cram} {sig:>6}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Summary statistics
# ═══════════════════════════════════════════════════════════════════════════
n_sig = (rank_df["significant_p05"] == "Yes").sum()
n_leakage = (rank_df["type"] == "[LEAKAGE]").sum()
n_safe = (rank_df["type"] == "[SAFE]").sum()

print(f"\n{sep}")
print("  SUMMARY")
print(sep)
print(f"  Total features tested:     {len(rank_df)}")
print(f"  Significant at p < 0.05:   {n_sig} / {len(rank_df)} ({n_sig/len(rank_df)*100:.1f}%)")
print(f"  [LEAKAGE] columns:         {n_leakage}")
print(f"  [SAFE] columns:            {n_safe}")
print(f"  Excluded flag columns:     {len(EXCLUDE_COLS)}")

# Top 10 by category
print(f"\n{sep}")
print("  TOP 10 LEAKAGE FEATURES (by Chi-Square)")
print(sep)
leakage_top = rank_df[rank_df["type"] == "[LEAKAGE]"].head(10)
if not leakage_top.empty:
    for _, row in leakage_top.iterrows():
        print(f"  {int(row['rank']):<5} {str(row['original_feature'])[:50]:<50}  "
              f"-log10(p)={row['neg_log10_p']:.2f}, CramersV={row['cramers_v']:.4f}")
else:
    print("  None.")

print(f"\n{sep}")
print("  TOP 10 SAFE FEATURES (by Chi-Square)")
print(sep)
safe_top = rank_df[rank_df["type"] == "[SAFE]"].head(10)
if not safe_top.empty:
    for _, row in safe_top.iterrows():
        print(f"  {int(row['rank']):<5} {str(row['original_feature'])[:50]:<50}  "
              f"-log10(p)={row['neg_log10_p']:.2f}, CramersV={row['cramers_v']:.4f}")
else:
    print("  None.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Save full text report
# ═══════════════════════════════════════════════════════════════════════════
report_path = os.path.join(OUT_DIR, "27_chi_square_ranking.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Feature Ranking — Chi-Square ONLY (Script 27)\n")
    f.write(f"Target: {TARGET}\n")
    f.write(f"Excluded columns: {', '.join(EXCLUDE_COLS)}\n")
    f.write(f"{'='*95}\n\n")
    f.write(f"Total features: {len(rank_df)}\n")
    f.write(f"Significant (p<0.05): {n_sig} / {len(rank_df)}\n\n")
    f.write(hdr + "\n")
    f.write("-" * 95 + "\n")
    for _, row in rank_df.iterrows():
        feat = str(row["original_feature"])[:38]
        chi2_str = f"{row['chi2_statistic']:>10.2f}" if pd.notna(row['chi2_statistic']) else f"{'N/A':>10}"
        p_str = f"{row['chi2_p_value']:>12.2e}" if pd.notna(row['chi2_p_value']) else f"{'N/A':>12}"
        neglog = f"{row['neg_log10_p']:>10.2f}"
        cram = f"{row['cramers_v']:>10.4f}"
        sig = str(row["significant_p05"])[:6]
        f.write(f"{int(row['rank']):<5} {str(row['type']):<11} {feat:<40} "
                f"{chi2_str} {p_str} {neglog} {cram} {sig:>6}\n")

print(f"\n[Step 5] Saved full text report -> {report_path}")
print(f"\n{'='*75}")
print("  DONE")
print(f"{'='*75}")