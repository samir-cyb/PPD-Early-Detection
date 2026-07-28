"""
03_association_eda.py
======================
Statistical association between every SAFE input feature (46 raw columns,
no EPDS/PHQ-9 items or scores) and each of the 3 possible targets:
  - EPDS Result   (Low / Medium / High)
  - PHQ9 Result   (Minimal / Mild / Moderate / Moderately Severe / Severe)
  - PPD_binary    (0 / 1, EPDS Score >= 13)

This mirrors "Table 1: Risk Factors for Postpartum Depression" in the
reference IEEE-style paper (chi-square tests per feature) and gives you a
data-driven shortlist of which raw features matter most -- useful both as
a sanity check and to justify which raw columns go into the 4 planned
composite indices in the next step (04_feature_engineering.py).

Method
------
- Categorical safe feature vs. categorical target -> Chi-square test of
  independence (scipy.stats.chi2_contingency).
- Numeric safe feature (Age, Number of the latest pregnancy) vs. categorical
  target -> One-way ANOVA (scipy.stats.f_oneway) across the target's groups.
  Kruskal-Wallis (non-parametric) is also reported since Likert/ordinal-like
  distributions are often non-normal.

Output (./outputs/):
  - association_EPDS_Result.csv
  - association_PHQ9_Result.csv
  - association_PPD_binary.csv
  (each sorted by p-value ascending = most significant first)

Usage
-----
    pip install scipy --break-system-packages   # if needed, in addition to
                                                 # pandas/numpy already installed
    python 03_association_eda.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, kruskal

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
CLEANED_PATH = os.path.join(OUT_DIR, "PPD_dataset_cleaned_v2.csv")

if not os.path.exists(CLEANED_PATH):
    raise SystemExit(
        f"Cleaned dataset not found at {CLEANED_PATH}.\n"
        f"Run 02_data_cleaning.py first."
    )

df = pd.read_csv(CLEANED_PATH)
print(f"Loaded cleaned dataset: {df.shape}")

# ---------------------------------------------------------------------------
# 1. Define columns
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
TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
SCORE_COLS = ["EPDS Score", "PHQ9 Score"]

LEAKAGE_COLS = EPDS_ITEMS + PHQ9_ITEMS + SCORE_COLS + TARGETS
LEAKAGE_COLS = [c for c in LEAKAGE_COLS if c in df.columns]

# missing-indicator flag columns created by the cleaning script -- keep them
# in the safe-feature set, they are legitimate (did this field require
# imputation or not).
SAFE_FEATURES = [c for c in df.columns if c not in LEAKAGE_COLS]

print(f"\nSafe input features being tested: {len(SAFE_FEATURES)}")
print(f"Targets: {TARGETS}")

# Only these two are genuinely continuous/count numeric features. Everything
# else that happens to be int64 (e.g. the 0/1 "_was_missing" flag columns
# created by the cleaning script) is conceptually categorical and is tested
# with chi-square, not ANOVA.
NUMERIC_FEATURES_EXPLICIT = ["Age", "Number of the latest pregnancy"]
numeric_features = [c for c in NUMERIC_FEATURES_EXPLICIT if c in SAFE_FEATURES]
categorical_features = [c for c in SAFE_FEATURES if c not in numeric_features]

print(f"  Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"  Categorical features ({len(categorical_features)})")


# ---------------------------------------------------------------------------
# 2. Association test helpers
# ---------------------------------------------------------------------------
def chi_square_test(feature, target):
    """Chi-square test of independence between two categorical columns."""
    contingency = pd.crosstab(df[feature], df[target])
    # chi2_contingency requires no zero-row/col issues; it handles them fine
    # as long as the table isn't degenerate (e.g. only 1 category).
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return np.nan, np.nan, np.nan
    chi2, p, dof, _ = chi2_contingency(contingency)
    return chi2, p, dof


def numeric_group_tests(feature, target):
    """ANOVA + Kruskal-Wallis for a numeric feature across target groups."""
    groups = [g[feature].dropna().values for _, g in df.groupby(target)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan, np.nan
    f_stat, anova_p = f_oneway(*groups)
    h_stat, kw_p = kruskal(*groups)
    return f_stat, anova_p, h_stat, kw_p


# ---------------------------------------------------------------------------
# 3. Run tests for each target
# ---------------------------------------------------------------------------
for target in TARGETS:
    print(f"\n{'=' * 90}\nASSOCIATION TESTS vs TARGET: {target}\n{'=' * 90}")

    rows = []

    for feat in categorical_features:
        chi2, p, dof = chi_square_test(feat, target)
        rows.append({
            "feature": feat,
            "feature_type": "categorical",
            "test": "chi-square",
            "statistic": chi2,
            "dof": dof,
            "p_value": p,
        })

    for feat in numeric_features:
        f_stat, anova_p, h_stat, kw_p = numeric_group_tests(feat, target)
        rows.append({
            "feature": feat,
            "feature_type": "numeric",
            "test": "ANOVA",
            "statistic": f_stat,
            "dof": np.nan,
            "p_value": anova_p,
        })
        rows.append({
            "feature": feat,
            "feature_type": "numeric",
            "test": "Kruskal-Wallis",
            "statistic": h_stat,
            "dof": np.nan,
            "p_value": kw_p,
        })

    results = pd.DataFrame(rows).sort_values("p_value", ascending=True)
    results["significant_at_0.05"] = results["p_value"] < 0.05

    out_path = os.path.join(
        OUT_DIR, f"association_{target.replace(' ', '_')}.csv"
    )
    results.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    print(f"\nTop 15 most significant features for {target}:")
    print(results.head(15).to_string(index=False))

    n_sig = results["significant_at_0.05"].sum()
    print(f"\n{n_sig} / {len(results)} tests significant at p < 0.05")

print("\n\nDone. Use the 'Top 15' lists above (and the saved CSVs) to sanity-"
      "check the 4 planned composite indices in 04_feature_engineering.py -- "
      "the raw columns feeding each index should generally show up as "
      "significant here.")
