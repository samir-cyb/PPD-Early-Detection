"""
13_composite_weight_learning.py
=================================
Tier-2 improvement idea #1: DATA-DRIVEN composite weights.

04_feature_engineering.py's "weighted" composite indices combine each
domain's sub-features using a manually-derived weight: -log10(p) from a
chi-square test of each sub-feature against PPD_binary, normalized to sum
to 1. This is data-driven in the sense that the weight comes from the data,
but it is still a heuristic (there's no guarantee -log10(p) is the
statistically optimal way to combine sub-features for prediction).

This script tries a more direct alternative: for each of the 4 domains, fit
a small Logistic Regression using ONLY that domain's normalized
sub-features to predict PPD_binary, and use the model's OWN predicted
probability as the composite index (scaled to 0-10). This lets the model
learn the optimal linear combination of sub-features directly from the
outcome, rather than assuming -log10(p) is the right weighting scheme.

Both composite variants (existing chi-square-weighted vs this script's
LR-learned) are compared two ways:
  1. Association strength -- ANOVA/Kruskal-Wallis against all 3 targets
     (same method as 04_feature_engineering.py's own validation step).
  2. Predictive contribution -- swap the LR-learned composites into the
     Arm B "Raw + Composite" feature set (in place of the chi-square-
     weighted ones) and compare 5-fold CV F1-macro, identical folds, for
     all 3 targets.

Output (./outputs/):
  - composite_weight_comparison_association.csv
  - composite_weight_comparison_predictive.csv
  - composite_weight_learning_summary.txt

Usage
-----
    python 13_composite_weight_learning.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
CLEANED_PATH = os.path.join(OUT_DIR, "PPD_dataset_cleaned_v2.csv")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")

if not os.path.exists(CLEANED_PATH):
    raise SystemExit(f"{CLEANED_PATH} not found. Run 02_data_cleaning.py first.")
if not os.path.exists(MODEL_READY_PATH):
    raise SystemExit(f"{MODEL_READY_PATH} not found. Run 06_encoding.py first.")

df = pd.read_csv(CLEANED_PATH)
model_ready = pd.read_csv(MODEL_READY_PATH)
print(f"Loaded cleaned dataset: {df.shape}, model-ready dataset: {model_ready.shape}")

RANDOM_STATE = 42
TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]

# ---------------------------------------------------------------------------
# 1. Rebuild the 4 domains' normalized sub-feature tables -- IDENTICAL maps
#    and logic to 04_feature_engineering.py, so the comparison is apples-
#    to-apples against the existing chi-square-weighted composites.
# ---------------------------------------------------------------------------
def ordinal_normalize(series, mapping):
    scored = series.map(mapping)
    lo, hi = min(mapping.values()), max(mapping.values())
    if hi == lo:
        return scored * 0
    return (scored - lo) / (hi - lo)


INCOME_MAP = {"No Personal Income": 0, "Less than 5000": 1, "5000 to 10000": 2,
              "10000 to 20000": 3, "20000 to 30000": 4, "More than 30000": 5}
OCCUPATION_MAP = {"Housewife": 0, "Student": 0, "Other": 1, "Service": 2,
                   "Teacher": 3, "Business": 4, "Doctor": 5}
EDUCATION_MAP = {"University": 1, "College": 2, "High School": 3, "Primary School": 4}
FAMILY_TYPE_MAP = {"Joint": 0, "Nuclear": 1}
RELATIONSHIP_MAP = {"Bad": 0, "Poor": 1, "Neutral": 2, "Good": 3, "Friendly": 4}
HOUSEHOLD_MAP = {"9 or more": 0, "6 to 8": 1, "2 to 5": 2}
SUPPORT_MAP = {"Low": 0, "Medium": 1, "High": 2}
PHQ2_MAP = {"Negative": 0, "Positive": 1}
DISEASE_MAP = {"No Disease": 0, "Non-Chronic Disease": 1, "Chronic Disease": 2}
ABUSE_MAP = {"Yes": 0, "No": 1}
LOSS_MAP = {"No Pregnancy Loss": 0, "Miscarriage": 1, "Still-born Delivery": 2}
DELIVERY_MAP = {"Normal Delivery": 0, "Caesarean Section": 1}
YESNO_RISK_MAP = {"No": 0, "Yes": 1}

esi_parts = pd.DataFrame({
    "income_now": ordinal_normalize(df["Current monthly income"], INCOME_MAP),
    "husband_income": ordinal_normalize(df["Husband's monthly income"], INCOME_MAP),
    "occupation": ordinal_normalize(df["Occupation After Your Latest Childbirth"],
                                     OCCUPATION_MAP),
    "education": ordinal_normalize(df["Education Level"], EDUCATION_MAP),
})
ssi_parts = pd.DataFrame({
    "family_type": ordinal_normalize(df["Family type"], FAMILY_TYPE_MAP),
    "relationship_in_laws": ordinal_normalize(df["Relationship with the in-laws"],
                                               RELATIONSHIP_MAP),
    "relationship_husband": ordinal_normalize(df["Relationship with husband"],
                                               RELATIONSHIP_MAP),
    "household_members": ordinal_normalize(df["Number of household members"],
                                            HOUSEHOLD_MAP),
    "received_support": ordinal_normalize(df["Received Support"], SUPPORT_MAP),
})
mhri_parts = pd.DataFrame({
    "phq2_before": ordinal_normalize(df["Depression before pregnancy (PHQ2)"], PHQ2_MAP),
    "phq2_during": ordinal_normalize(df["Depression during pregnancy (PHQ2)"], PHQ2_MAP),
    "disease_before": ordinal_normalize(df["Disease before pregnancy"], DISEASE_MAP),
    "abuse": ordinal_normalize(df["Abuse"], ABUSE_MAP),
    "pregnancy_loss": ordinal_normalize(df["History of pregnancy loss"], LOSS_MAP),
})
nsi_parts = pd.DataFrame({
    "delivery_mode": ordinal_normalize(df["Mode of delivery"], DELIVERY_MAP),
    "birth_complications": ordinal_normalize(df["Birth complications"], YESNO_RISK_MAP),
    "newborn_illness": ordinal_normalize(df["Newborn illness"], YESNO_RISK_MAP),
    "worry_newborn": ordinal_normalize(df["Worry about newborn"], YESNO_RISK_MAP),
})

DOMAINS = {
    "Economic_Stability_Index_LR": esi_parts,
    "Social_Support_Index_LR": ssi_parts,
    "Maternal_MentalHealth_Risk_Index_LR": mhri_parts,
    "Neonatal_Delivery_Stress_Index_LR": nsi_parts,
}

# ---------------------------------------------------------------------------
# 2. Fit a small Logistic Regression per domain, use its predicted
#    probability (scaled x10) as the new, data-driven composite index.
# ---------------------------------------------------------------------------
y_binary = df["PPD_binary"].values
print(f"\n{'=' * 90}\nLEARNING PER-DOMAIN LOGISTIC REGRESSION WEIGHTS\n{'=' * 90}")
for name, parts_df in DOMAINS.items():
    lr = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(parts_df, y_binary)
    df[name] = lr.predict_proba(parts_df)[:, 1] * 10
    coef_summary = {col: round(c, 3) for col, c in
                     zip(parts_df.columns, lr.coef_[0])}
    print(f"\n{name}:")
    print(f"  learned coefficients (higher |coef| = more influence "
          f"within this domain): {coef_summary}")
    print(f"  index summary: min={df[name].min():.2f}, "
          f"mean={df[name].mean():.2f}, max={df[name].max():.2f}")

LR_LEARNED_COMPOSITES = list(DOMAINS.keys())
WEIGHTED_COMPOSITES = [
    "Economic_Stability_Index_Weighted", "Social_Support_Index_Weighted",
    "Maternal_MentalHealth_Risk_Index_Weighted",
    "Neonatal_Delivery_Stress_Index_Weighted",
]

# ---------------------------------------------------------------------------
# 3. Association test comparison: LR-learned vs chi-square-weighted, both
#    against all 3 targets (same method as 04_feature_engineering.py).
# ---------------------------------------------------------------------------
print(f"\n{'=' * 90}\nASSOCIATION TEST: LR-learned vs chi-square-weighted composites\n{'=' * 90}")
# Need the original weighted composites too -- rebuild them by reading the
# already-engineered file (they're saved there from script 04).
engineered_path = os.path.join(OUT_DIR, "PPD_dataset_with_composite_features.csv")
if os.path.exists(engineered_path):
    engineered = pd.read_csv(engineered_path)
    for col in WEIGHTED_COMPOSITES:
        df[col] = engineered[col].values
else:
    print(f"[warn] {engineered_path} not found -- skipping direct comparison "
          f"against the chi-square-weighted composites' values.")
    WEIGHTED_COMPOSITES = []

assoc_rows = []
for target in TARGETS:
    for col in LR_LEARNED_COMPOSITES + WEIGHTED_COMPOSITES:
        groups = [g[col].dropna().values for _, g in df.groupby(target)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, anova_p = f_oneway(*groups)
        h_stat, kw_p = kruskal(*groups)
        assoc_rows.append({
            "target": target, "composite_index": col,
            "variant": "LR-learned" if col in LR_LEARNED_COMPOSITES else "Chi-square-weighted",
            "anova_p": anova_p, "kruskal_p": kw_p,
            "significant_at_0.05": anova_p < 0.05,
        })

assoc_df = pd.DataFrame(assoc_rows)
print(assoc_df.to_string(index=False))
assoc_path = os.path.join(OUT_DIR, "composite_weight_comparison_association.csv")
assoc_df.to_csv(assoc_path, index=False)
print(f"\nSaved -> {assoc_path}")

# ---------------------------------------------------------------------------
# 4. Predictive comparison: swap composites into Arm B, compare 5-fold CV
#    F1-macro on identical folds, for all 3 targets.
# ---------------------------------------------------------------------------
print(f"\n{'=' * 90}\nPREDICTIVE COMPARISON: Arm B with each composite variant\n{'=' * 90}")
EQUAL_WEIGHT_COMPOSITES = [
    "Economic_Stability_Index", "Social_Support_Index",
    "Maternal_MentalHealth_Risk_Index", "Neonatal_Delivery_Stress_Index",
]
model_ready_targets = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
raw_only_cols = [c for c in model_ready.columns
                 if c not in model_ready_targets + EQUAL_WEIGHT_COMPOSITES
                 + WEIGHTED_COMPOSITES]

pred_rows = []
for target in TARGETS:
    le = LabelEncoder()
    y = le.fit_transform(model_ready[target])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    X_raw_only = model_ready[raw_only_cols].reset_index(drop=True)

    variants = {"LR-learned composites": LR_LEARNED_COMPOSITES}
    if WEIGHTED_COMPOSITES:
        variants["Chi-square-weighted composites (current)"] = WEIGHTED_COMPOSITES

    for variant_name, cols in variants.items():
        X_variant = pd.concat(
            [X_raw_only, df[cols].reset_index(drop=True)], axis=1)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=RANDOM_STATE)
        scores = cross_val_score(rf, X_variant, y, cv=skf, scoring="f1_macro")
        pred_rows.append({
            "target": target, "composite_variant": variant_name,
            "n_features": X_variant.shape[1],
            "cv_f1_macro_mean": scores.mean(), "cv_f1_macro_std": scores.std(),
        })
        print(f"  {target} / {variant_name}: CV F1-macro = "
              f"{scores.mean():.3f} +/- {scores.std():.3f}")

pred_df = pd.DataFrame(pred_rows)
pred_path = os.path.join(OUT_DIR, "composite_weight_comparison_predictive.csv")
pred_df.to_csv(pred_path, index=False)
print(f"\nSaved -> {pred_path}")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
summary_lines = []
n_lr_sig = int(assoc_df[assoc_df["variant"] == "LR-learned"]["significant_at_0.05"].sum())
n_lr_total = int((assoc_df["variant"] == "LR-learned").sum())
summary_lines.append(
    f"LR-learned composites: {n_lr_sig}/{n_lr_total} association tests significant.")
if WEIGHTED_COMPOSITES:
    n_cs_sig = int(assoc_df[assoc_df["variant"] == "Chi-square-weighted"]["significant_at_0.05"].sum())
    n_cs_total = int((assoc_df["variant"] == "Chi-square-weighted").sum())
    summary_lines.append(
        f"Chi-square-weighted composites: {n_cs_sig}/{n_cs_total} association tests significant.")

for target in TARGETS:
    sub = pred_df[pred_df["target"] == target].sort_values(
        "cv_f1_macro_mean", ascending=False)
    best = sub.iloc[0]
    summary_lines.append(f"{target}: best composite variant by CV F1-macro = "
                          f"{best['composite_variant']} ({best['cv_f1_macro_mean']:.3f})")

summary_path = os.path.join(OUT_DIR, "composite_weight_learning_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
print("\n".join(summary_lines))
print(f"Saved -> {summary_path}")
print("\nDone. If LR-learned composites match or beat the chi-square-weighted "
      "ones on both association strength and predictive CV F1-macro, they are "
      "a defensible, more principled replacement; if not, the existing "
      "chi-square-weighted composites remain the right choice.")
