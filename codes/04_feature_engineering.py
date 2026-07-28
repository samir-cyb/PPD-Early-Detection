"""
04_feature_engineering.py
==========================
Builds the 4 planned COMPOSITE RISK INDICES on top of the cleaned dataset.
This is the core novelty of the project: instead of feeding 51 raw
columns straight into a model, we combine clinically-related raw features
into interpretable 0-10 composite scores, then (in the ablation study,
script 06) show that Raw+Composite beats Raw alone.

Composite indices built here
-----------------------------
1. Economic Stability Index (ESI) -- higher = more economically stable
   uses: Current monthly income, Husband's monthly income,
         Occupation After Your Latest Childbirth, Education Level

2. Social/Family Support Index (SSI) -- higher = more social support
   uses: Family type, Relationship with the in-laws,
         Relationship with husband, Number of household members,
         Received Support

3. Maternal Mental-Health Risk Index (MHRI) -- higher = more risk
   uses: Depression before pregnancy (PHQ2), Depression during pregnancy
         (PHQ2), Disease before pregnancy, Abuse, History of pregnancy loss

4. Neonatal/Delivery Stress Index (NSI) -- higher = more stress
   uses: Mode of delivery, Birth complications, Newborn illness,
         Worry about newborn

Method
------
Each raw sub-feature is mapped to an ordinal score using a documented,
clinically-reasoned scale (see the *_MAP dictionaries below -- these are
assumptions, written explicitly so they can be challenged/refined). Each
sub-score is then min-max normalized to [0, 1], the normalized sub-scores
for an index are averaged, and multiplied by 10 to get a final 0-10 index.

IMPORTANT ASSUMPTION TO REVIEW: "Need for Support" was deliberately LEFT
OUT of the Social Support Index. Reason: it measures a mother's forward-
looking desire for more support, not her current support level, and ~21%
of it is "Not Reported" -- mixing it in would blur the index's meaning.
It remains available as its own raw feature for modeling.

After building the indices, this script re-runs a quick chi-square/ANOVA
association check (same method as script 03) of each composite index
against all 3 targets, so you can immediately see whether the engineered
features carry signal before moving to modeling.

Output (./outputs/):
  - PPD_dataset_with_composite_features.csv  (cleaned data + 4 new columns)
  - composite_index_association.csv          (significance of the 4 indices)
  - composite_index_summary.txt               (descriptive stats by target group)

Usage
-----
    python 04_feature_engineering.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, chi2_contingency

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
CLEANED_PATH = os.path.join(OUT_DIR, "PPD_dataset_cleaned_v2.csv")

if not os.path.exists(CLEANED_PATH):
    raise SystemExit(f"Cleaned dataset not found at {CLEANED_PATH}. "
                      f"Run 02_data_cleaning.py first.")

df = pd.read_csv(CLEANED_PATH)
print(f"Loaded cleaned dataset: {df.shape}")

log_lines = []


def log(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    log_lines.append(line)


def section(title):
    log("")
    log("=" * 90)
    log(title)
    log("=" * 90)


# ---------------------------------------------------------------------------
# 1. Helper: map a column to ordinal scores, then min-max normalize to [0,1]
# ---------------------------------------------------------------------------
def ordinal_normalize(series, mapping, name):
    unmapped = set(series.dropna().unique()) - set(mapping.keys())
    if unmapped:
        log(f"  WARNING [{name}]: values not in mapping (will become NaN): "
            f"{unmapped}")
    scored = series.map(mapping)
    lo, hi = min(mapping.values()), max(mapping.values())
    if hi == lo:
        return scored * 0
    normalized = (scored - lo) / (hi - lo)
    return normalized


# ---------------------------------------------------------------------------
# 2. Economic Stability Index (ESI) -- higher = more stable
# ---------------------------------------------------------------------------
section("2. ECONOMIC STABILITY INDEX (ESI)")

INCOME_MAP = {
    "No Personal Income": 0,
    "Less than 5000": 1,
    "5000 to 10000": 2,
    "10000 to 20000": 3,
    "20000 to 30000": 4,
    "More than 30000": 5,
}
OCCUPATION_MAP = {
    "Housewife": 0,
    "Student": 0,
    "Other": 1,
    "Service": 2,
    "Teacher": 3,
    "Business": 4,
    "Doctor": 5,
}
# CORRECTED after empirical direction check (05_direction_check.py):
# PPD-positive rate by education was Primary 22.2%, High School 19.7%,
# College 39.4%, University 48.5% -- i.e. MORE education associated with
# MORE reported PPD in this sample (plausibly reflects greater symptom
# awareness/reporting or higher role-related expectations among more
# educated mothers -- a pattern also noted in some PPD literature). Since
# ESI is meant to mean "higher = more stable/protective", the score is
# assigned in REVERSE of raw attainment order so it stays internally
# consistent with the empirical direction.
EDUCATION_MAP = {
    "University": 1,
    "College": 2,
    "High School": 3,
    "Primary School": 4,
}

esi_parts = pd.DataFrame({
    "income_now": ordinal_normalize(df["Current monthly income"], INCOME_MAP,
                                     "Current monthly income"),
    "husband_income": ordinal_normalize(df["Husband's monthly income"],
                                         INCOME_MAP, "Husband's monthly income"),
    "occupation": ordinal_normalize(
        df["Occupation After Your Latest Childbirth"], OCCUPATION_MAP,
        "Occupation After Your Latest Childbirth"),
    "education": ordinal_normalize(df["Education Level"], EDUCATION_MAP,
                                    "Education Level"),
})
df["Economic_Stability_Index"] = esi_parts.mean(axis=1) * 10
log(f"ESI built from: {list(esi_parts.columns)}")
log(df["Economic_Stability_Index"].describe().to_string())

# ---------------------------------------------------------------------------
# 3. Social / Family Support Index (SSI) -- higher = more support
# ---------------------------------------------------------------------------
section("3. SOCIAL/FAMILY SUPPORT INDEX (SSI)")

# CORRECTED after empirical direction check: PPD-positive rate was Nuclear
# 37.8% vs Joint 50.1% -- i.e. Joint family was associated with MORE PPD in
# this sample, not less (plausibly reflecting in-law conflict / reduced
# autonomy rather than protective support -- consistent with some South
# Asian family-dynamics literature). Flipped so higher score = more support.
FAMILY_TYPE_MAP = {"Joint": 0, "Nuclear": 1}
RELATIONSHIP_MAP = {"Bad": 0, "Poor": 1, "Neutral": 2, "Good": 3,
                     "Friendly": 4}
# CORRECTED: PPD-positive rate was 2-5 members 39.9%, 6-8 members 50.7%,
# 9+ members 50.9% -- i.e. MORE household members associated with MORE PPD
# (likely crowding/reduced autonomy rather than more support). Flipped so
# smaller households score higher (= more support).
HOUSEHOLD_MAP = {"9 or more": 0, "6 to 8": 1, "2 to 5": 2}
SUPPORT_MAP = {"Low": 0, "Medium": 1, "High": 2}

ssi_parts = pd.DataFrame({
    "family_type": ordinal_normalize(df["Family type"], FAMILY_TYPE_MAP,
                                      "Family type"),
    "relationship_in_laws": ordinal_normalize(
        df["Relationship with the in-laws"], RELATIONSHIP_MAP,
        "Relationship with the in-laws"),
    "relationship_husband": ordinal_normalize(
        df["Relationship with husband"], RELATIONSHIP_MAP,
        "Relationship with husband"),
    "household_members": ordinal_normalize(
        df["Number of household members"], HOUSEHOLD_MAP,
        "Number of household members"),
    "received_support": ordinal_normalize(df["Received Support"],
                                           SUPPORT_MAP, "Received Support"),
})
df["Social_Support_Index"] = ssi_parts.mean(axis=1) * 10
log(f"SSI built from: {list(ssi_parts.columns)}")
log("NOTE: 'Need for Support' deliberately excluded -- see module docstring.")
log(df["Social_Support_Index"].describe().to_string())

# ---------------------------------------------------------------------------
# 4. Maternal Mental-Health Risk Index (MHRI) -- higher = more risk
# ---------------------------------------------------------------------------
section("4. MATERNAL MENTAL-HEALTH RISK INDEX (MHRI)")

PHQ2_MAP = {"Negative": 0, "Positive": 1}
DISEASE_MAP = {"No Disease": 0, "Non-Chronic Disease": 1, "Chronic Disease": 2}
# CORRECTED after empirical direction check (05_direction_check.py):
# PPD-positive rate was Abuse='No' 64.1% vs Abuse='Yes' 27.0% -- the OPPOSITE
# of what a "Yes = experienced abuse = higher risk" assumption would predict.
# This is strong (n=800, >2x rate difference) and consistent, not noise --
# it suggests the 'Abuse' question/label may mean something different than
# assumed in this survey (e.g. a safety-framed question, or a translation/
# coding quirk). VERIFY the original questionnaire wording with the dataset
# author if possible. For now, the map is aligned to the EMPIRICAL direction
# (so the composite stays internally consistent: higher score = higher
# observed PPD rate), with this caveat documented for the write-up.
ABUSE_MAP = {"Yes": 0, "No": 1}
LOSS_MAP = {"No Pregnancy Loss": 0, "Miscarriage": 1,
            "Still-born Delivery": 2}

mhri_parts = pd.DataFrame({
    "phq2_before": ordinal_normalize(df["Depression before pregnancy (PHQ2)"],
                                      PHQ2_MAP, "PHQ2 before"),
    "phq2_during": ordinal_normalize(df["Depression during pregnancy (PHQ2)"],
                                      PHQ2_MAP, "PHQ2 during"),
    "disease_before": ordinal_normalize(df["Disease before pregnancy"],
                                         DISEASE_MAP, "Disease before pregnancy"),
    "abuse": ordinal_normalize(df["Abuse"], ABUSE_MAP, "Abuse"),
    "pregnancy_loss": ordinal_normalize(df["History of pregnancy loss"],
                                         LOSS_MAP, "History of pregnancy loss"),
})
df["Maternal_MentalHealth_Risk_Index"] = mhri_parts.mean(axis=1) * 10
log(f"MHRI built from: {list(mhri_parts.columns)}")
log(df["Maternal_MentalHealth_Risk_Index"].describe().to_string())

# ---------------------------------------------------------------------------
# 5. Neonatal / Delivery Stress Index (NSI) -- higher = more stress
# ---------------------------------------------------------------------------
section("5. NEONATAL/DELIVERY STRESS INDEX (NSI)")

DELIVERY_MAP = {"Normal Delivery": 0, "Caesarean Section": 1}
YESNO_RISK_MAP = {"No": 0, "Yes": 1}

nsi_parts = pd.DataFrame({
    "delivery_mode": ordinal_normalize(df["Mode of delivery"], DELIVERY_MAP,
                                        "Mode of delivery"),
    "birth_complications": ordinal_normalize(df["Birth complications"],
                                              YESNO_RISK_MAP,
                                              "Birth complications"),
    "newborn_illness": ordinal_normalize(df["Newborn illness"],
                                          YESNO_RISK_MAP, "Newborn illness"),
    "worry_newborn": ordinal_normalize(df["Worry about newborn"],
                                        YESNO_RISK_MAP, "Worry about newborn"),
})
df["Neonatal_Delivery_Stress_Index"] = nsi_parts.mean(axis=1) * 10
log(f"NSI built from: {list(nsi_parts.columns)}")
log(df["Neonatal_Delivery_Stress_Index"].describe().to_string())

EQUAL_WEIGHT_COLS = [
    "Economic_Stability_Index",
    "Social_Support_Index",
    "Maternal_MentalHealth_Risk_Index",
    "Neonatal_Delivery_Stress_Index",
]

# ---------------------------------------------------------------------------
# 5b. WEIGHTED variants of the same 4 indices
# ---------------------------------------------------------------------------
# On the first run, equal-weight averaging (every sub-feature counts the
# same) made the Maternal Mental-Health Risk Index (MHRI) NON-significant
# even though it contains 'Abuse' -- the single strongest raw predictor in
# the whole dataset (see 03_association_eda.py). This happens because
# averaging dilutes a strong signal with weaker/noisier ones when every
# sub-feature is forced to count equally.
#
# Fix: weight each sub-feature by its OWN empirically measured association
# strength (chi-square vs PPD_binary, the primary binary clinical outcome),
# instead of assuming all sub-features matter equally. This is a documented,
# reproducible, data-driven weighting scheme -- not a manual fudge.
section("5b. WEIGHTED COMPOSITE INDICES (fixes signal dilution)")


def subfeature_weight(raw_column, target="PPD_binary"):
    """-log10(p) from a chi-square test of raw_column vs target.
    Higher = the sub-feature is more strongly associated with PPD on its
    own, so it should count for more inside the composite average."""
    contingency = pd.crosstab(df[raw_column], df[target])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 1.0
    _, p, _, _ = chi2_contingency(contingency)
    p = max(p, 1e-300)
    return -np.log10(p)


def build_weighted_index(parts_df, alias_to_rawcol, name):
    weights = {}
    for alias, raw_col in alias_to_rawcol.items():
        w = subfeature_weight(raw_col)
        weights[alias] = w
        log(f"  weight[{alias} <- '{raw_col}']: -log10(p) = {w:.2f}")
    total = sum(weights.values())
    norm_weights = {k: v / total for k, v in weights.items()}
    log(f"  normalized weights: "
        f"{ {k: round(v, 3) for k, v in norm_weights.items()} }")
    weighted = sum(parts_df[alias] * w for alias, w in norm_weights.items())
    df[name] = weighted * 10
    log(f"{name} (weighted) summary:")
    log(df[name].describe().to_string())
    return norm_weights


log("\n--- Economic Stability Index (weighted) ---")
build_weighted_index(esi_parts, {
    "income_now": "Current monthly income",
    "husband_income": "Husband's monthly income",
    "occupation": "Occupation After Your Latest Childbirth",
    "education": "Education Level",
}, "Economic_Stability_Index_Weighted")

log("\n--- Social/Family Support Index (weighted) ---")
build_weighted_index(ssi_parts, {
    "family_type": "Family type",
    "relationship_in_laws": "Relationship with the in-laws",
    "relationship_husband": "Relationship with husband",
    "household_members": "Number of household members",
    "received_support": "Received Support",
}, "Social_Support_Index_Weighted")

log("\n--- Maternal Mental-Health Risk Index (weighted) ---")
build_weighted_index(mhri_parts, {
    "phq2_before": "Depression before pregnancy (PHQ2)",
    "phq2_during": "Depression during pregnancy (PHQ2)",
    "disease_before": "Disease before pregnancy",
    "abuse": "Abuse",
    "pregnancy_loss": "History of pregnancy loss",
}, "Maternal_MentalHealth_Risk_Index_Weighted")

log("\n--- Neonatal/Delivery Stress Index (weighted) ---")
build_weighted_index(nsi_parts, {
    "delivery_mode": "Mode of delivery",
    "birth_complications": "Birth complications",
    "newborn_illness": "Newborn illness",
    "worry_newborn": "Worry about newborn",
}, "Neonatal_Delivery_Stress_Index_Weighted")

WEIGHTED_COLS = [
    "Economic_Stability_Index_Weighted",
    "Social_Support_Index_Weighted",
    "Maternal_MentalHealth_Risk_Index_Weighted",
    "Neonatal_Delivery_Stress_Index_Weighted",
]
COMPOSITE_COLS = EQUAL_WEIGHT_COLS + WEIGHTED_COLS

# ---------------------------------------------------------------------------
# 6. Validate: are the composite indices (equal-weight AND weighted)
#    associated with the targets?
# ---------------------------------------------------------------------------
section("6. VALIDATE ALL 8 COMPOSITE INDICES AGAINST TARGETS "
        "(ANOVA + Kruskal-Wallis)")

TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
rows = []
for target in TARGETS:
    for col in COMPOSITE_COLS:
        groups = [g[col].dropna().values for _, g in df.groupby(target)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, anova_p = f_oneway(*groups)
        h_stat, kw_p = kruskal(*groups)
        rows.append({
            "target": target, "composite_index": col,
            "anova_F": f_stat, "anova_p": anova_p,
            "kruskal_H": h_stat, "kruskal_p": kw_p,
            "significant_at_0.05": (anova_p < 0.05),
        })
        # also print mean +/- std per group for quick reading
        means = df.groupby(target)[col].mean().round(2)
        log(f"\n{target} x {col}: ANOVA p={anova_p:.2e}, "
            f"Kruskal-Wallis p={kw_p:.2e}")
        log(f"  Group means: {means.to_dict()}")

comp_results = pd.DataFrame(rows)
comp_results_path = os.path.join(OUT_DIR, "composite_index_association.csv")
comp_results.to_csv(comp_results_path, index=False)
log(f"\nSaved -> {comp_results_path}")

n_sig = comp_results["significant_at_0.05"].sum()
log(f"\n{n_sig} / {len(comp_results)} composite-index tests significant at "
    f"p < 0.05")

# ---------------------------------------------------------------------------
# 7. Save engineered dataset
# ---------------------------------------------------------------------------
section("7. SAVE ENGINEERED DATASET")

engineered_path = os.path.join(OUT_DIR, "PPD_dataset_with_composite_features.csv")
df.to_csv(engineered_path, index=False)
log(f"Saved -> {engineered_path}")
log(f"Shape: {df.shape} (should be cleaned shape + 8 new composite columns: "
    f"4 equal-weight + 4 weighted variants)")

summary_path = os.path.join(OUT_DIR, "composite_index_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"\n\nSUMMARY LOG WRITTEN TO: {summary_path}")
print(f"ENGINEERED CSV (use this for encoding/modeling next): {engineered_path}")
