"""
02_data_cleaning.py
====================
Full, documented, and AUDITABLE cleaning of PPD_dataset_v3.csv (raw, n=800).

Why this script exists
-----------------------
Running 01_dataset_overview.py on the RAW csv exposed three classes of
problems that must be fixed before any feature engineering / modeling,
otherwise the model will learn noise instead of signal:

  A) Category-label inconsistency (typos / case mismatches / stray Bangla
     text) that make Python treat one real-world category as 2-3 different
     categories.
  B) "Missing" values that are not actually missing at random -- they are
     STRUCTURAL: the question did not apply to that mother (e.g. she has no
     addiction, no pregnancy loss, no personal income, only one child).
     These must be filled with an explicit category, NOT imputed with
     median/mode, otherwise you inject false information.
  C) A small number of TRUE random missing values (Abuse, Husband's income,
     Education Level, etc.) which are fine to impute with the most frequent
     category, following the same approach the reference IEEE paper used.
  D) An inconsistent target label ("Normal" inside PHQ9 Result, which is not
     a standard PHQ-9 severity band) that is recomputed directly from the
     numeric score using the standard clinical cut-offs, so the label is
     guaranteed internally consistent.

Every transformation below prints a BEFORE/AFTER count so you (and your
thesis committee) can audit exactly what changed and why. Nothing is
silently dropped.

Output (written to ./outputs/):
  - PPD_dataset_cleaned_v2.csv   <- use this file for feature engineering / modeling
  - cleaning_log.txt             <- full audit trail of every fix applied
  - missing_after_cleaning.csv   <- should be all zeros except explicitly
                                    flagged "unknown" categories

Usage
-----
    python 02_data_cleaning.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.dirname(THIS_DIR)
CSV_PATH = os.path.join(DATASET_DIR, "PPD_dataset_v3.csv")
OUT_DIR = os.path.join(THIS_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "cleaning_log.txt")
CLEANED_PATH = os.path.join(OUT_DIR, "PPD_dataset_cleaned_v2.csv")
MISSING_AFTER_PATH = os.path.join(OUT_DIR, "missing_after_cleaning.csv")

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
# 1. Load raw data
# ---------------------------------------------------------------------------
section("1. LOAD RAW DATA")

NA_VALUES = ["None", "none", "NaN", "nan", "", "N/A", "n/a", "NA"]
df = pd.read_csv(CSV_PATH, na_values=NA_VALUES)
log(f"Loaded raw shape: {df.shape}")

if "sr" in df.columns:
    df = df.drop(columns=["sr"])
    log("Dropped 'sr' index column.")

# ---------------------------------------------------------------------------
# 2. Clean column names
# ---------------------------------------------------------------------------
section("2. CLEAN COLUMN NAMES")

df.columns = df.columns.str.strip().str.replace("’", "'", regex=False)
df = df.rename(columns={
    "Recieved Support": "Received Support",
    "rouble concentrating on things": "Trouble concentrating on things",
    "Birth compliancy": "Birth complications",
})

# strip whitespace inside every string cell
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

log(f"Columns after cleanup: {len(df.columns)}")

# ---------------------------------------------------------------------------
# 3. Fix category-label inconsistencies (typos / case / stray Bangla text)
# ---------------------------------------------------------------------------
section("3. FIX CATEGORY LABEL INCONSISTENCIES")

def fix_column(col, mapping):
    if col not in df.columns:
        log(f"  [skip] column not found: {col}")
        return
    before = df[col].nunique(dropna=True)
    df[col] = df[col].replace(mapping)
    after = df[col].nunique(dropna=True)
    log(f"  {col}: {before} -> {after} unique values "
        f"(mapping applied: {mapping})")


fix_column("Education Level", {
    "High school": "High School",
    "Primaryschool": "Primary School",
    "Primary school": "Primary School",
})
fix_column("Husband's education level", {
    "High school": "High School",
    "Primaryschool": "Primary School",
    "Primary school": "Primary School",
})
fix_column("Total children", {
    "More than two": "More than Two",
})
fix_column("History of pregnancy loss", {
    "Still-born delivery": "Still-born Delivery",
})
fix_column("Diseases during pregnancy", {
    "Chronic disease": "Chronic Disease",
    "Non-chronic disease": "Non-Chronic Disease",
    "Non-chronic Disease": "Non-Chronic Disease",
})
fix_column("Disease before pregnancy", {
    "Chronic disease": "Chronic Disease",
    "Non-chronic disease": "Non-Chronic Disease",
    "Non-chronic Disease": "Non-Chronic Disease",
    "1.2": np.nan,   # data-entry error -> treat as missing
})
for occ_col in ["Occupation before latest pregnancy",
                "Occupation After Your Latest Childbirth"]:
    fix_column(occ_col, {"House wife": "Housewife"})

# --- Likert-scale items used by EPDS (10) and PHQ-9 (9) --------------------
LIKERT_MAP = {
    "Several Days": "Several days",
    "several days": "Several days",
    "মাঝে মধ্যে (Several days)": "Several days",
}

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

log("\nFixing Likert-scale item columns (EPDS + PHQ-9), unifying "
    "'Several Days' / Bangla text -> 'Several days':")
for col in EPDS_ITEMS + PHQ9_ITEMS:
    fix_column(col, LIKERT_MAP)

# Confirm every Likert item now has exactly the 4 canonical categories
VALID_LIKERT = {"Not at all", "Several days", "Around half the days",
                 "Nearly every day"}
log("\nValidation - any Likert item column NOT in the 4 canonical labels?")
for col in EPDS_ITEMS + PHQ9_ITEMS:
    if col in df.columns:
        bad = set(df[col].dropna().unique()) - VALID_LIKERT
        if bad:
            log(f"  STILL INCONSISTENT -> {col}: unexpected values {bad}")
log("  (no output above other than this line = all clean)")

# ---------------------------------------------------------------------------
# 4. Structural missing values -> explicit categories (NOT imputation)
# ---------------------------------------------------------------------------
section("4. STRUCTURAL MISSING VALUES -> EXPLICIT 'NOT APPLICABLE' CATEGORIES")

log("These columns have high missing % because the question did not apply "
    "to that mother (e.g. no addiction, no disease, only one child) -- "
    "filling with median/mode would inject false information. We instead "
    "give them an explicit, honest category.")

STRUCTURAL_FILL = {
    "Addiction": "No Addiction",
    "History of pregnancy loss": "No Pregnancy Loss",
    "Disease before pregnancy": "No Disease",
    "Diseases during pregnancy": "No Disease",
    "Age of immediate older children": "Not Applicable (Only One Child)",
}

for col, fill_value in STRUCTURAL_FILL.items():
    if col not in df.columns:
        log(f"  [skip] column not found: {col}")
        continue
    n_missing = df[col].isnull().sum()
    df[col] = df[col].fillna(fill_value)
    log(f"  {col}: filled {n_missing} missing -> '{fill_value}'")

# --- Income columns: missing correlates with Housewife/Student (no personal
#     income), verify the assumption before filling -----------------------
log("\nChecking whether missing income correlates with occupation "
    "(sanity check before filling):")
for income_col, occ_col in [
    ("Monthly income before latest pregnancy",
     "Occupation before latest pregnancy"),
    ("Current monthly income", "Occupation After Your Latest Childbirth"),
]:
    if income_col in df.columns and occ_col in df.columns:
        missing_mask = df[income_col].isnull()
        log(f"\n  Occupation breakdown for missing '{income_col}':")
        log("  " + df.loc[missing_mask, occ_col].value_counts(dropna=False)
            .to_string().replace("\n", "\n  "))

for income_col in ["Monthly income before latest pregnancy",
                    "Current monthly income"]:
    if income_col in df.columns:
        n_missing = df[income_col].isnull().sum()
        df[income_col] = df[income_col].fillna("No Personal Income")
        log(f"\n  {income_col}: filled {n_missing} missing -> "
            f"'No Personal Income'")

# --- Ambiguous column: keep as an honest explicit bucket -------------------
if "Feeling for regular activities" in df.columns:
    n_missing = df["Feeling for regular activities"].isnull().sum()
    df["Feeling for regular activities"] = (
        df["Feeling for regular activities"].fillna("Not Reported")
    )
    log(f"\n  Feeling for regular activities: filled {n_missing} missing -> "
        f"'Not Reported' (meaning not clearly structural -- keep as its own "
        f"category rather than guessing mode)")

# --- Need for Support: 167 missing (20.88%) -- was MISSED in the first
#     version of this script (caught during review of the run output).
#     Diagnose whether it correlates with 'Received Support' (possible skip
#     logic: mother already gets High support -> "need" question skipped)
#     before deciding how to fill it. ---------------------------------------
if "Need for Support" in df.columns:
    n_missing = df["Need for Support"].isnull().sum()
    log(f"\n  Need for Support: {n_missing} missing -- diagnosing correlation "
        f"with 'Received Support' before filling:")
    if "Received Support" in df.columns:
        missing_mask = df["Need for Support"].isnull()
        log("  Received Support breakdown for missing 'Need for Support':")
        log("  " + df.loc[missing_mask, "Received Support"]
            .value_counts(dropna=False).to_string().replace("\n", "\n  "))
        log("  Received Support breakdown for the FULL dataset (for comparison):")
        log("  " + df["Received Support"]
            .value_counts(dropna=False).to_string().replace("\n", "\n  "))
    # No confirmed structural cause identified yet (unlike income/addiction),
    # so we do NOT guess a skip-logic label. Keep it as its own transparent
    # category rather than mode-imputing 167 rows (~21% of the data), which
    # would be too aggressive an assumption.
    df["Need for Support"] = df["Need for Support"].fillna("Not Reported")
    log(f"  Need for Support: filled {n_missing} missing -> 'Not Reported' "
        f"(kept as its own category; revisit if the crosstab above shows a "
        f"clear pattern, e.g. mostly 'High' Received Support)")

# ---------------------------------------------------------------------------
# 5. True random missing values (small %) -> mode imputation + missing flag
# ---------------------------------------------------------------------------
section("5. TRUE RANDOM MISSING VALUES -> MODE IMPUTATION (with flag column)")

RANDOM_MISSING_COLS = [
    "Abuse",
    "Husband's monthly income",
    "Husband's education level",
    "Education Level",
    "Trust and share feelings",
]

for col in RANDOM_MISSING_COLS:
    if col not in df.columns:
        log(f"  [skip] column not found: {col}")
        continue
    n_missing = df[col].isnull().sum()
    if n_missing == 0:
        continue
    flag_col = f"{col}_was_missing"
    df[flag_col] = df[col].isnull().astype(int)
    mode_val = df[col].mode(dropna=True).iloc[0]
    df[col] = df[col].fillna(mode_val)
    log(f"  {col}: filled {n_missing} missing -> mode '{mode_val}' "
        f"(flag column '{flag_col}' added, sum={df[flag_col].sum()})")

# ---------------------------------------------------------------------------
# 6. Recompute target labels from the numeric scores (fixes the "Normal"
#    category and guarantees EPDS Result / PHQ9 Result are internally
#    consistent with EPDS Score / PHQ9 Score)
# ---------------------------------------------------------------------------
section("6. RECOMPUTE TARGET CATEGORIES FROM SCORES (fixes inconsistent labels)")


def epds_band(score):
    if score <= 8:
        return "Low"
    elif score <= 12:
        return "Medium"
    else:
        return "High"


def phq9_band(score):
    if score <= 4:
        return "Minimal"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    elif score <= 19:
        return "Moderately Severe"
    else:
        return "Severe"


if "EPDS Score" in df.columns and "EPDS Result" in df.columns:
    recomputed = df["EPDS Score"].apply(epds_band)
    mismatch = (recomputed != df["EPDS Result"]).sum()
    log(f"EPDS Result: {mismatch} / {len(df)} rows disagreed with the "
        f"score-based band (0-8 Low, 9-12 Medium, 13-30 High). "
        f"Replacing 'EPDS Result' with the recomputed, guaranteed-"
        f"consistent version.")
    df["EPDS Result"] = recomputed
    log("New EPDS Result distribution:")
    log(df["EPDS Result"].value_counts().to_string())

if "PHQ9 Score" in df.columns and "PHQ9 Result" in df.columns:
    recomputed = df["PHQ9 Score"].apply(phq9_band)
    mismatch = (recomputed != df["PHQ9 Result"]).sum()
    log(f"\nPHQ9 Result: {mismatch} / {len(df)} rows disagreed with the "
        f"standard score-based band (0-4 Minimal ... 20+ Severe), "
        f"including all rows previously labeled the non-standard 'Normal'. "
        f"Replacing 'PHQ9 Result' with the recomputed, guaranteed-"
        f"consistent version.")
    df["PHQ9 Result"] = recomputed
    log("New PHQ9 Result distribution:")
    log(df["PHQ9 Result"].value_counts().to_string())

# Binary PPD target (used by the IEEE reference paper): EPDS >= 13 -> 1
if "EPDS Score" in df.columns:
    df["PPD_binary"] = (df["EPDS Score"] >= 13).astype(int)
    log("\nAdded binary target 'PPD_binary' (EPDS Score >= 13 -> 1):")
    log(df["PPD_binary"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 7. Final checks
# ---------------------------------------------------------------------------
section("7. FINAL CHECKS")

log(f"Duplicate rows: {df.duplicated().sum()}")
log(f"Final shape: {df.shape}")

missing_after = df.isnull().sum()
missing_after = missing_after[missing_after > 0]
if missing_after.empty:
    log("No missing values remain anywhere in the dataset.")
else:
    log("Columns that STILL have missing values (should be none / review!):")
    log(missing_after.to_string())
missing_after.to_csv(MISSING_AFTER_PATH)

# ---------------------------------------------------------------------------
# 8. Save cleaned dataset
# ---------------------------------------------------------------------------
section("8. SAVE CLEANED DATASET")

df.to_csv(CLEANED_PATH, index=False)
log(f"Saved cleaned dataset -> {CLEANED_PATH}")
log(f"Shape saved: {df.shape}")

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"\n\nFULL CLEANING LOG WRITTEN TO: {LOG_PATH}")
print(f"CLEANED CSV (use this for feature engineering next): {CLEANED_PATH}")
