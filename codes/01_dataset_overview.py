"""
01_dataset_overview.py
=======================
Purpose
-------
Full, one-shot overview of the "Data for Postpartum Depression Prediction in
Bangladesh" dataset (PPD_dataset_v3.csv, n=800). Run this BEFORE any feature
engineering / modeling so you (and anyone reading the output) know exactly:

  1. What is in every single column (dtype, unique values, missing %).
  2. Which columns are TARGET CANDIDATES (EPDS Result / EPDS Score /
     PHQ9 Result / PHQ9 Score).
  3. Which columns are LEAKAGE columns (EPDS's 10 items, PHQ-9's 9 items,
     and the two score columns) that must NEVER be used as model input
     whichever target you pick -- because they are literally what the
     target is computed from.
  4. Which columns are SAFE INPUT FEATURES (demographic, family, economic,
     pregnancy, neonatal, PHQ-2 history) usable for a leakage-free
     early-detection model.
  5. Basic descriptive stats, missingness, class balance for both targets.

Outputs (written into ./codes/outputs/):
  - dataset_overview_report.txt   -> full human-readable report
  - column_catalogue.csv          -> one row per column with dtype, role,
                                     n_unique, missing_count, missing_pct,
                                     sample values
  - missing_values.csv            -> only columns with missing data
  - epds_class_distribution.png
  - phq9_class_distribution.png
  - missingness_bar.png

Usage
-----
    pip install pandas numpy matplotlib --break-system-packages   # if needed
    python 01_dataset_overview.py

The script expects PPD_dataset_v3.csv to be one folder above this script
(i.e. in the "dataset with exit paper" folder). Adjust CSV_PATH below if you
move things around.
"""

import os
import re
import sys
import json

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.dirname(THIS_DIR)          # parent folder
CSV_PATH = os.path.join(DATASET_DIR, "PPD_dataset_v3.csv")
OUT_DIR = os.path.join(THIS_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(OUT_DIR, "dataset_overview_report.txt")
CATALOGUE_PATH = os.path.join(OUT_DIR, "column_catalogue.csv")
MISSING_PATH = os.path.join(OUT_DIR, "missing_values.csv")

report_lines = []


def log(*args):
    """Print to console AND collect into the report file."""
    line = " ".join(str(a) for a in args)
    print(line)
    report_lines.append(line)


def section(title):
    log("")
    log("=" * 90)
    log(title)
    log("=" * 90)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
section("1. LOADING DATASET")

if not os.path.exists(CSV_PATH):
    sys.exit(f"CSV not found at expected path: {CSV_PATH}\n"
             f"Edit CSV_PATH at the top of this script if your file lives "
             f"somewhere else.")

# The raw CSV stores missing values as the literal string "None" (not blank).
NA_VALUES = ["None", "none", "NaN", "nan", "", "N/A", "n/a", "NA"]

df = pd.read_csv(CSV_PATH, na_values=NA_VALUES)
log(f"Loaded: {CSV_PATH}")
log(f"Raw shape (rows, cols): {df.shape}")

# Drop the pure row-index column if present
if "sr" in df.columns:
    df = df.drop(columns=["sr"])
    log("Dropped index column 'sr'.")

# ---------------------------------------------------------------------------
# 2. Clean column names (mirrors the cleaning already started in the notebook)
# ---------------------------------------------------------------------------
section("2. CLEANING COLUMN NAMES")

df.columns = (
    df.columns.str.strip()
    .str.replace("’", "'", regex=False)  # curly apostrophe -> straight
)

RENAME_MAP = {
    "Recieved Support": "Received Support",
    "rouble concentrating on things": "Trouble concentrating on things",
    "Birth compliancy": "Birth complications",
}
df = df.rename(columns=RENAME_MAP)
log(f"Cleaned column count: {len(df.columns)}")

# Strip whitespace inside categorical (object) cell values too
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

log("Final shape after cleaning:", df.shape)

# ---------------------------------------------------------------------------
# 3. Define column roles
# ---------------------------------------------------------------------------
section("3. COLUMN ROLE DEFINITIONS (for leakage-free modeling)")

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

TARGET_CANDIDATES = ["EPDS Score", "EPDS Result", "PHQ9 Score", "PHQ9 Result"]

# PHQ-2 (before/during pregnancy) is a *different* time point than the
# postpartum EPDS/PHQ-9 screening -> it is prior psychiatric history, a
# legitimate predictor, NOT leakage.
PHQ2_HISTORY = ["Depression before pregnancy (PHQ2)",
                "Depression during pregnancy (PHQ2)"]

# Sanity check: make sure every name above actually exists in the dataframe
missing_defined_cols = [c for c in EPDS_ITEMS + PHQ9_ITEMS + TARGET_CANDIDATES
                         + PHQ2_HISTORY if c not in df.columns]
if missing_defined_cols:
    log("WARNING - these expected columns were not found (check spelling / "
        "the RENAME_MAP above):")
    for c in missing_defined_cols:
        log("   -", c)

LEAKAGE_COLS = [c for c in (EPDS_ITEMS + PHQ9_ITEMS + TARGET_CANDIDATES)
                if c in df.columns]

SAFE_INPUT_FEATURES = [c for c in df.columns if c not in LEAKAGE_COLS]

log(f"Target candidates ({len(TARGET_CANDIDATES)}):")
for c in TARGET_CANDIDATES:
    log("   -", c)

log(f"\nLeakage columns to EXCLUDE from input, {len(LEAKAGE_COLS)} total "
    f"(EPDS 10 items + PHQ-9 9 items + 4 score/result columns):")
for c in LEAKAGE_COLS:
    log("   -", c)

log(f"\nSAFE input features for a leakage-free model, {len(SAFE_INPUT_FEATURES)} total:")
for c in SAFE_INPUT_FEATURES:
    log("   -", c)

log(f"\nNote: PHQ-2 columns {PHQ2_HISTORY} are KEPT as safe input features "
    f"(prior psychiatric history, not part of the postpartum EPDS/PHQ-9 "
    f"scoring), consistent with the two reference papers in this folder.")

# ---------------------------------------------------------------------------
# 4. Dtypes & missingness
# ---------------------------------------------------------------------------
section("4. DTYPES & MISSING VALUES")

dtype_series = df.dtypes.astype(str)
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df) * 100).round(2)

log(df.dtypes.to_string())

missing_table = pd.DataFrame({
    "missing_count": missing_count,
    "missing_pct": missing_pct,
}).sort_values("missing_pct", ascending=False)
missing_table_nonzero = missing_table[missing_table["missing_count"] > 0]

log("\nColumns with missing values (sorted by % missing):")
log(missing_table_nonzero.to_string())
missing_table_nonzero.to_csv(MISSING_PATH)
log(f"\nSaved -> {MISSING_PATH}")

# ---------------------------------------------------------------------------
# 5. Column catalogue (the single most useful artifact)
# ---------------------------------------------------------------------------
section("5. BUILDING FULL COLUMN CATALOGUE")

rows = []
for col in df.columns:
    if col in TARGET_CANDIDATES:
        role = "TARGET_CANDIDATE"
    elif col in LEAKAGE_COLS:
        role = "LEAKAGE_ITEM (exclude from input)"
    elif col in PHQ2_HISTORY:
        role = "SAFE_INPUT (psychiatric history)"
    else:
        role = "SAFE_INPUT"

    n_unique = df[col].nunique(dropna=True)
    sample_vals = df[col].dropna().unique()[:6]
    sample_vals_str = " | ".join(str(v) for v in sample_vals)

    rows.append({
        "column": col,
        "dtype": str(df[col].dtype),
        "role": role,
        "n_unique": n_unique,
        "missing_count": int(missing_count[col]),
        "missing_pct": float(missing_pct[col]),
        "sample_values": sample_vals_str,
    })

catalogue = pd.DataFrame(rows)
catalogue.to_csv(CATALOGUE_PATH, index=False)
log(f"Saved full column catalogue -> {CATALOGUE_PATH}")
log("\nPreview:")
log(catalogue.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Numeric summary
# ---------------------------------------------------------------------------
section("6. NUMERIC FEATURE SUMMARY")

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
log("Numeric columns:", numeric_cols)
if numeric_cols:
    log(df[numeric_cols].describe().T.to_string())

# ---------------------------------------------------------------------------
# 7. Target distributions
# ---------------------------------------------------------------------------
section("7. TARGET DISTRIBUTIONS")

if "EPDS Result" in df.columns:
    log("\nEPDS Result value counts:")
    log(df["EPDS Result"].value_counts(dropna=False).to_string())
    log("\nEPDS Result percentages:")
    log((df["EPDS Result"].value_counts(normalize=True, dropna=False) * 100)
        .round(2).to_string())

if "PHQ9 Result" in df.columns:
    log("\nPHQ9 Result value counts:")
    log(df["PHQ9 Result"].value_counts(dropna=False).to_string())
    log("\nPHQ9 Result percentages:")
    log((df["PHQ9 Result"].value_counts(normalize=True, dropna=False) * 100)
        .round(2).to_string())

# Binary PPD target used in the IEEE reference paper: EPDS >= 13 -> 1
if "EPDS Score" in df.columns:
    df["_PPD_binary_preview"] = (df["EPDS Score"] >= 13).astype(int)
    log("\nBinary PPD (EPDS>=13) preview distribution:")
    log(df["_PPD_binary_preview"].value_counts(dropna=False).to_string())
    log((df["_PPD_binary_preview"].value_counts(normalize=True) * 100)
        .round(2).to_string())
    df = df.drop(columns=["_PPD_binary_preview"])

# ---------------------------------------------------------------------------
# 8. Categorical value frequencies (all safe input features)
# ---------------------------------------------------------------------------
section("8. CATEGORICAL FREQUENCIES (SAFE INPUT FEATURES ONLY)")

cat_cols = [c for c in SAFE_INPUT_FEATURES
            if df[c].dtype == "object"]

for col in cat_cols:
    log(f"\n--- {col} ---")
    vc = df[col].value_counts(dropna=False)
    pct = df[col].value_counts(dropna=False, normalize=True) * 100
    tab = pd.DataFrame({"count": vc, "pct": pct.round(2)})
    log(tab.to_string())

# ---------------------------------------------------------------------------
# 9. Duplicate check
# ---------------------------------------------------------------------------
section("9. DUPLICATE ROWS")
log("Duplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------------------------
# 10. Composite-feature building blocks (for the novelty step, next script)
# ---------------------------------------------------------------------------
section("10. RAW COLUMNS EARMARKED FOR COMPOSITE INDEX ENGINEERING (next step)")

composite_plan = {
    "Economic Stability Index (ESI)": [
        "Current monthly income", "Husband's monthly income",
        "Occupation After Your Latest Childbirth", "Education Level",
    ],
    "Social/Family Support Index (SSI)": [
        "Family type", "Relationship with the in-laws",
        "Relationship with husband", "Number of household members",
        "Received Support", "Need for Support",
    ],
    "Maternal Mental-Health Risk Index (MHRI)": [
        "Depression before pregnancy (PHQ2)",
        "Depression during pregnancy (PHQ2)",
        "Disease before pregnancy", "Abuse", "History of pregnancy loss",
    ],
    "Neonatal/Delivery Stress Index (NSI)": [
        "Mode of delivery", "Birth complications", "Newborn illness",
        "Worry about newborn",
    ],
}
for idx_name, cols in composite_plan.items():
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    log(f"\n{idx_name}:")
    log("   uses:", present)
    if missing:
        log("   NOT FOUND (check spelling):", missing)

with open(os.path.join(OUT_DIR, "composite_feature_plan.json"), "w") as f:
    json.dump(composite_plan, f, indent=2)
log(f"\nSaved composite feature plan -> "
    f"{os.path.join(OUT_DIR, 'composite_feature_plan.json')}")

# ---------------------------------------------------------------------------
# 11. Plots
# ---------------------------------------------------------------------------
section("11. PLOTS")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "EPDS Result" in df.columns:
        plt.figure(figsize=(5, 4))
        df["EPDS Result"].value_counts().plot(kind="bar", color="#4C72B0")
        plt.title("EPDS Result Class Distribution (n=800)")
        plt.ylabel("Count")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "epds_class_distribution.png")
        plt.savefig(p, dpi=150)
        plt.close()
        log(f"Saved -> {p}")

    if "PHQ9 Result" in df.columns:
        plt.figure(figsize=(6, 4))
        df["PHQ9 Result"].value_counts().plot(kind="bar", color="#DD8452")
        plt.title("PHQ-9 Result Class Distribution (n=800)")
        plt.ylabel("Count")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "phq9_class_distribution.png")
        plt.savefig(p, dpi=150)
        plt.close()
        log(f"Saved -> {p}")

    if not missing_table_nonzero.empty:
        plt.figure(figsize=(8, max(4, 0.3 * len(missing_table_nonzero))))
        missing_table_nonzero["missing_pct"].sort_values().plot(kind="barh",
                                                                 color="#55A868")
        plt.title("Missing Value % by Column")
        plt.xlabel("% missing")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "missingness_bar.png")
        plt.savefig(p, dpi=150)
        plt.close()
        log(f"Saved -> {p}")

except ImportError:
    log("matplotlib not installed - skipping plots. "
        "Run: pip install matplotlib --break-system-packages")

# ---------------------------------------------------------------------------
# 12. Save cleaned dataframe for downstream scripts
# ---------------------------------------------------------------------------
section("12. SAVING CLEANED DATASET")

cleaned_path = os.path.join(OUT_DIR, "PPD_dataset_cleaned.csv")
df.to_csv(cleaned_path, index=False)
log(f"Saved cleaned dataset (col names fixed, 'None' -> NaN) -> {cleaned_path}")

# ---------------------------------------------------------------------------
# Write full report to disk
# ---------------------------------------------------------------------------
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n\nFULL REPORT WRITTEN TO: {REPORT_PATH}")
print(f"COLUMN CATALOGUE (open this in Excel first): {CATALOGUE_PATH}")
