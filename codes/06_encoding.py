"""
06_encoding.py
==============
Turns the engineered dataset (cleaned raw features + 8 composite index
columns) into a fully numeric, model-ready table.

Key design decisions
---------------------
1. LEAKAGE REMOVAL: the 10 EPDS items, 9 PHQ-9 items, and the 2 raw scores
   (EPDS Score, PHQ9 Score) are dropped entirely -- they were only ever
   needed to construct the targets in earlier scripts and must never be
   used as model input.

2. TARGETS ARE KEPT, NOT ENCODED: `EPDS Result`, `PHQ9 Result`, and
   `PPD_binary` are preserved as-is at the end of the file. The modeling
   script (07) will pick ONE of these as `y` and drop the other two from
   `X` for that experiment (never train on more than one target at a time).

3. RAW FEATURES ARE KEPT ALONGSIDE COMPOSITES (not replaced): both the
   original raw columns (e.g. `Abuse`, `Education Level`) AND the 8
   composite index columns built from them are present in the output.
   This is intentional -- the ablation study (script 08) needs a
   "Raw only" version and a "Raw + Composite" version of the SAME feature
   set, and that comparison only works if both exist side by side here.

4. ONE-HOT ENCODING with drop_first=True for every categorical (object
   dtype) column -- standard practice to avoid the dummy-variable trap.
   Numeric columns (Age, Number of the latest pregnancy, the 5
   `_was_missing` flags, and all 8 composite indices) are left untouched;
   feature SCALING (needed for Logistic Regression/SVM) is deliberately
   deferred to the modeling script's sklearn Pipeline, so scaling
   parameters are fit on the training fold only (no leakage between
   train/test splits later).

Output (./outputs/):
  - PPD_model_ready.csv     -- fully numeric features + 3 target columns
  - feature_columns.txt     -- list of every feature column going into X
  - encoding_map.csv        -- which one-hot columns came from which
                               original raw column (for later SHAP grouping)

Usage
-----
    python 06_encoding.py
"""

import os
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
ENGINEERED_PATH = os.path.join(OUT_DIR, "PPD_dataset_with_composite_features.csv")

if not os.path.exists(ENGINEERED_PATH):
    raise SystemExit(f"{ENGINEERED_PATH} not found. Run 04_feature_engineering.py first.")

df = pd.read_csv(ENGINEERED_PATH)
print(f"Loaded engineered dataset: {df.shape}")

# ---------------------------------------------------------------------------
# 1. Drop leakage columns; keep targets separately
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
TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]

DROP_COLS = [c for c in (EPDS_ITEMS + PHQ9_ITEMS + RAW_SCORES) if c in df.columns]
print(f"\nDropping {len(DROP_COLS)} leakage columns (EPDS/PHQ-9 items + raw scores).")
df = df.drop(columns=DROP_COLS)

targets_df = df[TARGETS].copy()
features_df = df.drop(columns=TARGETS)

print(f"Features before encoding: {features_df.shape[1]} columns")
print(f"Targets kept as-is: {TARGETS}")

# ---------------------------------------------------------------------------
# 2. One-hot encode categorical columns
# ---------------------------------------------------------------------------
categorical_cols = features_df.select_dtypes(include="object").columns.tolist()
numeric_cols = [c for c in features_df.columns if c not in categorical_cols]

print(f"\nCategorical columns to one-hot encode: {len(categorical_cols)}")
print(f"Numeric columns left as-is (incl. composite indices + missing "
      f"flags): {len(numeric_cols)}")
print("  " + ", ".join(numeric_cols))

# dtype=int forces plain 0/1 integer columns. Without this, recent pandas
# versions (2.x/3.x) default one-hot columns to BOOLEAN dtype, which trips
# up some downstream tools (e.g. StandardScaler / SHAP expecting numeric
# int/float) even though it's logically equivalent to 0/1.
encoded = pd.get_dummies(features_df, columns=categorical_cols,
                          drop_first=True, dtype=int)

print(f"\nShape after one-hot encoding: {encoded.shape}")

# Build a simple mapping of which dummy columns came from which raw column,
# useful later for grouping SHAP values back to the original question.
encoding_map_rows = []
for col in categorical_cols:
    dummies = [c for c in encoded.columns if c.startswith(col + "_")]
    for d in dummies:
        encoding_map_rows.append({"original_column": col, "encoded_column": d})
encoding_map = pd.DataFrame(encoding_map_rows)
encoding_map_path = os.path.join(OUT_DIR, "encoding_map.csv")
encoding_map.to_csv(encoding_map_path, index=False)
print(f"Saved -> {encoding_map_path}")

# ---------------------------------------------------------------------------
# 3. Recombine with targets and save
# ---------------------------------------------------------------------------
final_df = pd.concat([encoded.reset_index(drop=True),
                       targets_df.reset_index(drop=True)], axis=1)

print(f"\nFinal model-ready shape: {final_df.shape} "
      f"({encoded.shape[1]} features + {len(TARGETS)} target columns)")

model_ready_path = os.path.join(OUT_DIR, "PPD_model_ready.csv")
final_df.to_csv(model_ready_path, index=False)
print(f"Saved -> {model_ready_path}")

feature_cols_path = os.path.join(OUT_DIR, "feature_columns.txt")
with open(feature_cols_path, "w", encoding="utf-8") as f:
    f.write("\n".join(encoded.columns.tolist()))
print(f"Saved -> {feature_cols_path}")

# ---------------------------------------------------------------------------
# 4. Sanity checks
# ---------------------------------------------------------------------------
print("\n--- Sanity checks ---")
print("Any non-numeric columns left in features?",
      encoded.select_dtypes(exclude="number").columns.tolist() or "None -- good.")
print("Any missing values left in features?",
      int(encoded.isnull().sum().sum()), "(should be 0)")
print("\nTarget distributions in the final file:")
for t in TARGETS:
    print(f"\n{t}:")
    print(final_df[t].value_counts().to_string())

print(f"\nDone. {model_ready_path} is ready for 07_modeling.py.")
