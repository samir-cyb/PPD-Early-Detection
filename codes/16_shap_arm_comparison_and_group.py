"""
16_shap_arm_comparison_and_group.py
======================================
Two more explainability angles, both extending 09_shap_explainability.py
rather than replacing it:

PART A -- SHAP WITHOUT composites (Arm A) vs WITH composites (Arm B).
Step 9 only ever ran SHAP on Arm B ("Raw + Composite"). This script also
runs SHAP on Arm A ("Raw only", no composite indices) on the IDENTICAL
train/test split, so you can show, side by side: "without the composite
indices, SHAP's top features are individual raw answers (Age, a single
income bracket, a single support level, ...); with the composite indices
added, the composite indices themselves rise to the very top." This is a
direct, visual demonstration that the composite indices are not just
statistically significant in isolation (already shown in step 4/13) but
are the model's OWN preferred way of summarizing that information once
given the option.

PART B -- Group-level SHAP importance.
15_age_features_and_group_importance.py grouped one-hot dummies back to
their original question using PERMUTATION importance. This script does the
analogous thing for SHAP: since SHAP values are additive by construction
(they sum to the model's output), a group's combined pull on any single
prediction is simply the SUM of its dummies' SHAP values for that row --
so grouping is a direct sum, not an approximation. This is arguably the
more principled version of script 15's grouping idea, using Arm B's
already-computed SHAP values (reloaded from step 9's saved .npz files, not
recomputed) for all 3 targets.

Output (./outputs/):
  - shap_arm_comparison_<target>.csv     -- Arm A vs Arm B top features
  - shap_group_importance_<target>.csv   -- grouped SHAP importance
  - shap_arm_and_group_summary.txt

Usage
-----
    python 16_shap_arm_comparison_and_group.py

NOTE: requires 09_shap_explainability.py to have already been run (this
script reloads its saved shap_values_<target>.npz files for the Arm B side
instead of recomputing them).
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

try:
    import shap
except ImportError:
    raise SystemExit("shap is not installed. Run: pip install shap --break-system-packages")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")
ENCODING_MAP_PATH = os.path.join(OUT_DIR, "encoding_map.csv")

if not os.path.exists(MODEL_READY_PATH):
    raise SystemExit(f"{MODEL_READY_PATH} not found. Run 06_encoding.py first.")

df = pd.read_csv(MODEL_READY_PATH)
print(f"Loaded model-ready dataset: {df.shape}")

RANDOM_STATE = 42
TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
FOCUS_CLASS = {"PPD_binary": "1", "EPDS Result": "High", "PHQ9 Result": "Severe"}
TOP_N = 15

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
ARM_B_COLS = RAW_ONLY_COLS + WEIGHTED_COMPOSITES

summary_lines = []

# ===========================================================================
# PART A: SHAP on Arm A (no composites) vs Arm B (reloaded from step 9)
# ===========================================================================
print(f"\n{'#' * 90}\nPART A: SHAP without composites (Arm A) vs with (Arm B)\n{'#' * 90}")

for target_col in TARGETS:
    print(f"\n--- TARGET: {target_col} ---")
    safe_name = target_col.replace(" ", "_")
    focus_label = FOCUS_CLASS[target_col]

    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    class_names = [str(c) for c in le.classes_]
    focus_idx = class_names.index(focus_label) if focus_label in class_names else len(class_names) - 1

    # --- Arm A: raw only, fresh SHAP run ---
    X_a = df[RAW_ONLY_COLS]
    X_train_a, X_test_a, y_train, y_test = train_test_split(
        X_a, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model_a = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=RANDOM_STATE)
    model_a.fit(X_train_a, y_train)
    explainer_a = shap.TreeExplainer(model_a)
    shap_raw_a = explainer_a.shap_values(X_test_a)
    if isinstance(shap_raw_a, list):
        shap_a = np.asarray(shap_raw_a[focus_idx])
    else:
        shap_raw_a = np.asarray(shap_raw_a)
        shap_a = shap_raw_a[:, :, focus_idx] if shap_raw_a.ndim == 3 else shap_raw_a
    importance_a = pd.Series(np.abs(shap_a).mean(axis=0), index=RAW_ONLY_COLS) \
        .sort_values(ascending=False)

    # --- Arm B: reload step 9's saved SHAP values (not recomputed) ---
    npz_path = os.path.join(OUT_DIR, f"shap_values_{safe_name}.npz")
    if not os.path.exists(npz_path):
        print(f"  [warn] {npz_path} not found -- run 09_shap_explainability.py "
              f"first. Skipping Arm B side for {target_col}.")
        continue
    saved = np.load(npz_path, allow_pickle=True)
    shap_b = saved["shap_values"]
    feature_names_b = list(saved["feature_names"])
    importance_b = pd.Series(np.abs(shap_b).mean(axis=0), index=feature_names_b) \
        .sort_values(ascending=False)

    print(f"  Top {TOP_N} WITHOUT composites (Arm A):")
    print("  " + importance_a.head(TOP_N).to_string().replace("\n", "\n  "))
    print(f"\n  Top {TOP_N} WITH composites (Arm B, from step 9):")
    print("  " + importance_b.head(TOP_N).to_string().replace("\n", "\n  "))

    n_composite_in_top10_b = sum(1 for f in importance_b.head(10).index
                                  if f in WEIGHTED_COMPOSITES)
    print(f"\n  -> Without composites, the top features are individual raw "
          f"answers. With composites added, {n_composite_in_top10_b}/4 "
          f"composite indices take over the top-10 (see step 9's findings).")

    comparison_df = pd.DataFrame({
        "rank": range(1, TOP_N + 1),
        "without_composites (Arm A)": importance_a.head(TOP_N).index,
        "without_composites_score": importance_a.head(TOP_N).values,
        "with_composites (Arm B)": importance_b.head(TOP_N).index,
        "with_composites_score": importance_b.head(TOP_N).values,
    })
    out_path = os.path.join(OUT_DIR, f"shap_arm_comparison_{safe_name}.csv")
    comparison_df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")
    summary_lines.append(
        f"{target_col}: without composites, top feature = "
        f"'{importance_a.index[0]}'; with composites, top feature = "
        f"'{importance_b.index[0]}' ({n_composite_in_top10_b}/4 composites "
        f"in the with-composite top 10).")

    # =======================================================================
    # PART B: Group-level SHAP importance (Arm B, additive grouping)
    # =======================================================================
    if os.path.exists(ENCODING_MAP_PATH):
        encoding_map = pd.read_csv(ENCODING_MAP_PATH)
    else:
        encoding_map = None

    def build_groups(feature_cols, encoding_map):
        groups = {}
        grouped_cols = set()
        if encoding_map is not None:
            for orig_col, sub in encoding_map.groupby("original_column"):
                cols = [c for c in sub["encoded_column"].tolist() if c in feature_cols]
                if cols:
                    groups[orig_col] = cols
                    grouped_cols.update(cols)
        for col in feature_cols:
            if col not in grouped_cols:
                groups[col] = [col]
        return groups

    groups = build_groups(feature_names_b, encoding_map)
    shap_b_df = pd.DataFrame(shap_b, columns=feature_names_b)
    group_rows = []
    for group_name, cols in groups.items():
        grouped_shap_per_row = shap_b_df[cols].sum(axis=1)  # SHAP is additive
        group_rows.append({
            "group": group_name, "n_columns_in_group": len(cols),
            "mean_abs_grouped_shap": grouped_shap_per_row.abs().mean(),
        })
    group_shap_df = pd.DataFrame(group_rows).sort_values(
        "mean_abs_grouped_shap", ascending=False)
    print(f"\n  Top 15 groups by mean |grouped SHAP| (additive, Arm B):")
    print("  " + group_shap_df.head(15).to_string(index=False).replace("\n", "\n  "))

    group_out_path = os.path.join(OUT_DIR, f"shap_group_importance_{safe_name}.csv")
    group_shap_df.to_csv(group_out_path, index=False)
    print(f"  Saved -> {group_out_path}")

summary_path = os.path.join(OUT_DIR, "shap_arm_and_group_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"\n\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
print("\n".join(summary_lines))
print(f"Saved -> {summary_path}")
print("\nDone. Part A gives you the 'before/after' SHAP picture for the "
      "composite-index contribution story. Part B gives a SHAP-based "
      "(additive, more principled) version of script 15's grouped "
      "importance -- compare the two group rankings for consistency.")
