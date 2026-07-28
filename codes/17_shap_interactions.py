"""
17_shap_interactions.py
==========================
Improvement idea: SHAP INTERACTION VALUES.

Every SHAP analysis so far (step 9, script 16) only reports MAIN effects --
how much a single feature, on its own, pushes a prediction up or down. It
never asked whether two features work TOGETHER in a way that's more than
the sum of their parts -- e.g. does low income only matter when support is
also low? Does Abuse history matter more at certain ages? Random Forest can
capture such interactions internally (that's what makes tree models
powerful), and SHAP interaction values are the tool that surfaces them
explicitly, splitting each prediction's explanation into pure main effects
plus pairwise interaction effects.

Because interaction values are computed for every FEATURE PAIR (an O(n^2)
computation), this script restricts the analysis to the top 20 most
important features per target (from step 9's saved
shap_feature_importance_<target>.csv) rather than all 112 -- a fresh, small
Random Forest is trained on just those 20 features specifically for this
interaction analysis (clearly a diagnostic/exploratory model, not a
replacement for the main pipeline's model).

Output (./outputs/):
  - shap_interaction_top_pairs_<target>.csv
  - shap_interaction_heatmap_<target>.png
  - shap_interaction_summary.txt

Usage
-----
    python 17_shap_interactions.py

NOTE: requires 09_shap_explainability.py to have been run already (this
script reuses its saved shap_feature_importance_<target>.csv files to know
which 20 features to focus on).
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

if not os.path.exists(MODEL_READY_PATH):
    raise SystemExit(f"{MODEL_READY_PATH} not found. Run 06_encoding.py first.")

df = pd.read_csv(MODEL_READY_PATH)
print(f"Loaded model-ready dataset: {df.shape}")

RANDOM_STATE = 42
TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
FOCUS_CLASS = {"PPD_binary": "1", "EPDS Result": "High", "PHQ9 Result": "Severe"}
TOP_N_FEATURES = 20

# Clinically-interesting pairs to specifically call out, if both features
# (or a close match) are present in a target's top-20 set. Matched by
# substring since exact one-hot column names vary (e.g. "Abuse_Yes").
INTERESTING_PAIR_HINTS = [
    ("Age", "Support"), ("Age", "Social_Support"),
    ("Age", "Angry"),
    ("Economic_Stability", "Abuse"), ("income", "Abuse"),
    ("Economic_Stability", "Maternal_MentalHealth"),
]

summary_lines = []

for target_col in TARGETS:
    print(f"\n{'=' * 90}\nTARGET: {target_col}\n{'=' * 90}")
    safe_name = target_col.replace(" ", "_")
    imp_path = os.path.join(OUT_DIR, f"shap_feature_importance_{safe_name}.csv")
    if not os.path.exists(imp_path):
        print(f"  [warn] {imp_path} not found -- run 09_shap_explainability.py "
              f"first. Skipping {target_col}.")
        continue
    imp_df = pd.read_csv(imp_path)
    top_features = imp_df.sort_values("mean_abs_shap", ascending=False) \
        .head(TOP_N_FEATURES)["feature"].tolist()
    print(f"  Using top {len(top_features)} features from step 9's ranking: "
          f"{top_features}")

    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    class_names = [str(c) for c in le.classes_]
    focus_label = FOCUS_CLASS[target_col]
    focus_idx = class_names.index(focus_label) if focus_label in class_names else len(class_names) - 1

    X = df[top_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print(f"  Diagnostic model (top-{TOP_N_FEATURES} features only) test "
          f"accuracy: {model.score(X_test, y_test):.3f}")

    explainer = shap.TreeExplainer(model)
    interaction_raw = explainer.shap_interaction_values(X_test)

    if isinstance(interaction_raw, list):
        interaction = np.asarray(interaction_raw[focus_idx])
    else:
        interaction_raw = np.asarray(interaction_raw)
        if interaction_raw.ndim == 4:
            # (n_samples, n_features, n_features, n_classes)
            interaction = interaction_raw[:, :, :, focus_idx]
        else:
            interaction = interaction_raw  # (n_samples, n_features, n_features)

    mean_abs_interaction = np.abs(interaction).mean(axis=0)  # (n_feat, n_feat)
    n_feat = len(top_features)

    # Extract off-diagonal pairs (self-interaction on the diagonal is really
    # the main effect, not a true pairwise interaction -- excluded here).
    pair_rows = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            pair_rows.append({
                "feature_1": top_features[i], "feature_2": top_features[j],
                "mean_abs_interaction": mean_abs_interaction[i, j] * 2,
                # x2 because SHAP splits each pair's interaction evenly
                # across the (i,j) and (j,i) cells; summing both halves
                # gives the pair's total interaction magnitude.
            })
    pairs_df = pd.DataFrame(pair_rows).sort_values(
        "mean_abs_interaction", ascending=False)

    print(f"\n  Top 15 feature-pair interactions:")
    print(pairs_df.head(15).to_string(index=False))

    # Check the user-suggested clinically interesting pairs specifically
    print("\n  Checking clinically-motivated pairs of interest:")
    for hint_a, hint_b in INTERESTING_PAIR_HINTS:
        match = pairs_df[
            pairs_df["feature_1"].str.contains(hint_a, case=False, na=False)
            & pairs_df["feature_2"].str.contains(hint_b, case=False, na=False)
            | pairs_df["feature_1"].str.contains(hint_b, case=False, na=False)
            & pairs_df["feature_2"].str.contains(hint_a, case=False, na=False)]
        if len(match) > 0:
            row = match.iloc[0]
            rank = int(pairs_df.index.get_loc(row.name)) + 1
            print(f"    '{hint_a}' x '{hint_b}': found as "
                  f"'{row['feature_1']}' x '{row['feature_2']}', "
                  f"interaction={row['mean_abs_interaction']:.4f} "
                  f"(rank {rank}/{len(pairs_df)})")
        else:
            print(f"    '{hint_a}' x '{hint_b}': neither feature in this "
                  f"target's top-{TOP_N_FEATURES} set -- not checked.")

    out_path = os.path.join(OUT_DIR, f"shap_interaction_top_pairs_{safe_name}.csv")
    pairs_df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    try:
        plt.figure(figsize=(10, 9))
        plt.imshow(mean_abs_interaction, cmap="viridis")
        plt.colorbar(label="Mean |SHAP interaction value|")
        plt.xticks(range(n_feat), top_features, rotation=90, fontsize=7)
        plt.yticks(range(n_feat), top_features, fontsize=7)
        plt.title(f"{target_col} -- SHAP Interaction Heatmap (top {n_feat} features)")
        plt.tight_layout()
        heatmap_path = os.path.join(OUT_DIR, f"shap_interaction_heatmap_{safe_name}.png")
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"  Saved -> {heatmap_path}")
    except Exception as e:
        print(f"  [plot failed] interaction heatmap: {e}")

    top_pair = pairs_df.iloc[0]
    summary_lines.append(
        f"{target_col}: strongest interaction = "
        f"'{top_pair['feature_1']}' x '{top_pair['feature_2']}' "
        f"(mean |interaction| = {top_pair['mean_abs_interaction']:.4f}).")

summary_path = os.path.join(OUT_DIR, "shap_interaction_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"\n\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
print("\n".join(summary_lines))
print(f"Saved -> {summary_path}")
print("\nDone. Look at each target's top pairs list and heatmap -- do any "
      "pairs make clinical sense (e.g. a support-related feature interacting "
      "with an economic or mental-health feature)? Report the strongest, "
      "most clinically-plausible pairs in the discussion section.")
