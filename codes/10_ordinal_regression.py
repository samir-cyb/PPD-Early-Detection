"""
10_ordinal_regression.py
==========================
Improvement idea #1 (highest priority from the post-mortem discussion):
EPDS Result and PHQ9 Result are ORDINAL targets (Low < Medium < High;
Minimal < Mild < Moderate < Moderately Severe < Severe), but every prior
script (07, 07b, 08, 09) treated them as plain, unordered multi-class
targets. Two consequences of that choice are addressed here:

1. UNFAIR METRIC: F1-macro/accuracy penalize "predicted Low, actual Severe"
   exactly the same as "predicted Moderate, actual Moderately Severe" -- a
   one-band miss and a four-band miss count equally. Quadratic Weighted
   Kappa (QWK) is the standard ordinal-aware metric that penalizes distant
   misses more than adjacent ones, and is added here for the first time.

2. UNTRIED APPROACH: instead of classifying the category directly, this
   script trains REGRESSION models to predict the underlying continuous
   EPDS Score / PHQ9 Score from the same leakage-free Arm B feature set,
   then applies the exact same clinical cut-offs used to build the
   official labels (see 02_data_cleaning.py's epds_band()/phq9_band()) to
   turn the predicted score back into a category. Regression's loss
   function (squared error) naturally respects order -- a prediction of
   11 when the truth is 13 is a small error, a prediction of 2 is a large
   one -- which a plain classifier's loss does not capture at all.

Both the existing "direct classifier" approach and the new "regression +
threshold" approach are evaluated on the IDENTICAL train/test split and
IDENTICAL feature set (Arm B "Raw + Composite", same as steps 8 and 9), so
the comparison is fair.

IMPORTANT: this script starts from PPD_dataset_with_composite_features.csv
(the output of 04_feature_engineering.py), NOT PPD_model_ready.csv --
because 06_encoding.py deliberately drops "EPDS Score"/"PHQ9 Score" as
leakage columns (they're needed to build the target, so they can't be a
model INPUT), but this script needs them as the regression TARGET, so the
encoding step is redone here from the pre-drop file. The dropped-as-input,
used-as-target logic still holds: EPDS Score/PHQ9 Score/EPDS items/PHQ9
items are never in X, only ever used to construct y or to validate labels.

Output (./outputs/):
  - ordinal_regression_results_EPDS_Result.csv
  - ordinal_regression_results_PHQ9_Result.csv
  - ordinal_comparison_summary.txt

Usage
-----
    python 10_ordinal_regression.py
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              accuracy_score, f1_score, cohen_kappa_score)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
ENGINEERED_PATH = os.path.join(OUT_DIR, "PPD_dataset_with_composite_features.csv")

if not os.path.exists(ENGINEERED_PATH):
    raise SystemExit(f"{ENGINEERED_PATH} not found. Run 04_feature_engineering.py first.")

df = pd.read_csv(ENGINEERED_PATH)
print(f"Loaded engineered dataset: {df.shape}")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Same leakage bookkeeping as 06_encoding.py, but keep the raw SCORES
#    around (as regression targets), not just the category labels.
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
CATEGORY_TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]

# Drop the 19 item columns entirely -- never used as input OR target here.
df = df.drop(columns=[c for c in EPDS_ITEMS + PHQ9_ITEMS if c in df.columns])

# ---------------------------------------------------------------------------
# 2. Rebuild the Arm B "Raw + Composite" feature set exactly as in
#    06_encoding.py / 08_ablation_study.py, EXCLUDING EPDS Score/PHQ9 Score/
#    the 3 category targets from X (they are never inputs).
# ---------------------------------------------------------------------------
NON_FEATURE_COLS = ["EPDS Score", "PHQ9 Score"] + CATEGORY_TARGETS
features_df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])

categorical_cols = features_df.select_dtypes(include="object").columns.tolist()
encoded = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True,
                          dtype=int)

EQUAL_WEIGHT_COMPOSITES = [
    "Economic_Stability_Index", "Social_Support_Index",
    "Maternal_MentalHealth_Risk_Index", "Neonatal_Delivery_Stress_Index",
]
WEIGHTED_COMPOSITES = [
    "Economic_Stability_Index_Weighted", "Social_Support_Index_Weighted",
    "Maternal_MentalHealth_Risk_Index_Weighted",
    "Neonatal_Delivery_Stress_Index_Weighted",
]
ALL_COLS = encoded.columns.tolist()
RAW_ONLY_COLS = [c for c in ALL_COLS
                 if c not in EQUAL_WEIGHT_COMPOSITES + WEIGHTED_COMPOSITES]
FEATURE_COLS = RAW_ONLY_COLS + WEIGHTED_COMPOSITES  # Arm B, matches step 8/9

X = encoded[FEATURE_COLS]
print(f"Feature set: Arm B 'Raw + Composite' -- {len(FEATURE_COLS)} features "
      f"(matches 08_ablation_study.py / 09_shap_explainability.py).")

# ---------------------------------------------------------------------------
# 3. Clinical banding functions -- IDENTICAL to 02_data_cleaning.py, so a
#    regression prediction is turned back into a category the exact same
#    way the official labels were built. Also define the correct ORDINAL
#    (not alphabetical) integer order for each category, required for QWK
#    to mean anything.
# ---------------------------------------------------------------------------
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


EPDS_ORDER = {"Low": 0, "Medium": 1, "High": 2}
PHQ9_ORDER = {"Minimal": 0, "Mild": 1, "Moderate": 2,
              "Moderately Severe": 3, "Severe": 4}

# Optional regressor imports (same pattern as every other script)
EXTRA_REGRESSORS = {}
try:
    from xgboost import XGBRegressor
    EXTRA_REGRESSORS["XGBoost"] = lambda: XGBRegressor(
        n_estimators=300, random_state=RANDOM_STATE)
except ImportError:
    print("[skip] xgboost not installed.")
try:
    from lightgbm import LGBMRegressor
    EXTRA_REGRESSORS["LightGBM"] = lambda: LGBMRegressor(
        n_estimators=300, random_state=RANDOM_STATE, verbose=-1)
except ImportError:
    print("[skip] lightgbm not installed.")
try:
    from catboost import CatBoostRegressor
    EXTRA_REGRESSORS["CatBoost"] = lambda: CatBoostRegressor(
        iterations=300, random_state=RANDOM_STATE, verbose=False)
except ImportError:
    print("[skip] catboost not installed.")


def get_regressors():
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE),
    }
    for name, factory in EXTRA_REGRESSORS.items():
        models[name] = factory()
    return models


# ---------------------------------------------------------------------------
# 4. Run the comparison for one ordinal target
# ---------------------------------------------------------------------------
def run_ordinal_comparison(score_col, category_col, band_fn, order_map):
    print(f"\n{'=' * 90}\nTARGET: {category_col} (via regression on '{score_col}')"
          f"\n{'=' * 90}")

    y_score = df[score_col].values
    y_cat_true = df[category_col].values
    y_cat_true_code = np.array([order_map[c] for c in y_cat_true])

    X_train, X_test, yscore_train, yscore_test, ycat_train, ycat_test, \
        ycatcode_train, ycatcode_test = train_test_split(
            X, y_score, y_cat_true, y_cat_true_code,
            test_size=0.2, stratify=y_cat_true, random_state=RANDOM_STATE)

    rows = []

    # --- 4a. Baseline: DIRECT CLASSIFIER on the category label (same
    #     approach as steps 7/8/9), same split, for a fair comparison ---
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=RANDOM_STATE)
    clf.fit(X_train, ycat_train)
    ycat_pred_direct = clf.predict(X_test)
    ycat_pred_direct_code = np.array([order_map[c] for c in ycat_pred_direct])
    rows.append({
        "approach": "Direct Classifier (Random Forest)",
        "r2": np.nan, "mae": np.nan, "rmse": np.nan,
        "accuracy": accuracy_score(ycat_test, ycat_pred_direct),
        "f1_macro": f1_score(ycat_test, ycat_pred_direct, average="macro",
                              zero_division=0),
        "qwk": cohen_kappa_score(ycatcode_test, ycat_pred_direct_code,
                                  weights="quadratic"),
    })

    # --- 4b. Regression + threshold, for every regressor family ---
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in get_regressors().items():
        cv_r2 = cross_val_score(model, X, y_score, cv=kf, scoring="r2")
        cv_mae = -cross_val_score(model, X, y_score, cv=kf,
                                   scoring="neg_mean_absolute_error")
        print(f"  {name}: CV R2={cv_r2.mean():.3f} +/- {cv_r2.std():.3f}, "
              f"CV MAE={cv_mae.mean():.3f}")

        model.fit(X_train, yscore_train)
        pred_score = model.predict(X_test)
        # Clip to the observed training range -- a regressor can predict
        # outside the valid clinical scale (e.g. negative), which would be
        # meaningless once banded into a category.
        pred_score_clipped = np.clip(pred_score, yscore_train.min(),
                                      yscore_train.max())
        pred_cat = np.array([band_fn(s) for s in pred_score_clipped])
        pred_cat_code = np.array([order_map[c] for c in pred_cat])

        rows.append({
            "approach": f"Regression + Threshold ({name})",
            "r2": r2_score(yscore_test, pred_score),
            "mae": mean_absolute_error(yscore_test, pred_score),
            "rmse": mean_squared_error(yscore_test, pred_score) ** 0.5,
            "accuracy": accuracy_score(ycat_test, pred_cat),
            "f1_macro": f1_score(ycat_test, pred_cat, average="macro",
                                  zero_division=0),
            "qwk": cohen_kappa_score(ycatcode_test, pred_cat_code,
                                      weights="quadratic"),
        })

    results_df = pd.DataFrame(rows).sort_values("qwk", ascending=False)
    print(f"\nResults for {category_col} (sorted by QWK, best first):")
    print(results_df.to_string(index=False))

    out_path = os.path.join(
        OUT_DIR, f"ordinal_regression_results_{category_col.replace(' ', '_')}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    # Also report QWK for the classifier's ORIGINAL macro-F1-optimized view,
    # to make the "did QWK reveal we were underselling ourselves" question
    # answerable directly: compare direct classifier's F1-macro (already
    # known from steps 7/8) against its QWK computed here.
    direct_row = results_df[results_df["approach"].str.contains("Direct")].iloc[0]
    print(f"\n  Direct classifier: F1-macro={direct_row['f1_macro']:.3f} vs "
          f"QWK={direct_row['qwk']:.3f} -- if QWK is notably higher than "
          f"F1-macro, most of the model's errors are near-miss (adjacent "
          f"band), not wild misses.")

    return results_df


results_epds = run_ordinal_comparison("EPDS Score", "EPDS Result", epds_band,
                                       EPDS_ORDER)
results_phq9 = run_ordinal_comparison("PHQ9 Score", "PHQ9 Result", phq9_band,
                                       PHQ9_ORDER)

# ---------------------------------------------------------------------------
# 5. Plain-language summary
# ---------------------------------------------------------------------------
summary_lines = []
for target_name, results_df in [("EPDS Result", results_epds),
                                 ("PHQ9 Result", results_phq9)]:
    best_row = results_df.iloc[0]
    direct_row = results_df[results_df["approach"].str.contains("Direct")].iloc[0]
    summary_lines.append(
        f"{target_name}:\n"
        f"  Direct Classifier -- accuracy={direct_row['accuracy']:.3f}, "
        f"F1-macro={direct_row['f1_macro']:.3f}, QWK={direct_row['qwk']:.3f}\n"
        f"  Best overall (by QWK) -- {best_row['approach']}: "
        f"accuracy={best_row['accuracy']:.3f}, F1-macro={best_row['f1_macro']:.3f}, "
        f"QWK={best_row['qwk']:.3f}\n"
        f"  -> {'Regression-then-threshold BEAT the direct classifier on QWK.' if best_row['approach'] != direct_row['approach'] else 'Direct classifier remained the best approach even on QWK.'}\n"
    )

summary_path = os.path.join(OUT_DIR, "ordinal_comparison_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print(f"\n\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
print("\n".join(summary_lines))
print(f"Saved -> {summary_path}")
print("\nDone. Compare QWK vs F1-macro for the direct classifier first (does "
      "QWK reveal the model is better than F1-macro suggested?), then check "
      "whether any regression-based approach beats the direct classifier.")
