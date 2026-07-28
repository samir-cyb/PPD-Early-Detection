"""
12_catboost_native_and_repeated_cv.py
========================================
Two further improvement ideas bundled together (both cheap to run, no new
data required):

PART A -- CatBoost with NATIVE categorical handling.
Every prior script (06 onward) one-hot encodes every categorical safe
feature before modeling, turning ~53 original questions into 108 sparse
binary dummy columns. One-hot encoding is necessary for models like
Logistic Regression, Random Forest, and XGBoost/LightGBM (in their default
mode), but CatBoost has a native categorical-feature mode (ordered target
statistics internally) that often outperforms one-hot encoding, especially
when many categorical columns are involved, because it avoids diluting a
single underlying question across many sparse dummy columns. This script
trains CatBoost BOTH ways (one-hot vs native categorical) on the identical
train/test split for all 3 targets, so the comparison is direct.

PART B -- Repeated Cross-Validation for higher statistical power.
08_ablation_study.py's paired significance test (Arm B vs Arm A) used a
single 5-fold split -- only 5 paired observations per comparison, which
has very low statistical power (a real but small effect will often fail
to reach p<0.05 with n=5). This script re-runs that same Arm A vs Arm B
comparison using RepeatedStratifiedKFold (5 folds x 5 repeats = 25 paired
observations per model per target) -- the same underlying 800 rows, no new
data, but a much more stable and higher-powered significance estimate.

Output (./outputs/):
  - catboost_native_vs_onehot_<target>.csv
  - repeated_cv_significance.csv
  - improvement_experiments_summary.txt

Usage
-----
    pip install catboost --break-system-packages   (if not already installed)
    python 12_catboost_native_and_repeated_cv.py

NOTE: Part B fits many models (3 targets x up to 5 model types x 25 folds
x 2 arms). This is more compute than earlier scripts -- expect several
minutes to run. If it's too slow on your machine, lower N_REPEATS below
(e.g. to 3) for a faster, still-improved-power run.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      RepeatedStratifiedKFold, cross_validate)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
ENGINEERED_PATH = os.path.join(OUT_DIR, "PPD_dataset_with_composite_features.csv")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")

RANDOM_STATE = 42
TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
N_REPEATS = 5  # 5 folds x 5 repeats = 25 paired observations per comparison

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("catboost is not installed -- Part A needs it. "
          "Run: pip install catboost --break-system-packages")

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

# ===========================================================================
# PART A: CatBoost native categorical vs one-hot
# ===========================================================================
if HAS_CAT and os.path.exists(ENGINEERED_PATH):
    print(f"\n{'#' * 90}\nPART A: CatBoost -- native categorical vs one-hot\n{'#' * 90}")
    raw_df = pd.read_csv(ENGINEERED_PATH)
    drop_cols = [c for c in EPDS_ITEMS + PHQ9_ITEMS + RAW_SCORES
                 if c in raw_df.columns]
    raw_df = raw_df.drop(columns=drop_cols)

    for target_col in TARGETS:
        print(f"\n--- TARGET: {target_col} ---")
        other_targets = [t for t in TARGETS if t != target_col]
        features_native = raw_df.drop(columns=[t for t in other_targets
                                                 if t in raw_df.columns] + [target_col])
        cat_cols = features_native.select_dtypes(include="object").columns.tolist()
        # CatBoost requires no NaN in categorical columns as float; fill any
        # stray NaN with a literal string category (there should be none
        # left post-cleaning, but this guards against surprises).
        features_native[cat_cols] = features_native[cat_cols].fillna("Unknown")
        cat_feature_idx = [features_native.columns.get_loc(c) for c in cat_cols]

        le = LabelEncoder()
        y = le.fit_transform(raw_df[target_col])

        Xn_train, Xn_test, y_train, y_test = train_test_split(
            features_native, y, test_size=0.2, stratify=y,
            random_state=RANDOM_STATE)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        # Native categorical CatBoost -- NOTE: sklearn's cross_validate()
        # calls clone() internally, and CatBoostClassifier does not clone
        # cleanly when cat_features is set (a known CatBoost/sklearn
        # compatibility quirk -- clone() raises RuntimeError). Worked around
        # by doing the 5-fold CV manually: a brand-new CatBoostClassifier is
        # constructed for every fold instead of being cloned.
        native_acc_scores, native_f1_scores = [], []
        for tr_idx, te_idx in skf.split(features_native, y):
            fold_model = CatBoostClassifier(iterations=300, random_state=RANDOM_STATE,
                                             verbose=False, cat_features=cat_feature_idx)
            fold_model.fit(features_native.iloc[tr_idx], y[tr_idx])
            fold_pred = fold_model.predict(features_native.iloc[te_idx]).ravel()
            native_acc_scores.append(accuracy_score(y[te_idx], fold_pred))
            native_f1_scores.append(f1_score(y[te_idx], fold_pred,
                                              average="macro", zero_division=0))
        cv_native = {"test_accuracy": np.array(native_acc_scores),
                     "test_f1_macro": np.array(native_f1_scores)}

        native_model = CatBoostClassifier(iterations=300, random_state=RANDOM_STATE,
                                           verbose=False, cat_features=cat_feature_idx)
        native_model.fit(Xn_train, y_train)
        pred_native = native_model.predict(Xn_test).ravel()
        native_holdout_f1 = f1_score(y_test, pred_native, average="macro",
                                      zero_division=0)

        # One-hot CatBoost, identical split (rebuild one-hot from the same
        # underlying rows so indices line up)
        encoded = pd.get_dummies(features_native, columns=cat_cols,
                                  drop_first=True, dtype=int)
        Xo_train, Xo_test, _, _ = train_test_split(
            encoded, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        onehot_model = CatBoostClassifier(iterations=300, random_state=RANDOM_STATE,
                                           verbose=False)
        cv_onehot = cross_validate(onehot_model, encoded, y, cv=skf,
                                    scoring=["accuracy", "f1_macro"])
        onehot_model.fit(Xo_train, y_train)
        pred_onehot = onehot_model.predict(Xo_test).ravel()
        onehot_holdout_f1 = f1_score(y_test, pred_onehot, average="macro",
                                      zero_division=0)

        comp_df = pd.DataFrame([
            {"encoding": "Native categorical", "n_features": features_native.shape[1],
             "cv_f1_macro_mean": cv_native["test_f1_macro"].mean(),
             "cv_accuracy_mean": cv_native["test_accuracy"].mean(),
             "holdout_f1_macro": native_holdout_f1},
            {"encoding": "One-hot", "n_features": encoded.shape[1],
             "cv_f1_macro_mean": cv_onehot["test_f1_macro"].mean(),
             "cv_accuracy_mean": cv_onehot["test_accuracy"].mean(),
             "holdout_f1_macro": onehot_holdout_f1},
        ])
        print(comp_df.to_string(index=False))
        out_path = os.path.join(
            OUT_DIR, f"catboost_native_vs_onehot_{target_col.replace(' ', '_')}.csv")
        comp_df.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")
else:
    if not HAS_CAT:
        print("Skipping Part A (catboost not installed).")
    else:
        print(f"Skipping Part A ({ENGINEERED_PATH} not found).")

# ===========================================================================
# PART B: Repeated CV significance test (Arm A vs Arm B)
# ===========================================================================
if os.path.exists(MODEL_READY_PATH):
    print(f"\n{'#' * 90}\nPART B: Repeated CV (5x{N_REPEATS}={5*N_REPEATS} folds) "
          f"significance test\n{'#' * 90}")
    df = pd.read_csv(MODEL_READY_PATH)

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
    RAW_PLUS_COMPOSITE_COLS = RAW_ONLY_COLS + WEIGHTED_COMPOSITES

    EXTRA_MODELS = {}
    try:
        from xgboost import XGBClassifier
        EXTRA_MODELS["XGBoost"] = lambda: XGBClassifier(
            n_estimators=300, random_state=RANDOM_STATE, eval_metric="logloss")
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        EXTRA_MODELS["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=300, random_state=RANDOM_STATE, verbose=-1)
    except ImportError:
        pass
    if HAS_CAT:
        EXTRA_MODELS["CatBoost"] = lambda: CatBoostClassifier(
            iterations=300, random_state=RANDOM_STATE, verbose=False)

    def get_models():
        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                            random_state=RANDOM_STATE)),
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=RANDOM_STATE),
        }
        for name, factory in EXTRA_MODELS.items():
            models[name] = factory()
        return models

    all_sig_rows = []
    for target_col in TARGETS:
        print(f"\n--- TARGET: {target_col} ---")
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS,
                                        random_state=RANDOM_STATE)

        fold_f1_a, fold_f1_b = {}, {}
        for name, model in get_models().items():
            cv_a = cross_validate(model, df[RAW_ONLY_COLS], y, cv=rskf,
                                   scoring=["f1_macro"])
            fold_f1_a[name] = cv_a["test_f1_macro"]
            model_b = get_models()[name]
            cv_b = cross_validate(model_b, df[RAW_PLUS_COMPOSITE_COLS], y,
                                   cv=rskf, scoring=["f1_macro"])
            fold_f1_b[name] = cv_b["test_f1_macro"]
            print(f"  {name}: Arm A mean F1={fold_f1_a[name].mean():.3f}, "
                  f"Arm B mean F1={fold_f1_b[name].mean():.3f} "
                  f"(n={len(fold_f1_a[name])} folds)")

        for name in get_models():
            a_scores, b_scores = fold_f1_a[name], fold_f1_b[name]
            diff = b_scores - a_scores
            try:
                t_stat, p_t = ttest_rel(b_scores, a_scores)
            except Exception:
                t_stat, p_t = np.nan, np.nan
            try:
                w_stat, p_w = wilcoxon(b_scores, a_scores)
            except Exception:
                w_stat, p_w = np.nan, np.nan
            all_sig_rows.append({
                "target": target_col, "model": name,
                "n_folds": len(a_scores),
                "mean_diff_f1_macro": diff.mean(),
                "paired_ttest_pvalue": p_t, "wilcoxon_pvalue": p_w,
                "significant_at_0.05": bool(p_t < 0.05) if not np.isnan(p_t) else False,
            })

    sig_df = pd.DataFrame(all_sig_rows)
    sig_path = os.path.join(OUT_DIR, "repeated_cv_significance.csv")
    sig_df.to_csv(sig_path, index=False)
    print(f"\n{sig_df.to_string(index=False)}")
    print(f"\nSaved -> {sig_path}")
    n_sig = int(sig_df["significant_at_0.05"].sum())
    print(f"\n{n_sig}/{len(sig_df)} comparisons significant at p<0.05 with "
          f"{5*N_REPEATS} folds (compare to 08_ablation_study.py's 0/30 "
          f"with only 5 folds).")
else:
    print(f"Skipping Part B ({MODEL_READY_PATH} not found).")

print("\n\nDone. Part A shows whether native-categorical CatBoost beats "
      "one-hot; Part B shows whether more folds (same data, more repeats) "
      "reveal statistically significant composite-feature gains that the "
      "original 5-fold test lacked the power to detect.")
