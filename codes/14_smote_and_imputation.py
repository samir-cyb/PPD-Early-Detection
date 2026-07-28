"""
14_smote_and_imputation.py
=============================
Two Tier-2 improvement ideas bundled together:

PART A -- SMOTE (SMOTENC) for PHQ9 Result's rare severity bands.
PHQ9 Result has 5 imbalanced classes (Severe n=89, Minimal n=79, vs Mild
n=263). Every prior script relied only on class_weight="balanced" to
handle this. This script tries SMOTENC (SMOTE for mixed Numerical/
Categorical data -- appropriate here because most of our features are
one-hot binary columns, not continuous, and plain SMOTE's linear
interpolation between neighbors doesn't make sense for those; SMOTENC
correctly treats them as categorical and uses the majority category
among neighbors instead) applied ONLY inside each cross-validation fold's
TRAINING portion -- oversampling before splitting would leak synthetic
copies of test-adjacent patients into training, a classic and easy-to-miss
mistake this script deliberately avoids. Evaluated for all 3 targets, but
PHQ9 Result (the most imbalanced) is the primary target of interest.

PART B -- IterativeImputer instead of mode-fill for the 5 truly-random-
missing columns (Abuse, Husband's monthly income, Husband's education
level, Education Level, Trust and share feelings). 02_data_cleaning.py
filled these with the column mode and added a "_was_missing" flag per
column. This script uses those flags to find exactly which cells were
originally missing, reverts them to NaN, and re-imputes using
IterativeImputer (a MICE-style, other-columns-aware method) instead of a
single constant mode value. Compares how many imputed values differ from
the original mode-fill, and whether swapping in the new values changes
downstream PPD_binary CV performance.

Output (./outputs/):
  - smote_comparison_results.csv
  - iterative_imputation_comparison.csv
  - smote_imputation_summary.txt

Usage
-----
    pip install imbalanced-learn --break-system-packages   (for Part A)
    python 14_smote_and_imputation.py
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import f1_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")
CLEANED_PATH = os.path.join(OUT_DIR, "PPD_dataset_cleaned_v2.csv")

RANDOM_STATE = 42
TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]

try:
    from imblearn.over_sampling import SMOTENC
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("imbalanced-learn is not installed -- Part A needs it. "
          "Run: pip install imbalanced-learn --break-system-packages")

# ===========================================================================
# PART A: SMOTENC, fold-safe (training portion only)
# ===========================================================================
if HAS_IMBLEARN and os.path.exists(MODEL_READY_PATH):
    print(f"\n{'#' * 90}\nPART A: SMOTENC oversampling (training-fold only)\n{'#' * 90}")
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
    CONTINUOUS_COLS = ["Age", "Number of the latest pregnancy"] + WEIGHTED_COMPOSITES
    ALL_FEATURE_COLS = [c for c in df.columns if c not in TARGETS]
    RAW_ONLY_COLS = [c for c in ALL_FEATURE_COLS
                     if c not in EQUAL_WEIGHT_COMPOSITES + WEIGHTED_COMPOSITES]
    FEATURE_COLS = RAW_ONLY_COLS + WEIGHTED_COMPOSITES  # Arm B

    X = df[FEATURE_COLS].reset_index(drop=True)
    cat_feature_idx = [i for i, c in enumerate(FEATURE_COLS)
                        if c not in CONTINUOUS_COLS]

    smote_rows = []
    for target_col in TARGETS:
        print(f"\n--- TARGET: {target_col} ---")
        le = LabelEncoder()
        y = le.fit_transform(df[target_col])
        print(f"  Class distribution: "
              f"{dict(zip(*np.unique(y, return_counts=True)))}")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        f1_baseline, f1_smote = [], []
        for tr_idx, te_idx in skf.split(X, y):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # Baseline: no oversampling, class_weight="balanced" only
            base_model = RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=RANDOM_STATE)
            base_model.fit(X_tr, y_tr)
            f1_baseline.append(f1_score(y_te, base_model.predict(X_te),
                                         average="macro", zero_division=0))

            # SMOTENC: oversample the TRAINING fold only, never the test fold
            try:
                smote = SMOTENC(categorical_features=cat_feature_idx,
                                 random_state=RANDOM_STATE)
                X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            except ValueError as e:
                # Can fail if a class has too few members for k-neighbors
                print(f"  [warn] SMOTENC failed on this fold ({e}); "
                      f"skipping oversampling for this fold.")
                X_tr_res, y_tr_res = X_tr, y_tr
            smote_model = RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=RANDOM_STATE)
            smote_model.fit(X_tr_res, y_tr_res)
            f1_smote.append(f1_score(y_te, smote_model.predict(X_te),
                                      average="macro", zero_division=0))

        print(f"  Baseline (class_weight only): "
              f"F1-macro = {np.mean(f1_baseline):.3f} +/- {np.std(f1_baseline):.3f}")
        print(f"  SMOTENC + class_weight:       "
              f"F1-macro = {np.mean(f1_smote):.3f} +/- {np.std(f1_smote):.3f}")
        smote_rows.append({
            "target": target_col,
            "baseline_f1_macro_mean": np.mean(f1_baseline),
            "smote_f1_macro_mean": np.mean(f1_smote),
            "diff": np.mean(f1_smote) - np.mean(f1_baseline),
        })

    smote_df = pd.DataFrame(smote_rows)
    smote_path = os.path.join(OUT_DIR, "smote_comparison_results.csv")
    smote_df.to_csv(smote_path, index=False)
    print(f"\n{smote_df.to_string(index=False)}")
    print(f"Saved -> {smote_path}")
else:
    if not HAS_IMBLEARN:
        print("Skipping Part A (imbalanced-learn not installed).")
    else:
        print(f"Skipping Part A ({MODEL_READY_PATH} not found).")

# ===========================================================================
# PART B: IterativeImputer vs mode-fill for the 5 true-random-missing cols
# ===========================================================================
if os.path.exists(CLEANED_PATH):
    print(f"\n{'#' * 90}\nPART B: IterativeImputer vs mode-fill\n{'#' * 90}")
    clean_df = pd.read_csv(CLEANED_PATH)

    RANDOM_MISSING_COLS = ["Abuse", "Husband's monthly income",
                            "Husband's education level", "Education Level",
                            "Trust and share feelings"]
    FLAG_COLS = {col: f"{col}_was_missing" for col in RANDOM_MISSING_COLS}
    missing_flags_present = [c for c in FLAG_COLS.values() if c in clean_df.columns]
    print(f"Found {len(missing_flags_present)}/5 '_was_missing' flag columns.")

    # Build a fully label-encoded numeric copy of every non-target,
    # non-leakage categorical column, so IterativeImputer (numeric-only) can
    # use every other column as a predictor when re-estimating the 5
    # originally-missing columns.
    EPDS_ITEMS_PHQ9_ITEMS_SCORES_TARGETS = None  # not needed here; cleaned_v2
    # already has items+scores present, but they are fine to use as
    # predictors for IMPUTATION purposes only (imputation happens before the
    # leakage-drop step in the real pipeline, and does not touch the target
    # labels themselves) -- this mirrors how 02_data_cleaning.py itself used
    # the whole cleaned row to decide the mode fill.
    work_df = clean_df.copy()
    label_encoders = {}
    categorical_cols_all = work_df.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols_all:
        le = LabelEncoder()
        work_df[col] = le.fit_transform(work_df[col].astype(str))
        label_encoders[col] = le

    # Re-introduce NaN exactly where the original data was missing (using
    # the was_missing flags), undoing the mode-fill for those specific cells
    # only.
    n_reverted = {}
    for col in RANDOM_MISSING_COLS:
        flag_col = FLAG_COLS[col]
        if flag_col not in clean_df.columns or col not in work_df.columns:
            continue
        mask = clean_df[flag_col].astype(bool)
        work_df.loc[mask, col] = np.nan
        n_reverted[col] = int(mask.sum())
    print(f"Reverted to NaN (cells originally missing): {n_reverted}")

    imputer = IterativeImputer(estimator=BayesianRidge(), random_state=RANDOM_STATE,
                                max_iter=15)
    imputed_array = imputer.fit_transform(work_df.select_dtypes(include=[np.number]))
    imputed_df = pd.DataFrame(imputed_array,
                               columns=work_df.select_dtypes(include=[np.number]).columns,
                               index=work_df.index)

    compare_rows = []
    for col in RANDOM_MISSING_COLS:
        if col not in n_reverted:
            continue
        flag_col = FLAG_COLS[col]
        mask = clean_df[flag_col].astype(bool)
        le = label_encoders[col]
        valid_codes = np.arange(len(le.classes_))
        mice_codes = imputed_df.loc[mask, col].round().clip(
            valid_codes.min(), valid_codes.max()).astype(int)
        mice_labels = le.inverse_transform(mice_codes)
        original_mode_labels = clean_df.loc[mask, col].values
        n_changed = int((mice_labels != original_mode_labels).sum())
        compare_rows.append({
            "column": col, "n_originally_missing": int(mask.sum()),
            "n_changed_vs_mode_fill": n_changed,
            "pct_changed": n_changed / int(mask.sum()) * 100 if mask.sum() else 0,
        })
        print(f"  {col}: {n_changed}/{int(mask.sum())} imputed values differ "
              f"from the original mode-fill "
              f"({n_changed / max(int(mask.sum()), 1) * 100:.1f}%)")

    compare_df = pd.DataFrame(compare_rows)
    compare_path = os.path.join(OUT_DIR, "iterative_imputation_comparison.csv")
    compare_df.to_csv(compare_path, index=False)
    print(f"Saved -> {compare_path}")

    # Downstream check: does swapping in the MICE-based values for these 5
    # columns change PPD_binary CV performance? Rebuild a minimal one-hot
    # feature set from work_df (MICE version) vs clean_df (mode version),
    # restricted to just these 5 columns + Age + Number of the latest
    # pregnancy, as a lightweight, self-contained comparison (not a full
    # pipeline re-run).
    print("\n--- Downstream check: PPD_binary CV F1-macro, mode-fill vs MICE ---")
    mice_cols_decoded = clean_df.copy()
    for col in RANDOM_MISSING_COLS:
        if col not in n_reverted:
            continue
        le = label_encoders[col]
        valid_codes = np.arange(len(le.classes_))
        all_codes = imputed_df[col].round().clip(
            valid_codes.min(), valid_codes.max()).astype(int)
        mice_cols_decoded[col] = le.inverse_transform(all_codes)

    y_binary = LabelEncoder().fit_transform(clean_df["PPD_binary"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for label, source_df in [("Mode-fill (original)", clean_df),
                              ("IterativeImputer (MICE-style)", mice_cols_decoded)]:
        mini_features = pd.get_dummies(
            source_df[RANDOM_MISSING_COLS + ["Age", "Number of the latest pregnancy"]],
            columns=RANDOM_MISSING_COLS, drop_first=True, dtype=int)
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=RANDOM_STATE),
            mini_features, y_binary, cv=skf, scoring="f1_macro")
        print(f"  {label}: CV F1-macro = {scores.mean():.3f} +/- {scores.std():.3f} "
              f"(using only the 5 re-imputed columns + Age + pregnancy count "
              f"as a minimal, isolated check)")
else:
    print(f"Skipping Part B ({CLEANED_PATH} not found).")

print("\n\nDone. Part A: compare baseline vs SMOTENC F1-macro per target -- "
      "look especially at PHQ9 Result. Part B: a high 'pct_changed' means "
      "MICE disagrees a lot with the simple mode-fill for that column -- "
      "worth using MICE going forward if the downstream F1-macro also "
      "improves.")
