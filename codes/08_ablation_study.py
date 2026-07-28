"""
08_ablation_study.py
=====================
THE core novelty validation of the whole project: does adding the
composite risk indices, and then selecting the best features, actually
improve prediction over raw features alone?

For each target (EPDS Result / PHQ9 Result / PPD_binary), 3 feature ARMS
are compared using the identical models/protocol as 07_modeling.py:

  Arm A "Raw only"          -- all one-hot encoded raw features + the 5
                               missing-value flags. NO composite indices.
  Arm B "Raw + Composite"   -- Arm A's features PLUS the 4 WEIGHTED
                               composite indices (Economic Stability,
                               Social Support, Maternal Mental-Health Risk,
                               Neonatal/Delivery Stress). The 4 EQUAL-weight
                               composite variants are deliberately excluded
                               here to avoid redundant, highly-correlated
                               duplicate signal (weighted versions were
                               shown in script 04/05 to be strictly more
                               significant and correctly-directioned).
  Arm C "Selected + Composite" -- starts from Arm B's feature set, keeps
                               only the top-N most important features
                               (via Random Forest importance), i.e. a
                               smaller, cleaner model.

IMPORTANT METHODOLOGICAL NOTE on Arm C -- NESTED feature selection:
Feature selection is done SEPARATELY inside every CV fold's training
portion (and separately inside the 80% holdout-training portion) --
never using the data a model will later be tested on. An earlier version
of this script selected the top-N features once using the FULL dataset
before running CV/holdout, which is a mild form of double-dipping (the
test folds influenced which features got selected before ever being
"held out"). This nested version fixes that: every fold/split re-derives
its own top-N feature list from ONLY its own training data, so the
reported numbers are honest, not optimistically biased.

If Arm B beats Arm A, the composite indices are adding real value beyond
what's already in the raw features. If Arm C matches or beats Arm B with
far fewer features, feature selection is also justified as a contribution.

STATISTICAL SIGNIFICANCE (added after Samir asked "dont we need any test to
proof our results?"): comparing mean CV F1-macro scores alone doesn't tell
you whether a difference (e.g. Arm B vs Arm A) is real or just noise. This
script now also runs a PAIRED test per model per target: the same 5
StratifiedKFold folds are reused across all 3 arms (same random_state), so
fold i's held-out patients are identical whether we're scoring Arm A, B, or
C -- meaning we can pair up "fold i under Arm A" with "fold i under Arm B"
and test whether Arm B is significantly better on those exact same patients.
Both a paired t-test and a Wilcoxon signed-rank test (nonparametric, safer
with a small sample) are reported. CAVEAT: n=5 folds is a small sample, so
these tests have low statistical power -- treat p-values as supporting
evidence for the thesis discussion, not as definitive proof.

Output (./outputs/):
  - ablation_results_EPDS_Result.csv
  - ablation_results_PHQ9_Result.csv
  - ablation_results_PPD_binary.csv
  - ablation_summary.txt (plain-language arm-vs-arm comparison per target)
  - ablation_significance.csv (paired t-test + Wilcoxon results per model,
    per target, for B-vs-A and C-vs-B)

Usage
-----
    python 08_ablation_study.py
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import ttest_rel, wilcoxon

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")

if not os.path.exists(MODEL_READY_PATH):
    raise SystemExit(f"{MODEL_READY_PATH} not found. Run 06_encoding.py first.")

df = pd.read_csv(MODEL_READY_PATH)
print(f"Loaded model-ready dataset: {df.shape}")

TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]
RANDOM_STATE = 42
TOP_N_SELECTED = 30  # how many features Arm C keeps

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

print(f"Arm A 'Raw only': {len(RAW_ONLY_COLS)} features")
print(f"Arm B 'Raw + Composite': {len(RAW_PLUS_COMPOSITE_COLS)} features "
      f"(+{len(WEIGHTED_COMPOSITES)} weighted composite indices)")

# ---------------------------------------------------------------------------
# Optional model imports (same pattern as 07_modeling.py)
# ---------------------------------------------------------------------------
EXTRA_MODELS = {}
try:
    from xgboost import XGBClassifier
    EXTRA_MODELS["XGBoost"] = lambda: XGBClassifier(
        n_estimators=300, random_state=RANDOM_STATE, eval_metric="logloss")
except ImportError:
    print("[skip] xgboost not installed.")
try:
    from lightgbm import LGBMClassifier
    EXTRA_MODELS["LightGBM"] = lambda: LGBMClassifier(
        n_estimators=300, random_state=RANDOM_STATE, verbose=-1)
except ImportError:
    print("[skip] lightgbm not installed.")
try:
    from catboost import CatBoostClassifier
    EXTRA_MODELS["CatBoost"] = lambda: CatBoostClassifier(
        iterations=300, random_state=RANDOM_STATE, verbose=False)
except ImportError:
    print("[skip] catboost not installed.")


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


# ---------------------------------------------------------------------------
# Helper: evaluate one (target, arm, feature-set) combination
# ---------------------------------------------------------------------------
def run_arm(target_col, arm_name, feature_cols, y, skf):
    X = df[feature_cols]
    rows = []
    # Per-fold F1-macro scores per model, kept ONLY in memory (not written to
    # the main results CSV) so we can run a paired significance test against
    # another arm's per-fold scores later. This is a valid pairing because
    # `skf` is the same StratifiedKFold object/random_state reused for every
    # arm of a given target, so fold i always contains the same patients.
    fold_f1 = {}
    for name, model in get_models().items():
        cv = cross_validate(model, X, y, cv=skf,
                             scoring=["accuracy", "f1_macro"])
        fold_f1[name] = cv["test_f1_macro"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        roc_auc = np.nan
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if len(np.unique(y)) == 2:
                    roc_auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, proba, multi_class="ovr",
                                             average="weighted")
        except Exception:
            pass
        rows.append({
            "target": target_col, "arm": arm_name, "model": name,
            "n_features": len(feature_cols),
            "cv_accuracy_mean": cv["test_accuracy"].mean(),
            "cv_f1_macro_mean": cv["test_f1_macro"].mean(),
            "holdout_accuracy": acc,
            "holdout_f1_macro": f1,
            "holdout_roc_auc": roc_auc,
        })
    return rows, fold_f1


def select_top_features(X_train, y_train, feature_pool, top_n):
    """Rank `feature_pool` columns of X_train by Random Forest importance,
    fit on X_train/y_train ONLY (never on data that will be tested on)."""
    selector = RandomForestClassifier(n_estimators=300,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE)
    selector.fit(X_train[feature_pool], y_train)
    importances = pd.Series(selector.feature_importances_, index=feature_pool)
    return importances.sort_values(ascending=False).head(top_n)


def run_arm_c_nested(target_col, feature_pool, y, skf, top_n):
    """Arm C, done properly: feature selection is re-derived inside every
    CV fold's training split, and inside the holdout's training split --
    never using data a model will subsequently be evaluated on."""
    X_pool = df[feature_pool]
    models_dict = get_models()
    per_model_cv = {name: {"accuracy": [], "f1_macro": []}
                     for name in models_dict}

    for train_idx, test_idx in skf.split(X_pool, y):
        X_tr, X_te = X_pool.iloc[train_idx], X_pool.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        fold_importances = select_top_features(X_tr, y_tr, feature_pool, top_n)
        fold_top_feats = fold_importances.index.tolist()

        for name, model in get_models().items():
            model.fit(X_tr[fold_top_feats], y_tr)
            y_pred = model.predict(X_te[fold_top_feats])
            per_model_cv[name]["accuracy"].append(accuracy_score(y_te, y_pred))
            per_model_cv[name]["f1_macro"].append(
                f1_score(y_te, y_pred, average="macro", zero_division=0))

    # Holdout: select features using ONLY the 80% training split, apply the
    # same selected list to the untouched 20% test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X_pool, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    holdout_importances = select_top_features(X_train, y_train, feature_pool,
                                                top_n)
    holdout_top_feats = holdout_importances.index.tolist()
    n_composite_in_top = sum(1 for f in holdout_top_feats
                              if f in WEIGHTED_COMPOSITES)
    print(f"  [Arm C holdout] top {top_n} features selected from the 80% "
          f"TRAINING split only ({n_composite_in_top}/{len(WEIGHTED_COMPOSITES)} "
          f"composite indices made the cut):")
    print("  " + holdout_importances.to_string().replace("\n", "\n  "))

    rows = []
    for name in models_dict:
        model = get_models()[name]
        model.fit(X_train[holdout_top_feats], y_train)
        y_pred = model.predict(X_test[holdout_top_feats])
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        roc_auc = np.nan
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test[holdout_top_feats])
                if len(np.unique(y)) == 2:
                    roc_auc = roc_auc_score(y_test, proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, proba, multi_class="ovr",
                                             average="weighted")
        except Exception:
            pass
        rows.append({
            "target": target_col, "arm": "C_Selected_plus_Composite",
            "model": name, "n_features": top_n,
            "cv_accuracy_mean": np.mean(per_model_cv[name]["accuracy"]),
            "cv_f1_macro_mean": np.mean(per_model_cv[name]["f1_macro"]),
            "holdout_accuracy": acc,
            "holdout_f1_macro": f1,
            "holdout_roc_auc": roc_auc,
        })

    # Per-fold F1-macro scores per model, for the paired significance test
    # against Arm B (see run_arm's docstring note -- same reasoning applies).
    fold_f1 = {name: np.array(per_model_cv[name]["f1_macro"])
               for name in models_dict}
    return rows, fold_f1


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
summary_lines = []
all_rows = []
all_sig_rows = []


def paired_test_rows(target_col, comparison_label, scores_new, scores_old, models_common):
    """Paired t-test + Wilcoxon signed-rank test on per-fold F1-macro scores,
    comparing `scores_new` (e.g. Arm B) against `scores_old` (e.g. Arm A) for
    each model, fold-by-fold. Both tests are reported: the paired t-test is
    the standard choice in ML papers, Wilcoxon is the safer nonparametric
    check since n=5 folds is a small sample and doesn't guarantee normality.
    CAVEAT (important for the thesis write-up): with only 5 paired
    observations (5 CV folds), these tests have low statistical power --
    treat the p-values as indicative evidence, not definitive proof."""
    out = []
    for name in models_common:
        new_s, old_s = np.asarray(scores_new[name]), np.asarray(scores_old[name])
        diff = new_s - old_s
        try:
            t_stat, p_t = ttest_rel(new_s, old_s)
        except Exception:
            t_stat, p_t = np.nan, np.nan
        try:
            # wilcoxon errors out if all differences are exactly zero
            w_stat, p_w = wilcoxon(new_s, old_s)
        except Exception:
            w_stat, p_w = np.nan, np.nan
        out.append({
            "target": target_col, "model": name, "comparison": comparison_label,
            "mean_diff_f1_macro": diff.mean(),
            "fold_scores_new": np.round(new_s, 4).tolist(),
            "fold_scores_old": np.round(old_s, 4).tolist(),
            "paired_ttest_stat": t_stat, "paired_ttest_pvalue": p_t,
            "wilcoxon_stat": w_stat, "wilcoxon_pvalue": p_w,
            "significant_at_0.05_ttest": (p_t < 0.05) if not np.isnan(p_t) else False,
        })
    return out


for target_col in TARGETS:
    print(f"\n{'=' * 90}\nTARGET: {target_col}\n{'=' * 90}")
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # --- Arm A: Raw only ---
    print("\n--- Arm A: Raw only ---")
    rows_a, fold_f1_a = run_arm(target_col, "A_Raw_only", RAW_ONLY_COLS, y, skf)
    all_rows.extend(rows_a)

    # --- Arm B: Raw + Weighted Composite ---
    print("--- Arm B: Raw + Composite ---")
    rows_b, fold_f1_b = run_arm(target_col, "B_Raw_plus_Composite",
                                 RAW_PLUS_COMPOSITE_COLS, y, skf)
    all_rows.extend(rows_b)

    # --- Arm C: Selected + Composite (NESTED feature selection -- see
    #     module docstring for why this matters) ---
    print("\n--- Arm C: Selected + Composite (nested feature selection) ---")
    rows_c, fold_f1_c = run_arm_c_nested(target_col, RAW_PLUS_COMPOSITE_COLS, y, skf,
                                          TOP_N_SELECTED)
    all_rows.extend(rows_c)

    # --- Paired significance tests: is the arm-vs-arm difference real, or
    #     could it plausibly be noise? Uses the SAME 5 CV folds (skf reused
    #     across arms A/B/C for this target) so each model's fold-i score
    #     under Arm A and fold-i score under Arm B are directly comparable
    #     (same held-out patients both times). ---
    print(f"\n--- Statistical significance: paired tests across 5 CV folds "
          f"(caveat: n=5 folds -> low power, treat as indicative) ---")
    models_common = list(fold_f1_a.keys())
    sig_b_vs_a = paired_test_rows(target_col, "B_vs_A (composite effect)",
                                   fold_f1_b, fold_f1_a, models_common)
    sig_c_vs_b = paired_test_rows(target_col, "C_vs_B (feature-selection effect)",
                                   fold_f1_c, fold_f1_b, models_common)
    for row in sig_b_vs_a + sig_c_vs_b:
        flag = "SIGNIFICANT (p<0.05)" if row["significant_at_0.05_ttest"] else "not significant"
        print(f"  [{row['comparison']}] {row['model']}: "
              f"mean_diff={row['mean_diff_f1_macro']:+.4f}, "
              f"paired t-test p={row['paired_ttest_pvalue']:.3f}, "
              f"wilcoxon p={row['wilcoxon_pvalue']:.3f} -> {flag}")
    all_sig_rows.extend(sig_b_vs_a)
    all_sig_rows.extend(sig_c_vs_b)

    # --- Per-target summary (average across models, per arm) ---
    target_df = pd.DataFrame(rows_a + rows_b + rows_c)
    arm_summary = target_df.groupby("arm").agg(
        n_features=("n_features", "first"),
        avg_cv_accuracy=("cv_accuracy_mean", "mean"),
        avg_cv_f1_macro=("cv_f1_macro_mean", "mean"),
        avg_holdout_accuracy=("holdout_accuracy", "mean"),
        avg_holdout_f1_macro=("holdout_f1_macro", "mean"),
    ).reset_index()
    print(f"\n>>> Arm comparison for {target_col} (averaged across all models):")
    print(arm_summary.to_string(index=False))

    a_f1 = arm_summary.loc[arm_summary.arm == "A_Raw_only", "avg_cv_f1_macro"].iloc[0]
    b_f1 = arm_summary.loc[arm_summary.arm == "B_Raw_plus_Composite", "avg_cv_f1_macro"].iloc[0]
    c_f1 = arm_summary.loc[arm_summary.arm == "C_Selected_plus_Composite", "avg_cv_f1_macro"].iloc[0]
    delta_b = b_f1 - a_f1
    delta_c = c_f1 - a_f1

    verdict = (
        f"\n{target_col}: Raw={a_f1:.3f}, Raw+Composite={b_f1:.3f} "
        f"({'+' if delta_b >= 0 else ''}{delta_b:.3f}), "
        f"Selected+Composite={c_f1:.3f} ({'+' if delta_c >= 0 else ''}"
        f"{delta_c:.3f}) [avg CV F1-macro across all models]\n"
        f"  -> Composite features {'HELPED' if delta_b > 0 else 'did NOT help'} "
        f"for this target.\n"
        f"  -> Feature selection ({TOP_N_SELECTED} features) "
        f"{'matched/beat' if delta_c >= delta_b - 0.01 else 'underperformed vs'} "
        f"the full Raw+Composite set."
    )
    print(verdict)
    summary_lines.append(verdict)

    out_path = os.path.join(OUT_DIR,
                             f"ablation_results_{target_col.replace(' ', '_')}.csv")
    target_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

with open(os.path.join(OUT_DIR, "ablation_summary.txt"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

sig_df = pd.DataFrame(all_sig_rows)
sig_path = os.path.join(OUT_DIR, "ablation_significance.csv")
sig_df.to_csv(sig_path, index=False)

print(f"\n\n{'=' * 90}\nFINAL ABLATION SUMMARY (all targets)\n{'=' * 90}")
print("\n".join(summary_lines))
print(f"\nSaved -> {os.path.join(OUT_DIR, 'ablation_summary.txt')}")
print(f"Saved -> {sig_path}")

n_sig = int(sig_df["significant_at_0.05_ttest"].sum())
print(f"\n{n_sig}/{len(sig_df)} arm-vs-arm comparisons were statistically "
      f"significant at p<0.05 (paired t-test, n=5 folds -- low power, "
      f"read this as supporting evidence, not final proof).")
print("\nDone. This is the core evidence for the thesis's novelty claim: "
      "whether composite risk indices and feature selection genuinely "
      "improve prediction over raw features alone.")
