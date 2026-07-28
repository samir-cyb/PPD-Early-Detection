"""
09_shap_explainability.py
===========================
Step 9: "best model + SHAP explainability" -- the interpretability chapter
of the thesis. Answers: exactly which risk factors (raw + composite) drive
each model's predictions, and in which direction?

Design decisions
-----------------
1. FEATURE SET = Arm B "Raw + Composite" (all 108 raw + 4 weighted composite
   indices = 112 features), NOT Arm C's 30-feature subset. Why: Step 8's
   ablation study (with the honest nested-CV fix) showed Arm C's benefit
   over Arm B is small, inconsistent across targets, and NOT statistically
   significant in paired testing (0/30 comparisons reached p<0.05) -- for
   `PHQ9 Result` specifically, Arm C was even slightly WORSE than Arm B. So
   the full Raw+Composite set is the safer, more defensible choice for the
   "final" explainable model across all 3 targets, rather than picking a
   30-feature subset whose advantage isn't solidly proven.

2. MODEL = Random Forest (class_weight="balanced", n_estimators=300,
   random_state=42) -- the same model used throughout scripts 07/08. Chosen
   over the Stacking Ensemble (07b's best performer for PPD_binary) because
   shap.TreeExplainer gives fast, EXACT Shapley values for a single tree
   ensemble, whereas explaining a stacking ensemble (4 different tree models
   + a Logistic Regression meta-learner) would need a slower, approximate
   KernelExplainer and produces a much messier, harder-to-defend
   interpretability story. A single, well-understood Random Forest is the
   right trade-off for a THESIS explainability chapter (clarity > the last
   1-2% of accuracy).

3. SHAP VALUES ARE COMPUTED ON THE HELD-OUT 20% TEST SET (never-seen data),
   not on the training data -- so the explanation reflects how the model
   behaves on genuinely unseen patients, not patterns it merely memorized.

4. FOCUS CLASS per target (for multi-class targets, SHAP gives one set of
   values per class -- we pick the single most clinically relevant class to
   report on, to keep the interpretation simple and focused):
     - PPD_binary       -> class 1 ("PPD present")
     - EPDS Result       -> "High" (highest EPDS risk band)
     - PHQ9 Result       -> "Severe" (highest PHQ-9 severity band)

Output (./outputs/):
  - shap_feature_importance_<target>.csv   -- every feature's mean |SHAP|,
                                               ranked, with a flag for
                                               whether it's a composite index
  - shap_summary_bar_<target>.png          -- top-20 global importance bar
  - shap_beeswarm_<target>.png             -- top-20 beeswarm (direction +
                                               magnitude per patient)
  - shap_dependence_<CompositeName>_<target>.png -- one per composite index,
                                               shows how SHAP value changes
                                               as the composite score changes
  - shap_waterfall_highrisk_<target>.png   -- one real high-risk patient,
                                               feature-by-feature breakdown
  - shap_waterfall_lowrisk_<target>.png    -- one real low-risk patient, for
                                               contrast
  - shap_values_<target>.npz               -- raw SHAP values + test data,
                                               for any follow-up analysis
  - shap_interpretation_<target>.txt       -- plain-language summary

Usage
-----
    pip install shap matplotlib --break-system-packages
    python 09_shap_explainability.py

NOTE: SHAP's plotting API has changed across versions over the years. Every
plotting block below is wrapped in try/except so a version-mismatch failure
on ONE plot does not stop the whole script -- the CSV/interpretation-text
outputs (the most important deliverables) will still be produced even if a
plot fails. If a plot fails, paste the printed error back so it can be
fixed for your exact installed shap version.
"""

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend -- no display needed to save PNGs
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import shap
except ImportError:
    raise SystemExit(
        "shap is not installed. Run:\n"
        "    pip install shap --break-system-packages\n"
        "then re-run this script."
    )

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")

if not os.path.exists(MODEL_READY_PATH):
    raise SystemExit(f"{MODEL_READY_PATH} not found. Run 06_encoding.py first.")

df = pd.read_csv(MODEL_READY_PATH)
print(f"Loaded model-ready dataset: {df.shape}")

RANDOM_STATE = 42
TARGETS = ["EPDS Result", "PHQ9 Result", "PPD_binary"]

# Same composite-index bookkeeping as 08_ablation_study.py, so the feature
# set here (Arm B) matches exactly what the ablation study validated.
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
FEATURE_COLS = RAW_ONLY_COLS + WEIGHTED_COMPOSITES  # Arm B, 112 features

print(f"Using Arm B 'Raw + Composite' feature set: {len(FEATURE_COLS)} features "
      f"(matches 08_ablation_study.py's validated choice).")

# The single most clinically-relevant class to focus the SHAP report on, per
# target (multi-class targets get one SHAP array PER class -- reporting all
# of them would be overwhelming for a thesis chapter).
FOCUS_CLASS = {
    "PPD_binary": "1",
    "EPDS Result": "High",
    "PHQ9 Result": "Severe",
}

TOP_N_REPORT = 20


def get_shap_matrix_for_class(shap_raw, focus_idx, n_features):
    """Normalize SHAP's output (which varies across shap versions and
    binary-vs-multiclass) into a single (n_samples, n_features) array for
    the class we care about."""
    if isinstance(shap_raw, list):
        # Older shap: list of length n_classes, each (n_samples, n_features)
        return np.asarray(shap_raw[focus_idx])
    shap_raw = np.asarray(shap_raw)
    if shap_raw.ndim == 3:
        # Newer shap: (n_samples, n_features, n_classes)
        if shap_raw.shape[1] == n_features:
            return shap_raw[:, :, focus_idx]
        # Some versions transpose to (n_samples, n_classes, n_features)
        return shap_raw[:, focus_idx, :]
    # Already 2D -- binary case where shap only returns the positive class
    return shap_raw


def get_base_value_for_class(expected_value, focus_idx):
    ev = np.asarray(expected_value) if not np.isscalar(expected_value) else expected_value
    if np.isscalar(ev):
        return float(ev)
    ev = np.atleast_1d(ev)
    if len(ev) > focus_idx:
        return float(ev[focus_idx])
    return float(ev[0])


for target_col in TARGETS:
    print(f"\n{'=' * 90}\nTARGET: {target_col}\n{'=' * 90}")
    safe_name = target_col.replace(" ", "_")

    X = df[FEATURE_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    class_names = [str(c) for c in le.classes_]
    print(f"Classes: {class_names}")

    focus_label = FOCUS_CLASS[target_col]
    if focus_label not in class_names:
        print(f"  [warn] Focus class '{focus_label}' not found in "
              f"{class_names}; defaulting to the last (highest-coded) class.")
        focus_idx = len(class_names) - 1
        focus_label = class_names[focus_idx]
    else:
        focus_idx = class_names.index(focus_label)
    print(f"  Focus class for this report: '{focus_label}' (index {focus_idx})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"  Random Forest fit -- train accuracy={train_acc:.3f}, "
          f"test accuracy={test_acc:.3f}")

    # ------------------------------------------------------------------
    # SHAP values on the held-out test set (never seen during training)
    # ------------------------------------------------------------------
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_test)
    shap_2d = get_shap_matrix_for_class(shap_raw, focus_idx, X_test.shape[1])
    base_value = get_base_value_for_class(explainer.expected_value, focus_idx)
    print(f"  SHAP values computed on held-out test set: shape {shap_2d.shape}")

    # ------------------------------------------------------------------
    # Global feature importance = mean(|SHAP value|) per feature
    # ------------------------------------------------------------------
    mean_abs_shap = np.abs(shap_2d).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_shap": mean_abs_shap,
        "is_composite_index": [f in WEIGHTED_COMPOSITES for f in FEATURE_COLS],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df.insert(0, "rank", np.arange(1, len(importance_df) + 1))

    csv_path = os.path.join(OUT_DIR, f"shap_feature_importance_{safe_name}.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")

    print(f"\n  Top {TOP_N_REPORT} features by mean |SHAP| (class='{focus_label}'):")
    print("  " + importance_df.head(TOP_N_REPORT).to_string(index=False)
          .replace("\n", "\n  "))

    composite_ranks = importance_df[importance_df["is_composite_index"]]
    n_composite_in_top10 = int((composite_ranks["rank"] <= 10).sum())
    print(f"\n  {n_composite_in_top10}/4 composite indices are in the global "
          f"top 10 features by SHAP importance.")

    # ------------------------------------------------------------------
    # Plots (each wrapped so one failure doesn't kill the whole script)
    # ------------------------------------------------------------------
    top_feats = importance_df.head(TOP_N_REPORT)["feature"].tolist()
    top_idx = [FEATURE_COLS.index(f) for f in top_feats]

    try:
        plt.figure(figsize=(9, 8))
        plt.barh(top_feats[::-1], importance_df.head(TOP_N_REPORT)["mean_abs_shap"][::-1])
        plt.xlabel("Mean |SHAP value| (impact on model output)")
        plt.title(f"{target_col} -- Top {TOP_N_REPORT} features (class='{focus_label}')")
        plt.tight_layout()
        bar_path = os.path.join(OUT_DIR, f"shap_summary_bar_{safe_name}.png")
        plt.savefig(bar_path, dpi=150)
        plt.close()
        print(f"  Saved -> {bar_path}")
    except Exception as e:
        print(f"  [plot failed] summary bar: {e}")

    try:
        shap.summary_plot(shap_2d[:, top_idx], X_test.iloc[:, top_idx],
                           show=False, max_display=TOP_N_REPORT)
        beeswarm_path = os.path.join(OUT_DIR, f"shap_beeswarm_{safe_name}.png")
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=150)
        plt.close()
        print(f"  Saved -> {beeswarm_path}")
    except Exception as e:
        print(f"  [plot failed] beeswarm: {e}")

    for comp_col in WEIGHTED_COMPOSITES:
        try:
            comp_idx = FEATURE_COLS.index(comp_col)
            plt.figure()
            shap.dependence_plot(comp_idx, shap_2d, X_test, show=False,
                                  feature_names=FEATURE_COLS)
            dep_path = os.path.join(
                OUT_DIR, f"shap_dependence_{comp_col}_{safe_name}.png")
            plt.tight_layout()
            plt.savefig(dep_path, dpi=150)
            plt.close()
            print(f"  Saved -> {dep_path}")
        except Exception as e:
            print(f"  [plot failed] dependence plot for {comp_col}: {e}")

    # One high-risk and one low-risk real patient, for a concrete
    # "why did the model predict this?" example in the thesis.
    try:
        proba = model.predict_proba(X_test)[:, focus_idx]
        high_i = int(np.argmax(proba))
        low_i = int(np.argmin(proba))
        for label, i in [("highrisk", high_i), ("lowrisk", low_i)]:
            explanation = shap.Explanation(
                values=shap_2d[i], base_values=base_value,
                data=X_test.iloc[i].values, feature_names=FEATURE_COLS)
            plt.figure()
            shap.plots.waterfall(explanation, show=False, max_display=15)
            wf_path = os.path.join(OUT_DIR, f"shap_waterfall_{label}_{safe_name}.png")
            plt.tight_layout()
            plt.savefig(wf_path, dpi=150)
            plt.close()
            print(f"  Saved -> {wf_path} (predicted P({focus_label})={proba[i]:.3f})")
    except Exception as e:
        print(f"  [plot failed] waterfall examples: {e}")

    # Raw values saved for any later reuse without re-running the model.
    npz_path = os.path.join(OUT_DIR, f"shap_values_{safe_name}.npz")
    np.savez_compressed(npz_path, shap_values=shap_2d,
                         X_test=X_test.values, feature_names=np.array(FEATURE_COLS),
                         base_value=base_value)
    print(f"  Saved -> {npz_path}")

    # ------------------------------------------------------------------
    # Plain-language interpretation summary
    # ------------------------------------------------------------------
    top5_text = "\n".join(
        f"    {r.rank}. {r.feature}"
        f"{' [COMPOSITE INDEX]' if r.is_composite_index else ''}"
        f" (mean |SHAP| = {r.mean_abs_shap:.4f})"
        for r in importance_df.head(5).itertuples()
    )
    interp = (
        f"SHAP explainability summary -- {target_col} (focus class: "
        f"'{focus_label}')\n"
        f"{'=' * 70}\n"
        f"Model: Random Forest (class_weight=balanced, n_estimators=300)\n"
        f"Feature set: Raw + Composite (Arm B, {len(FEATURE_COLS)} features)\n"
        f"Train accuracy: {train_acc:.3f} | Held-out test accuracy: {test_acc:.3f}\n"
        f"SHAP values computed on the held-out 20% test set (unseen data).\n\n"
        f"Top 5 features driving predictions toward '{focus_label}':\n"
        f"{top5_text}\n\n"
        f"{n_composite_in_top10}/4 composite indices rank in the global top 10 "
        f"most important features (out of {len(FEATURE_COLS)} total).\n"
    )
    interp_path = os.path.join(OUT_DIR, f"shap_interpretation_{safe_name}.txt")
    with open(interp_path, "w", encoding="utf-8") as f:
        f.write(interp)
    print(f"  Saved -> {interp_path}")

print(f"\n\n{'=' * 90}\nDone. Step 9 complete.\n{'=' * 90}")
print("Review the shap_interpretation_<target>.txt files and the plots -- "
      "these are the core artifacts for the thesis's explainability chapter. "
      "Next: Step 10 -- write-up / clinical interpretation.")
