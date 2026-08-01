"""
23_additional_paper_figures.py
================================
Script 22 covered Step 7 (baseline models), Step 8 (ablation), Step 9 (SHAP),
and Step 18 (clusters). This script covers the parts of the pipeline that
were still missing a consolidated figure:

  Fig6  -- Step 7b: baseline vs TUNED vs Stacking Ensemble vs MLP, all in one
           ranked bar chart, per target (answers "did tuning/DL/ensembling
           actually help?").
  Fig7  -- Step 10 (Tier 1): ordinal-aware comparison -- Direct Classifier vs
           best Regression+Threshold, on Accuracy / F1-macro / QWK, for EPDS
           Result and PHQ9 Result side by side (the PHQ9 approach-change is
           one of the project's Novelties -- deserves its own figure).
  Fig8  -- Step 11 (Tier 1): PPD_binary decision-threshold comparison
           (Default 0.5 vs Max-F1 0.407 vs Screening 0.303) on
           Precision/Recall/F1.
  Fig9  -- Script 20 (Tier 2 closing-loop): baseline vs +interactions vs
           +cluster vs +both, F1-macro, all 3 targets -- the honest "nothing
           beat baseline" negative result.
  Fig10 -- Script 21 (this project's own new test): RF vs Stacking Ensemble
           holdout ROC-AUC with the bootstrap 95% CI as an error bar, so the
           "not statistically different" finding is visible, not just a
           number in a table.
  Fig11 -- Step 19 error analysis: mean composite-index values, FN vs TP and
           FP vs TN, PPD_binary -- shows the model's blind spot is driven by
           the same 2 composites in both directions.

Deliberately NOT turned into separate figures here (kept as CSV/table-only,
since they are secondary robustness checks rather than headline results):
CatBoost native-vs-onehot (12), SMOTE/imputation (14), composite
weight-learning (13), Age^2/group-importance (15). Say the word and a
script 24 can chart these too.

Usage
-----
    python 23_additional_paper_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

TARGETS = ["EPDS_Result", "PHQ9_Result", "PPD_binary"]
TARGET_LABELS = {"EPDS_Result": "EPDS Result", "PHQ9_Result": "PHQ9 Result",
                  "PPD_binary": "PPD_binary"}


# ===========================================================================
# Fig6 -- Step 7b: baseline vs tuned vs Stacking vs MLP, ranked, per target
# ===========================================================================
def fig6_advanced_modeling():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, t in zip(axes, TARGETS):
        path = os.path.join(OUT_DIR, f"model_performance_advanced_{t}.csv")
        if not os.path.exists(path):
            ax.set_visible(False)
            print(f"[skip fig6/{t}] {path} not found.")
            continue
        df = pd.read_csv(path).sort_values("cv_f1_macro_mean", ascending=True)

        def color_for(name):
            if "Stacking" in name:
                return "#C44E52"
            if "tuned" in name and "baseline" not in name:
                return "#DD8452"
            if "MLP" in name:
                return "#8172B2"
            return "#4C72B0"  # baseline, untuned

        colors = [color_for(m) for m in df["model"]]
        ax.barh(df["model"], df["cv_f1_macro_mean"], color=colors)
        ax.set_title(TARGET_LABELS[t], fontsize=11)
        ax.set_xlabel("CV F1-macro")
        ax.tick_params(axis="y", labelsize=8)
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="#4C72B0", label="Baseline (untuned)"),
                     Patch(facecolor="#DD8452", label="Tuned (RandomizedSearchCV)"),
                     Patch(facecolor="#C44E52", label="Stacking Ensemble"),
                     Patch(facecolor="#8172B2", label="MLP (deep learning)")]
    fig.legend(handles=legend_elems, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.suptitle("Does Tuning / Deep Learning / Stacking Beat the Baseline?",
                 y=1.13, fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig6_advanced_modeling_comparison.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ===========================================================================
# Fig7 -- Step 10: Direct Classifier vs best Regression+Threshold, EPDS & PHQ9
# ===========================================================================
def fig7_ordinal_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    metrics = ["accuracy", "f1_macro", "qwk"]
    metric_labels = ["Accuracy", "F1-macro", "QWK"]
    for ax, t, tlabel in zip(axes, ["EPDS_Result", "PHQ9_Result"],
                              ["EPDS Result", "PHQ9 Result"]):
        path = os.path.join(OUT_DIR, f"ordinal_regression_results_{t}.csv")
        if not os.path.exists(path):
            ax.set_visible(False)
            print(f"[skip fig7/{t}] {path} not found.")
            continue
        df = pd.read_csv(path)
        direct = df[df["approach"] == "Direct Classifier (Random Forest)"].iloc[0]
        # best regression = highest QWK among the "Regression + Threshold" rows
        reg_rows = df[df["approach"].str.startswith("Regression")]
        best_reg = reg_rows.loc[reg_rows["qwk"].idxmax()]

        x = np.arange(len(metrics))
        width = 0.35
        direct_vals = [direct[m] for m in metrics]
        reg_vals = [best_reg[m] for m in metrics]
        ax.bar(x - width / 2, direct_vals, width, label="Direct Classifier (RF)",
               color="#4C72B0")
        ax.bar(x + width / 2, reg_vals, width,
               label=f"Regression+Threshold\n({best_reg['approach'].split('(')[-1][:-1]})",
               color="#C44E52")
        for xi, v in zip(x - width / 2, direct_vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        for xi, v in zip(x + width / 2, reg_vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_title(tlabel)
        ax.set_ylim(0, 0.75)
        ax.legend(fontsize=8)
    fig.suptitle("Ordinal-Aware Modeling: Direct Classifier vs Regression+Threshold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig7_ordinal_regression_comparison.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ===========================================================================
# Fig8 -- Step 11: PPD_binary threshold operating points
# ===========================================================================
def fig8_threshold_comparison():
    path = os.path.join(OUT_DIR, "threshold_calibration_results.csv")
    if not os.path.exists(path):
        print(f"[skip fig8] {path} not found."); return
    df = pd.read_csv(path)
    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                       label=f"{row['operating_point']} (thr={row['threshold']:.3f})",
                       color=colors[i % len(colors)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("PPD_binary Decision-Threshold Comparison (Random Forest)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig8_threshold_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


# ===========================================================================
# Fig9 -- Script 20: closing-the-loop feature augmentation test
# ===========================================================================
def fig9_closing_loop():
    path = os.path.join(OUT_DIR, "feature_augmentation_results.csv")
    if not os.path.exists(path):
        print(f"[skip fig9] {path} not found."); return
    df = pd.read_csv(path)
    configs = ["baseline", "plus_interactions", "plus_cluster", "plus_both"]
    config_labels = ["Baseline", "+Interactions", "+Cluster", "+Both"]
    targets = df["target"].unique().tolist()
    x = np.arange(len(targets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, cfg in enumerate(configs):
        vals = []
        errs = []
        for t in targets:
            row = df[(df["target"] == t) & (df["config"] == cfg)]
            if row.empty:
                vals.append(np.nan); errs.append(0); continue
            vals.append(row["mean_f1_macro"].values[0])
            errs.append(row["std_f1_macro"].values[0])
        bars = ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=3,
                       label=config_labels[i], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Mean CV F1-macro (+/- std across folds)")
    ax.set_title("Script 20 Closing-Loop Test: No Configuration Beats Baseline\n"
                  "(every delta is smaller than its own fold-to-fold std)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig9_closing_loop_test.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


# ===========================================================================
# Fig10 -- Script 21: RF vs Stacking, bootstrap CI on AUC difference
# ===========================================================================
def fig10_rf_vs_stacking():
    path = os.path.join(OUT_DIR, "rf_vs_stacking_significance_PPD_binary.csv")
    if not os.path.exists(path):
        print(f"[skip fig10] {path} not found -- run 21_reliability_and_extra_tests.py first.")
        return
    df = pd.read_csv(path).iloc[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    aucs = [df["rf_holdout_auc"], df["stacking_holdout_auc"]]
    labels = ["Random Forest\n(adopted, explainable)", "Stacking Ensemble\n(best raw predictor)"]
    bars = ax.bar(labels, aucs, color=["#4C72B0", "#C44E52"], width=0.5)
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}",
                ha="center", fontsize=10)
    ax.set_ylabel("Holdout ROC-AUC")
    ax.set_ylim(0.7, 0.9)
    diff = df["auc_diff_rf_minus_stack"]
    ci_lo, ci_hi = df["bootstrap_auc_diff_ci_low"], df["bootstrap_auc_diff_ci_high"]
    mcnemar_p, boot_p = df["mcnemar_p"], df["bootstrap_p"]
    ax.set_title("RF vs Stacking Ensemble (PPD_binary)\n"
                 f"AUC diff={diff:+.4f}, bootstrap 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]\n"
                 f"McNemar p={mcnemar_p:.3f}, bootstrap p={boot_p:.3f} (both n.s.)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig10_rf_vs_stacking.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


# ===========================================================================
# Fig11 -- Step 19 error analysis: composite indices, FN vs TP / FP vs TN
# ===========================================================================
def fig11_error_analysis():
    fn_tp_path = os.path.join(OUT_DIR, "error_analysis_PPD_binary_FN_vs_TP.csv")
    fp_tn_path = os.path.join(OUT_DIR, "error_analysis_PPD_binary_FP_vs_TN.csv")
    if not (os.path.exists(fn_tp_path) and os.path.exists(fp_tn_path)):
        print("[skip fig11] error analysis CSVs not found."); return
    fn_tp = pd.read_csv(fn_tp_path)
    fp_tn = pd.read_csv(fp_tn_path)
    composite_names = [
        "Maternal_MentalHealth_Risk_Index_Weighted",
        "Social_Support_Index_Weighted",
        "Economic_Stability_Index_Weighted",
        "Neonatal_Delivery_Stress_Index_Weighted",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def plot_panel(ax, df, col_a, col_b, label_a, label_b, title):
        df = df[df["feature"].isin(composite_names)].set_index("feature").reindex(composite_names)
        x = np.arange(len(composite_names))
        width = 0.35
        bars_a = ax.bar(x - width / 2, df[col_a], width, label=label_a, color="#4C72B0")
        bars_b = ax.bar(x + width / 2, df[col_b], width, label=label_b, color="#C44E52")
        for xi, (_, row) in zip(x, df.iterrows()):
            star = "*" if row["significant_at_0.05"] else ""
            ax.text(xi, max(row[col_a], row[col_b]) + 0.2, star,
                    ha="center", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_Weighted", "").replace("_", "\n") for n in composite_names],
                            fontsize=7)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylabel("Mean composite index value")

    plot_panel(axes[0], fn_tp, "mean_FN", "mean_TP", "Missed cases (FN)",
               "Caught cases (TP)", "Among true PPD cases:\nwhat separates missed from caught?")
    plot_panel(axes[1], fp_tn, "mean_FP", "mean_TN", "False alarms (FP)",
               "Correctly cleared (TN)", "Among true non-PPD cases:\nwhat separates false alarms?")
    fig.suptitle("Error Analysis: Composite Index Profiles of Model Mistakes (PPD_binary)\n"
                 "(* = Mann-Whitney p<0.05)", y=1.05)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig11_error_analysis.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    fig6_advanced_modeling()
    fig7_ordinal_comparison()
    fig8_threshold_comparison()
    fig9_closing_loop()
    fig10_rf_vs_stacking()
    fig11_error_analysis()
    print(f"\nDone. All figures saved to: {FIG_DIR}")
