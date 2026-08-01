"""
22_paper_figures.py
=====================
Generates 5 consolidated, publication-quality figures for the paper, built
from data already saved in outputs/ by scripts 04/07/08/09/18 (no need to
reload the raw dataset). These are NEW figures, distinct from the existing
per-target SHAP/cluster plots (steps 9/18) -- each one puts all 3 targets (or
all 3 ablation arms, or all 4 composite indices) side by side in ONE image,
which is what a paper figure usually needs instead of 3 separate plots.

Figures produced (./outputs/paper_figures/):
  1. fig1_model_comparison.png       -- grouped bar: CV F1-macro, 5 models x 3 targets
  2. fig2_composite_boxplots.png     -- 2x2 boxplots, 4 weighted composites by PPD_binary
  3. fig3_ablation_comparison.png    -- grouped bar: Arm A/B/C avg F1-macro x 3 targets
  4. fig4_cluster_prevalence.png     -- bar chart: PPD_binary rate by unsupervised cluster
  5. fig5_shap_top10_consolidated.png-- 3-panel horizontal bar, top-10 SHAP features per
                                        target, composite indices highlighted in a
                                        different color from raw features

Usage
-----
    pip install matplotlib --break-system-packages   # already installed here
    python 22_paper_figures.py
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

TARGETS = ["EPDS_Result", "PHQ9_Result", "PPD_binary"]
TARGET_LABELS = {"EPDS_Result": "EPDS Result", "PHQ9_Result": "PHQ9 Result",
                  "PPD_binary": "PPD_binary"}
COLORS = {"Logistic Regression": "#4C72B0", "Random Forest": "#DD8452",
          "XGBoost": "#55A868", "LightGBM": "#C44E52", "CatBoost": "#8172B2"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ===========================================================================
# Figure 1 -- grouped bar: CV F1-macro, all 5 models x 3 targets
# ===========================================================================
def fig1_model_comparison():
    fig, ax = plt.subplots(figsize=(9, 5))
    models = list(COLORS.keys())
    n_targets = len(TARGETS)
    width = 0.15
    x = np.arange(n_targets)

    missing_models = set()
    for i, model in enumerate(models):
        vals = []
        for t in TARGETS:
            path = os.path.join(OUT_DIR, f"model_performance_{t}.csv")
            if not os.path.exists(path):
                vals.append(np.nan)
                continue
            df = pd.read_csv(path)
            row = df[df["model"] == model]
            if row.empty:
                vals.append(np.nan)
                missing_models.add(model)
            else:
                vals.append(row["cv_f1_macro_mean"].values[0])
        ax.bar(x + (i - len(models) / 2) * width + width / 2, vals, width,
               label=model, color=COLORS[model])

    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS])
    ax.set_ylabel("CV F1-macro (5-fold)")
    ax.set_title("Baseline Model Comparison Across All 3 Targets\n"
                  "(Arm B: raw + weighted composite features, untuned defaults)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_model_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}" + (f"  [note: not found in CSVs: {missing_models}]"
                                if missing_models else ""))


# ===========================================================================
# Figure 2 -- 2x2 boxplots: 4 weighted composite indices by PPD_binary
# ===========================================================================
def fig2_composite_boxplots():
    path = os.path.join(OUT_DIR, "PPD_dataset_with_composite_features.csv")
    if not os.path.exists(path):
        print(f"[skip fig2] {path} not found."); return
    df = pd.read_csv(path)
    cols = [
        ("Economic_Stability_Index_Weighted", "Economic Stability Index"),
        ("Social_Support_Index_Weighted", "Social/Family Support Index"),
        ("Maternal_MentalHealth_Risk_Index_Weighted", "Maternal Mental-Health Risk Index"),
        ("Neonatal_Delivery_Stress_Index_Weighted", "Neonatal/Delivery Stress Index"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (col, label) in zip(axes.flat, cols):
        data0 = df.loc[df["PPD_binary"] == 0, col].dropna()
        data1 = df.loc[df["PPD_binary"] == 1, col].dropna()
        bp = ax.boxplot([data0, data1], patch_artist=True, widths=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["No PPD", "PPD"])
        for patch, color in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Index score (0-10)")
    fig.suptitle("Composite Risk Indices by PPD Status (weighted variants)", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_composite_boxplots.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ===========================================================================
# Figure 3 -- grouped bar: Arm A/B/C avg F1-macro across models, x 3 targets
# ===========================================================================
def fig3_ablation_comparison():
    arm_labels = {"A_Raw_only": "Arm A: Raw only",
                  "B_Raw_plus_Composite": "Arm B: Raw + Composite",
                  "C_Selected_plus_Composite": "Arm C: Selected + Composite"}
    arm_colors = {"A_Raw_only": "#8C8C8C", "B_Raw_plus_Composite": "#4C72B0",
                  "C_Selected_plus_Composite": "#55A868"}
    fig, ax = plt.subplots(figsize=(8, 5))
    n_targets = len(TARGETS)
    width = 0.22
    x = np.arange(n_targets)
    for i, arm in enumerate(arm_labels):
        vals = []
        for t in TARGETS:
            path = os.path.join(OUT_DIR, f"ablation_results_{t}.csv")
            if not os.path.exists(path):
                vals.append(np.nan); continue
            df = pd.read_csv(path)
            sub = df[df["arm"] == arm]
            vals.append(sub["cv_f1_macro_mean"].mean() if not sub.empty else np.nan)
        bars = ax.bar(x + (i - 1) * width, vals, width, label=arm_labels[arm],
                       color=arm_colors[arm])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS])
    ax.set_ylabel("Mean CV F1-macro across 5 model families")
    ax.set_title("Ablation Study: Raw vs Raw+Composite vs Selected+Composite")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_ablation_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


# ===========================================================================
# Figure 4 -- PPD prevalence by unsupervised cluster
# ===========================================================================
def fig4_cluster_prevalence():
    path = os.path.join(OUT_DIR, "cluster_ppd_rates.csv")
    if not os.path.exists(path):
        print(f"[skip fig4] {path} not found."); return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#C44E52" if r > df["PPD_binary_rate"].median() else "#55A868"
              for r in df["PPD_binary_rate"]]
    bars = ax.bar(df["Cluster"].astype(str), df["PPD_binary_rate"] * 100, color=colors)
    for b, (_, row) in zip(bars, df.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{row['PPD_binary_rate']*100:.1f}%\n(n={row['n_mothers']})",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Cluster (K-Means on 4 composite indices + Age + pregnancy count, "
                  "PPD label never used)")
    ax.set_ylabel("PPD_binary prevalence (%)")
    ax.set_title("Unsupervised Cluster Confirmation: PPD Prevalence by Risk Profile\n"
                  "(chi-square p=3.73e-28)")
    ax.set_ylim(0, max(df["PPD_binary_rate"] * 100) + 15)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_cluster_prevalence.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out}")


# ===========================================================================
# Figure 5 -- consolidated top-10 SHAP features, 3 targets side by side,
#             composite indices highlighted
# ===========================================================================
def fig5_shap_consolidated():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, t in zip(axes, TARGETS):
        path = os.path.join(OUT_DIR, f"shap_feature_importance_{t}.csv")
        if not os.path.exists(path):
            ax.set_visible(False); continue
        df = pd.read_csv(path).head(10).iloc[::-1]  # reverse for horizontal bar top-to-bottom
        colors = ["#C44E52" if c else "#4C72B0" for c in df["is_composite_index"]]
        ax.barh(df["feature"], df["mean_abs_shap"], color=colors)
        ax.set_title(TARGET_LABELS[t], fontsize=11)
        ax.set_xlabel("mean |SHAP value|")
        ax.tick_params(axis="y", labelsize=8)
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="#C44E52", label="Composite index"),
                    Patch(facecolor="#4C72B0", label="Raw feature")]
    fig.legend(handles=legend_elems, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.04), fontsize=10)
    fig.suptitle("Top-10 SHAP Feature Importance, All 3 Targets (Random Forest, Arm B)",
                 y=1.1, fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig5_shap_top10_consolidated.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    fig1_model_comparison()
    fig2_composite_boxplots()
    fig3_ablation_comparison()
    fig4_cluster_prevalence()
    fig5_shap_consolidated()
    print(f"\nDone. All figures saved to: {FIG_DIR}")
