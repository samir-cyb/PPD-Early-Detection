"""
make_pipeline_steps.py
======================
Generates a clear 9-step methodology pipeline diagram for Chapter 3.

Output: Images/fig_pipeline_steps.png

Run from the paper/ folder:
    python make_pipeline_steps.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.join(THIS_DIR, "Images")
os.makedirs(IMG_DIR, exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────────────
PHASE_COLORS = {
    "data":   "#2E86AB",   # blue  – data preparation steps
    "model":  "#A23B72",   # purple – modelling steps
    "eval":   "#F18F01",   # orange – evaluation / analysis steps
}
BORDER   = "#1A1A2E"
ARROW_C  = "#1A1A2E"
BG       = "white"
TEXT_W   = "white"
TEXT_D   = "#F0F0F0"   # slightly dim for description text

# ── 9 steps ─────────────────────────────────────────────────────────────────
steps = [
    # (phase, step_number, short_title, description)
    ("data",  1,
     "Leakage-Free Feature Design",
     "Remove all 23 EPDS / PHQ-9 columns before any processing.\n"
     "108 safe input columns remain (demographic, economic, pregnancy, etc.).\n"
     "Three prediction targets defined: PPD_binary, EPDS Result, PHQ9 Result."),

    ("data",  2,
     "Composite Risk Index Engineering",
     "Group related raw features into 4 clinical domains.\n"
     "Apply chi-square weighting to each sub-feature within a domain.\n"
     "Produce four 0–10 risk scores: ESI · SSI · MHRI · NSI."),

    ("data",  3,
     "Feature Encoding & Dataset Assembly",
     "One-hot encode all 108 categorical raw features → 112 binary columns.\n"
     "Assemble three feature sets: Arm A (raw only), Arm B (raw + composite),\n"
     "Arm C (top-30 selected + composite via nested cross-validation)."),

    ("model", 4,
     "Baseline Model Training",
     "Train 5 model families on each arm × each target (45 experiments):\n"
     "Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost.\n"
     "80/20 stratified split · 5-fold CV · class_weight = balanced."),

    ("model", 5,
     "Advanced Modeling",
     "Hyperparameter tuning via RandomizedSearchCV (20 iter, 5 fold).\n"
     "Multi-layer Perceptron (3 architectures tested).\n"
     "Stacking Ensemble: 5 base models + Logistic Regression meta-learner."),

    ("model", 6,
     "Three-Arm Ablation Study",
     "Compare Arm A vs Arm B vs Arm C using paired t-test and Wilcoxon test.\n"
     "Repeated 5×5 cross-validation for composite effect significance.\n"
     "McNemar test: RF vs Stacking (p = 0.228 — not significant)."),

    ("eval",  7,
     "Multi-Angle Explainability (SHAP)",
     "Global importance: SHAP beeswarm for all three targets.\n"
     "Arm A vs Arm B comparison: SHAP shift from raw to composite features.\n"
     "Pairwise interaction heatmap · Single-patient waterfall explanation."),

    ("eval",  8,
     "Unsupervised Cluster Validation",
     "K-Means (K = 3) on 6-feature composite space — no PPD label used.\n"
     "Silhouette score selects K = 3.\n"
     "Chi-square test on cluster × PPD_binary: p = 3.73 × 10⁻²⁸."),

    ("eval",  9,
     "Robustness & Sensitivity Experiments",
     "SMOTE vs class-weight · MICE vs mode-fill imputation.\n"
     "Chi-square vs LR-learned composite weights.\n"
     "Threshold calibration · Repeated CV · Cronbach α / VIF / McNemar."),
]

# ── layout constants ─────────────────────────────────────────────────────────
BOX_W   = 9.8
BOX_H   = 1.62
GAP     = 0.38
FIG_W   = BOX_W + 2.6
FIG_H   = len(steps) * (BOX_H + GAP) + 1.4
LEFT    = (FIG_W - BOX_W) / 2   # x position of box left edge

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── phase legend labels & y tracking ────────────────────────────────────────
phase_label = {"data": None, "model": None, "eval": None}
phase_names = {
    "data":  "Data Preparation",
    "model": "Modelling",
    "eval":  "Evaluation & Analysis",
}

# ── draw boxes top → bottom ──────────────────────────────────────────────────
y = FIG_H - 0.7 - BOX_H
box_bottoms = []

for (phase, num, title, desc) in steps:
    color = PHASE_COLORS[phase]

    # outer rounded box
    rect = FancyBboxPatch(
        (LEFT, y), BOX_W, BOX_H,
        boxstyle="round,pad=0.10,rounding_size=0.18",
        facecolor=color, edgecolor=BORDER, linewidth=1.6, alpha=0.92,
    )
    ax.add_patch(rect)

    # left accent strip (slightly darker)
    strip = FancyBboxPatch(
        (LEFT, y), 0.45, BOX_H,
        boxstyle="round,pad=0.0",
        facecolor="#00000030", edgecolor="none",
    )
    ax.add_patch(strip)

    # step number circle
    cx = LEFT + 0.225
    cy = y + BOX_H / 2
    circle = plt.Circle((cx, cy), 0.19, color="white", zorder=3)
    ax.add_patch(circle)
    ax.text(cx, cy, str(num),
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=color, zorder=4)

    # title
    ax.text(LEFT + 0.65, y + BOX_H - 0.38, title,
            ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=TEXT_W)

    # description
    ax.text(LEFT + 0.65, y + 0.55, desc,
            ha="left", va="center",
            fontsize=8.8, color=TEXT_D, linespacing=1.45)

    # phase bracket — track first/last y for each phase
    if phase_label[phase] is None:
        phase_label[phase] = [y + BOX_H, y]
    else:
        phase_label[phase][1] = y   # extend to current bottom

    box_bottoms.append(y)
    y -= (BOX_H + GAP)

# ── downward arrows between consecutive boxes ─────────────────────────────
for i in range(len(steps) - 1):
    y_from = box_bottoms[i]
    y_to   = box_bottoms[i + 1] + BOX_H
    cx     = LEFT + BOX_W / 2
    arrow = FancyArrowPatch(
        (cx, y_from - 0.01), (cx, y_to + 0.01),
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=1.8,
        color=ARROW_C,
        zorder=5,
    )
    ax.add_patch(arrow)

# ── leakage boundary dashed line (between step 1 and step 2) ────────────────
boundary_y = (box_bottoms[0] + box_bottoms[1] + BOX_H) / 2
ax.plot([LEFT - 0.3, LEFT + BOX_W + 0.3], [boundary_y, boundary_y],
        color="#E74C3C", linewidth=1.6, linestyle="--", zorder=6)
ax.text(LEFT + BOX_W + 0.35, boundary_y, "Leakage\nboundary",
        va="center", ha="left", fontsize=7.8,
        color="#E74C3C", fontweight="bold")

# ── phase brackets on the right ─────────────────────────────────────────────
bracket_x = LEFT + BOX_W + 0.18
for phase, (y_top, y_bot) in phase_label.items():
    color = PHASE_COLORS[phase]
    mid_y = (y_top + y_bot) / 2
    # vertical line
    ax.plot([bracket_x, bracket_x], [y_bot, y_top],
            color=color, linewidth=3, solid_capstyle="round")
    # horizontal ticks
    ax.plot([bracket_x, bracket_x + 0.12], [y_top, y_top],
            color=color, linewidth=2)
    ax.plot([bracket_x, bracket_x + 0.12], [y_bot, y_bot],
            color=color, linewidth=2)
    ax.text(bracket_x + 0.18, mid_y,
            phase_names[phase],
            va="center", ha="left", fontsize=8.0,
            color=color, fontweight="bold", rotation=0)

# ── legend ───────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=PHASE_COLORS["data"],  edgecolor=BORDER,
                   label="Data Preparation (Steps 1–3)"),
    mpatches.Patch(facecolor=PHASE_COLORS["model"], edgecolor=BORDER,
                   label="Modelling (Steps 4–6)"),
    mpatches.Patch(facecolor=PHASE_COLORS["eval"],  edgecolor=BORDER,
                   label="Evaluation & Analysis (Steps 7–9)"),
]
ax.legend(handles=legend_patches,
          loc="lower center",
          bbox_to_anchor=(0.45, -0.002),
          ncol=3, fontsize=8.5, framealpha=0.9,
          edgecolor=BORDER)

# ── title ────────────────────────────────────────────────────────────────────
ax.set_title(
    "Leakage-Free PPD Prediction Framework — Nine-Step Methodology Pipeline",
    fontsize=13, fontweight="bold", pad=8, color=BORDER,
)

fig.tight_layout(pad=0.5)
out = os.path.join(IMG_DIR, "fig_pipeline_steps.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved → {out}")
