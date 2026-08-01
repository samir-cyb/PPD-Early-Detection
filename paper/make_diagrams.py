"""
make_diagrams.py
Generates two thesis diagrams and saves them to Images/
Run from the paper/ directory:
    python make_diagrams.py

Outputs:
    Images/fig_data_processing.jpg   -- detailed data processing flowchart
    Images/fig_system_architecture.jpg -- high-level system architecture
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

os.makedirs("Images", exist_ok=True)

# ===========================================================
# DIAGRAM 1 — Data Processing Flowchart
# ===========================================================
fig1, ax1 = plt.subplots(figsize=(11, 16))
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 18)
ax1.axis("off")
ax1.set_facecolor("white")
fig1.patch.set_facecolor("white")

# colour palette
C_INPUT  = "#D6E4F0"   # light blue  — data inputs
C_PROC   = "#D5F5E3"   # light green — processing steps
C_FEAT   = "#FDEBD0"   # light orange — feature engineering
C_OUTPUT = "#E8DAEF"   # light purple — outputs / arms
C_WARN   = "#FADBD8"   # light red    — leakage exclusion
BORDER   = "#2C3E50"

def box(ax, x, y, w, h, text, color, fontsize=9.5, bold=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=BORDER, linewidth=1.3)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, wrap=True,
            multialignment="center", color="#1A1A1A")

def arrow(ax, x, y_top, y_bot, label=""):
    ax.annotate("", xy=(x, y_bot + 0.05),
                xytext=(x, y_top - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.4))
    if label:
        ax.text(x + 0.18, (y_top + y_bot) / 2, label,
                fontsize=7.5, color="#555555", va="center")

def split_arrow(ax, x_from, y_from, x_to, y_to):
    ax.annotate("", xy=(x_to, y_to + 0.05),
                xytext=(x_from, y_from - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=BORDER,
                                lw=1.2, connectionstyle="arc3,rad=0.0"))

# ---- Step blocks (top to bottom) ----
steps = [
    # (x_centre, y_centre, width, height, label, color, bold)
    (5.0, 17.2, 7.5, 0.85,
     "Raw Survey CSV   (n = 800 Bangladeshi mothers, 151 raw columns)",
     C_INPUT, True),

    (5.0, 15.8, 7.5, 0.85,
     "Step 1 — Leakage Exclusion\n"
     "Remove 23 EPDS items, 23 PHQ-9 items, and 4 score/result columns\n"
     "→ 108 raw feature columns remain",
     C_WARN, False),

    (5.0, 14.3, 7.5, 0.85,
     "Step 2 — Data Cleaning\n"
     "Drop 1 malformed row  |  Mode-fill 5 randomly-missing columns\n"
     "Final dataset: n = 800, no missing values",
     C_PROC, False),

    (5.0, 12.8, 7.5, 0.85,
     "Step 3 — One-Hot Encoding\n"
     "Encode all 108 categorical raw features → 112 binary dummy columns\n"
     "Direction check: confirm high-risk direction for each dummy",
     C_PROC, False),

    (5.0, 11.3, 7.5, 0.85,
     "Step 4 — Composite Index Engineering\n"
     "Group related features into 4 domains: ESI, SSI, MHRI, NSI\n"
     "Chi-square weight each sub-feature  →  score ∈ [0, 10] per domain\n"
     "Validate: ANOVA + Kruskal-Wallis  (24/24 tests p < 0.05)",
     C_FEAT, False),

    (5.0, 9.65, 7.5, 0.85,
     "Step 5 — Train / Test Split\n"
     "80 % training (640 mothers)  |  20 % held-out test (160 mothers)\n"
     "Stratified by PPD_binary label  —  split done once before any training",
     C_PROC, False),
]

for (x, y, w, h, txt, col, bld) in steps:
    box(ax1, x, y, w, h, txt, col, fontsize=8.8, bold=bld)

# arrows between main steps
ys = [s[1] for s in steps]
hs = [s[3] for s in steps]
for i in range(len(steps) - 1):
    y_top = ys[i]   - hs[i]   / 2
    y_bot = ys[i+1] + hs[i+1] / 2
    arrow(ax1, 5.0, y_top, y_bot)

# ---- Three-arm split ----
split_y = 9.65 - 0.85/2   # bottom of step 5
ax1.plot([5.0, 5.0], [split_y - 0.05, split_y - 0.55],
         color=BORDER, lw=1.4)
ax1.plot([2.0, 8.0], [split_y - 0.55, split_y - 0.55],
         color=BORDER, lw=1.4)

arm_y_top = split_y - 0.55
arm_box_y = 7.9

for x_arm, arm_label in [(2.0, "Arm A\n(108 raw features)\n108 columns"),
                          (5.0, "Arm B\n(raw + 4 composite)\n112 columns\n[Adopted]"),
                          (8.0, "Arm C\n(top-30 selected\n+ 4 composite)\n34 columns")]:
    ax1.annotate("", xy=(x_arm, arm_box_y + 0.43),
                 xytext=(x_arm, arm_y_top),
                 arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.3))
    col = C_OUTPUT if "Adopted" not in arm_label else "#D1F2EB"
    box(ax1, x_arm, arm_box_y, 2.9, 0.85, arm_label, col, fontsize=8.5)

# ---- Cross-validation ----
cv_y = 7.9 - 0.85/2
ax1.annotate("", xy=(5.0, 6.85 + 0.43),
             xytext=(5.0, cv_y),
             arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.4))
box(ax1, 5.0, 6.85, 7.5, 0.85,
    "Step 6 — 5-Fold Stratified Cross-Validation (on training set)\n"
    "5 model families × 3 targets × 3 arms = 45 CV experiments\n"
    "Metrics: F1-macro, ROC-AUC, QWK (PHQ9 only)",
    C_PROC, fontsize=8.8)

# ---- Model selection ----
arrow(ax1, 5.0, 6.85 - 0.43, 5.55)
box(ax1, 5.0, 5.55, 7.5, 0.85,
    "Step 7 — Model Selection & Threshold Calibration\n"
    "RF (Arm B) adopted for PPD_binary + EPDS Result\n"
    "LinReg+Threshold adopted for PHQ9 Result\n"
    "Max-F1 threshold = 0.407  |  Platt scaling calibration",
    "#D1F2EB", fontsize=8.8)

# ---- Explainability & Cluster ----
arrow(ax1, 5.0, 5.55 - 0.43, 4.25)
box(ax1, 5.0, 4.25, 7.5, 0.85,
    "Step 8 — Explainability & Cluster Validation\n"
    "SHAP TreeExplainer → beeswarm, interaction heatmap, Arm A vs B comparison\n"
    "K-Means (K=3) on composite space → chi-square p = 3.73 × 10⁻²⁸",
    C_FEAT, fontsize=8.8)

# ---- Outputs ----
arrow(ax1, 5.0, 4.25 - 0.43, 2.95)
box(ax1, 5.0, 2.95, 7.5, 0.85,
    "Final Outputs\n"
    "PPD_binary: AUC 0.826  |  F1 0.736  |  Threshold 0.407\n"
    "EPDS Result: F1 0.520        PHQ9 Result: F1 0.426  QWK 0.602",
    C_OUTPUT, fontsize=8.8, bold=True)

# ---- Legend ----
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,  edgecolor=BORDER, label="Data input"),
    mpatches.Patch(facecolor=C_WARN,   edgecolor=BORDER, label="Leakage exclusion"),
    mpatches.Patch(facecolor=C_PROC,   edgecolor=BORDER, label="Processing step"),
    mpatches.Patch(facecolor=C_FEAT,   edgecolor=BORDER, label="Feature engineering / Explainability"),
    mpatches.Patch(facecolor=C_OUTPUT, edgecolor=BORDER, label="Output / Arm"),
    mpatches.Patch(facecolor="#D1F2EB",edgecolor=BORDER, label="Adopted model"),
]
ax1.legend(handles=legend_items, loc="lower left",
           fontsize=8, framealpha=0.9, bbox_to_anchor=(0.01, 0.01))

ax1.set_title("Data Processing Pipeline\n(Leakage-Free PPD Prediction Framework)",
              fontsize=13, fontweight="bold", pad=6, color=BORDER)

fig1.tight_layout()
fig1.savefig("Images/fig_data_processing.jpg", dpi=180,
             bbox_inches="tight", facecolor="white")
print("Saved: Images/fig_data_processing.jpg")
plt.close(fig1)


# ===========================================================
# DIAGRAM 2 — System Architecture
# ===========================================================
fig2, ax2 = plt.subplots(figsize=(14, 8))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 8)
ax2.axis("off")
ax2.set_facecolor("white")
fig2.patch.set_facecolor("white")

# ---- Colour scheme ----
CA = "#D6E4F0"   # input layer
CB = "#D5F5E3"   # preprocessing
CC = "#FDEBD0"   # feature engineering
CD = "#EAF0FB"   # model core
CE = "#E8DAEF"   # post-processing
CF = "#FADBD8"   # clinical output

def rbox(ax, x, y, w, h, title, body, color, title_size=9, body_size=8):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=BORDER, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.22, title,
            ha="center", va="top", fontsize=title_size,
            fontweight="bold", color="#1A1A1A")
    ax.text(x + w/2, y + h - 0.52, body,
            ha="center", va="top", fontsize=body_size,
            color="#333333", multialignment="center")

def harrow(ax, x1, x2, y, label=""):
    ax.annotate("", xy=(x2, y),
                xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.5))
    if label:
        ax.text((x1 + x2) / 2, y + 0.12, label,
                ha="center", fontsize=7.5, color="#555555")

# ---- Layer 1: Input ----
rbox(ax2, 0.3, 2.8, 2.1, 2.4,
     "Data Input",
     "800 Bangladeshi\nmothers\n\n151 raw survey\ncolumns\n(CSV)",
     CA)

# ---- Layer 2: Preprocessing ----
rbox(ax2, 3.0, 4.6, 2.4, 0.9,
     "Leakage Guard",
     "Remove 23 EPDS\n+ 23 PHQ-9 columns",
     "#FADBD8", title_size=8.5)

rbox(ax2, 3.0, 3.5, 2.4, 0.9,
     "Data Cleaning",
     "Drop malformed rows\nMode-fill missing",
     CB, title_size=8.5)

rbox(ax2, 3.0, 2.4, 2.4, 0.9,
     "Encoding",
     "One-hot encode\n→ 112 binary columns",
     CB, title_size=8.5)

# preprocessing bracket
ax2.add_patch(FancyBboxPatch((2.85, 2.25), 2.7, 3.4,
              boxstyle="round,pad=0.08",
              facecolor="none", edgecolor="#7D9EC0",
              linewidth=1.2, linestyle="--"))
ax2.text(4.2, 5.75, "Pre-\nprocessing\nModule",
         ha="center", fontsize=7.5, color="#2471A3", fontweight="bold")

# ---- Layer 3: Feature Engineering ----
rbox(ax2, 6.1, 4.5, 2.3, 1.05,
     "Composite Index\nEngineering",
     "Chi-square weights\nESI · SSI · MHRI · NSI\n→ 4 domain scores",
     CC, title_size=8.5)

rbox(ax2, 6.1, 3.1, 2.3, 1.1,
     "Feature Assembly\n(Arm B)",
     "112 one-hot columns\n+ 4 composite indices\n= 116 features",
     CC, title_size=8.5)

ax2.add_patch(FancyBboxPatch((5.95, 2.9), 2.6, 2.8,
              boxstyle="round,pad=0.08",
              facecolor="none", edgecolor="#CA8A04",
              linewidth=1.2, linestyle="--"))
ax2.text(7.25, 5.82, "Feature\nEngineering\nModule",
         ha="center", fontsize=7.5, color="#92400E", fontweight="bold")

# ---- Layer 4: Model Core ----
rbox(ax2, 9.1, 3.0, 2.4, 2.5,
     "Model Module",
     "Random Forest\n(n_estimators=300)\nclass_weight=balanced\n\n5-fold stratified CV\nHoldout evaluation\n\nLinReg+Threshold\n(PHQ9 only)",
     CD)
ax2.text(10.3, 5.62, "Model Module",
         ha="center", fontsize=7.5, color="#1A237E", fontweight="bold")

# ---- Layer 5: Post-processing ----
rbox(ax2, 11.9, 4.2, 1.9, 1.3,
     "Threshold\nCalibration",
     "Platt scaling\nMax-F1 @ 0.407\nScreening @ 0.303",
     CE, title_size=8.5)

rbox(ax2, 11.9, 2.6, 1.9, 1.4,
     "Explainability\nModule",
     "SHAP TreeExplainer\nBeeswarm plot\nInteraction heatmap\nWaterfall (single pt.)",
     CE, title_size=8.5)

ax2.add_patch(FancyBboxPatch((11.75, 2.4), 2.1, 3.25,
              boxstyle="round,pad=0.08",
              facecolor="none", edgecolor="#7D3C98",
              linewidth=1.2, linestyle="--"))
ax2.text(12.85, 5.77, "Post-\nprocessing\nModule",
         ha="center", fontsize=7.5, color="#6C3483", fontweight="bold")

# ---- Layer 6: Clinical Output ----
rbox(ax2, 11.6, 0.5, 2.2, 1.85,
     "Clinical Output",
     "Risk score (0–1)\nRisk tier (cluster 0/1/2)\nSHAP explanation\nThreshold choice\n(clinic vs screening)",
     CF)

# ---- Arrows ----
# Input → Preprocessing
harrow(ax2, 2.4, 2.85, 4.0, "151 cols")
harrow(ax2, 2.4, 2.85, 4.0)
# actually vertical within preprocessing — handled by boxes
# Preprocessing leakage → cleaning
ax2.annotate("", xy=(4.2, 3.5 + 0.9), xytext=(4.2, 4.6),
             arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.2))
ax2.annotate("", xy=(4.2, 2.4 + 0.9), xytext=(4.2, 3.5),
             arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.2))

# Preprocessing → Feature Engineering
harrow(ax2, 5.55, 5.95, 4.0, "108 cols")

# Feature engineering internal arrow
ax2.annotate("", xy=(7.25, 3.1 + 1.1), xytext=(7.25, 4.5),
             arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.2))

# Feature Engineering → Model
harrow(ax2, 8.55, 9.1, 4.25, "116 features")

# Model → Post-processing
harrow(ax2, 11.5, 11.9, 4.75, "probabilities")
harrow(ax2, 11.5, 11.9, 3.3,  "SHAP values")

# Post-processing → Clinical Output
ax2.annotate("", xy=(12.7, 0.5 + 1.85), xytext=(12.7, 2.6),
             arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.3))

# ---- Input arrow ----
harrow(ax2, 2.4, 2.85, 3.95)

# ---- Also connect input directly to leakage box ----
ax2.annotate("", xy=(2.85, 5.05), xytext=(2.4, 5.05),
             arrowprops=dict(arrowstyle="-|>", color=BORDER, lw=1.3))

# ---- Legend ----
legend_items2 = [
    mpatches.Patch(facecolor=CA,        edgecolor=BORDER, label="Data Input"),
    mpatches.Patch(facecolor="#FADBD8", edgecolor=BORDER, label="Leakage Guard"),
    mpatches.Patch(facecolor=CB,        edgecolor=BORDER, label="Preprocessing"),
    mpatches.Patch(facecolor=CC,        edgecolor=BORDER, label="Feature Engineering"),
    mpatches.Patch(facecolor=CD,        edgecolor=BORDER, label="Model Module"),
    mpatches.Patch(facecolor=CE,        edgecolor=BORDER, label="Post-processing / Explainability"),
    mpatches.Patch(facecolor=CF,        edgecolor=BORDER, label="Clinical Output"),
]
ax2.legend(handles=legend_items2, loc="upper left",
           fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.0, 0.28))

ax2.set_title("System Architecture — Leakage-Free PPD Prediction Framework",
              fontsize=13, fontweight="bold", pad=8, color=BORDER)

fig2.tight_layout()
fig2.savefig("Images/fig_system_architecture.jpg", dpi=180,
             bbox_inches="tight", facecolor="white")
print("Saved: Images/fig_system_architecture.jpg")
plt.close(fig2)

print("Done. Both diagrams saved to Images/")
