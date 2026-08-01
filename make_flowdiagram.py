"""
make_flowdiagram.py
=====================
Generates the "Thesis Organization" flow diagram referenced in
chapter1_introduction.tex as Figure ch1-fig-flowdiag
(\\includegraphics{Images/flowdiag.png}).

Draws 5 stacked boxes, one per chapter, each listing what that chapter
covers (matching the wording already written in the Thesis Organization
section), connected by downward arrows to show reading order.

Output: Images/flowdiag.png (created relative to wherever this script is
run from -- run it from the same folder as chapter1_introduction.tex /
your main.tex so the relative path in \\includegraphics resolves correctly).

Usage
-----
    pip install matplotlib --break-system-packages   # already installed
    python make_flowdiagram.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(THIS_DIR, "Images")
os.makedirs(IMG_DIR, exist_ok=True)

chapters = [
    ("Chapter 1", "Introduction",
     "Motivation, Problem Statement,\nObjectives, Approach & Contribution"),
    ("Chapter 2", "Literature Review",
     "Related ML studies on PPD prediction +\nclinical grounding for the 4 composite indices"),
    ("Chapter 3", "Methodology",
     "Dataset, cleaning, composite risk index\ndesign, leakage-free features, models"),
    ("Chapter 4", "Results & Discussion",
     "Model comparison, ablation study, SHAP\nexplainability, cluster & error analysis"),
    ("Chapter 5", "Conclusion",
     "Summary of findings, limitations,\nfuture work"),
]

n = len(chapters)
box_w, box_h = 7.4, 1.7
gap = 0.9
fig_h = n * box_h + (n - 1) * gap + 1.2
fig_w = box_w + 2.0

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

colors = ["#2C5F8A", "#3D7EA6", "#4E9BB8", "#5FB8AA", "#6FCC8E"]

y = fig_h - 0.6 - box_h
box_centers = []
for i, (num, title, desc) in enumerate(chapters):
    x = (fig_w - box_w) / 2
    box = FancyBboxPatch(
        (x, y), box_w, box_h,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        linewidth=1.5, edgecolor="#1A3A52", facecolor=colors[i], alpha=0.92,
        mutation_aspect=1,
    )
    ax.add_patch(box)
    ax.text(x + box_w / 2, y + box_h - 0.42, f"{num}: {title}",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white")
    ax.text(x + box_w / 2, y + 0.55, desc,
            ha="center", va="center", fontsize=9.3, color="white",
            linespacing=1.5)
    box_centers.append((x + box_w / 2, y))
    y -= (box_h + gap)

# Downward arrows connecting consecutive boxes
for i in range(n - 1):
    x_top, y_top = box_centers[i]
    y_top_bottom = y_top  # bottom edge of box i
    y_next_top = box_centers[i + 1][1] + box_h  # top edge of box i+1
    arrow = FancyArrowPatch(
        (x_top, y_top_bottom - 0.03), (x_top, y_next_top + 0.03),
        arrowstyle="-|>", mutation_scale=22, linewidth=2, color="#1A3A52",
    )
    ax.add_patch(arrow)

fig.tight_layout(pad=0.6)
out_path = os.path.join(IMG_DIR, "flowdiag.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out_path}")
