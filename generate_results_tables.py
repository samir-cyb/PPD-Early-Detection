"""
generate_results_tables.py
===========================
Run this to create PPD_Results_Tables.docx in the same folder.

Requirements:
    pip install python-docx

Usage:
    python generate_results_tables.py
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(THIS_DIR, "PPD_Results_Tables.docx")

# ── helpers ───────────────────────────────────────────────────────────────────

def shade_cell(cell, hex_color="D9E1F2"):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def set_cell_text(cell, text, bold=False, size=10, align=WD_ALIGN_PARAGRAPH.LEFT,
                  color=None, italic=False):
    cell.text = ""
    para = cell.paragraphs[0]
    para.alignment = align
    run = para.add_run(str(text))
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = "Calibri"
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

def set_col_widths(table, widths_inches):
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            if i < len(widths_inches):
                cell.width = Inches(widths_inches[i])

def add_heading(doc, text, level=2):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = "Calibri"

def add_note(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(9)
    run.font.italic = True
    run.font.name = "Calibri"
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

def add_spacer(doc):
    doc.add_paragraph()

# Color constants
HDR_BLUE  = "D9E1F2"
BEST_GREEN = "E2EFDA"
WARN_YELLOW = "FFF2CC"
ADOPTED_BLUE = "DCE6F1"

# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Table 1: Baseline Models ──────────────────────────────────────────────────
BASELINE_ROWS = [
    # (name, ppd_cv, ppd_test, ppd_auc, epds_cv, epds_test, epds_auc, phq9_cv, phq9_test, phq9_auc, is_adopted)
    ("Logistic Regression",     "0.7267","0.7341","0.8032", "0.5461","0.4708","0.7160", "0.3458","0.3601","0.6556", False),
    ("Random Forest ★",         "0.7459","0.7362","0.8261", "0.5828","0.5201","0.7485", "0.3501","0.3668","0.6931", True),
    ("XGBoost",                 "0.7397","0.7350","0.8248", "0.5468","0.4822","0.7309", "0.3622","0.3591","0.6732", False),
    ("LightGBM",                "0.7588","0.7279","0.8186", "0.5391","0.4971","0.7234", "0.3397","0.3992","0.6770", False),
    ("CatBoost",                "0.7492","0.7561","0.8170", "0.5371","0.5171","0.7498", "0.3374","0.3356","0.7029", False),
    ("Lin. Reg. + Threshold ★", "—","—","—", "—","—","—", "Acc: 0.463","F1: 0.426\nQWK: 0.602","—", True),
]

# ── Table 2: Advanced Models ──────────────────────────────────────────────────
# Columns: model | PPD cv_f1 | PPD test_f1 | PPD auc | EPDS cv_f1 | EPDS test_f1 | EPDS auc | PHQ9 cv_f1 | PHQ9 test_f1 | PHQ9 auc
ADVANCED_ROWS = [
    # (name, ppd_cv, ppd_test, ppd_auc, epds_cv, epds_test, epds_auc, phq9_cv, phq9_test, phq9_auc, category)
    # category: "hybrid", "tuned", "dl"
    ("Stacking Ensemble (Hybrid) ⚠",
     "0.7618","0.7654","0.8089",
     "0.5771","0.5411","0.7604",
     "0.3551","0.3348","0.7257", "hybrid"),

    ("Random Forest (Tuned)",
     "0.7574","0.7570","0.7929",
     "0.5896","0.5020","0.7491",
     "0.3700","0.3167","0.6999", "tuned"),

    ("LightGBM (Tuned)",
     "0.7541","0.7193","0.8129",
     "0.5512","0.5058","0.7418",
     "0.3464","0.3796","0.6941", "tuned"),

    ("XGBoost (Tuned)",
     "0.7424","0.7502","0.8025",
     "0.5430","0.5368","0.7523",
     "0.3527","0.3485","0.6727", "tuned"),

    ("CatBoost (Tuned)",
     "0.7524","0.7491","0.8124",
     "0.5598","0.5398","0.7633",
     "0.3606","0.3383","0.7060", "tuned"),

    ("MLP — Deep Learning",
     "0.7403","0.7047","0.8043",
     "0.4994","0.3874","0.6446",
     "0.3131","0.3001","0.6181", "dl"),
]

# ── Table 3: RF vs Stacking Decision ─────────────────────────────────────────
RF_VS_STACKING = [
    # (criterion, rf_value, stacking_value, winner)
    ("PPD_binary Test F1",      "0.7362", "0.7654", "stacking"),
    ("PPD_binary AUC",          "0.8261", "0.8089", "rf"),
    ("EPDS Result Test F1",     "0.5201", "0.5411", "stacking"),
    ("EPDS Result AUC",         "0.7485", "0.7604", "stacking"),
    ("McNemar p-value",         "—",      "p = 0.228 (n.s.)", "rf"),
    ("Bootstrap p-value",       "—",      "p = 0.074 (n.s.)", "rf"),
    ("SHAP Explainability",     "✔ Full SHAP", "✘ Black box", "rf"),
    ("Clinical Interpretability","✔ Feature-level reasons", "✘ Cannot explain", "rf"),
    ("Model Complexity",        "Simple (1 model)", "High (6 models)", "rf"),
    ("Adopted?",                "✔ YES", "✘ No", "rf"),
]

# ── Table 4: Composite Index ──────────────────────────────────────────────────
COMPOSITE_ROWS = [
    ("ESI",  "Economic Stability Index",           "0.214", "1.044", "6/6",  "Items measure different economic stressors"),
    ("SSI",  "Social Support Index",               "0.456", "1.324", "6/6",  "Moderate α — breadth of social support"),
    ("MHRI", "Maternal Mental Health Risk Index",  "0.298", "1.337", "6/6",  "Diverse mental health markers"),
    ("NSI",  "Neonatal & Delivery Stress Index",   "0.343", "1.014", "6/6",  "Distinct perinatal stressors"),
]

# ── Table 5: Tier 2 Closing Loop ─────────────────────────────────────────────
TIER2_ROWS = [
    # (config, epds_f1, epds_std, ppd_f1, ppd_std, phq9_f1, phq9_std)
    ("Baseline",        "0.5785","0.0349", "0.7527","0.0272", "0.3523","0.0438"),
    ("+Interactions",   "0.5726","0.0435", "0.7557","0.0298", "0.3340","0.0399"),
    ("+Cluster",        "0.5831","0.0296", "0.7508","0.0264", "0.3527","0.0418"),
    ("+Both",           "0.5813","0.0351", "0.7597","0.0382", "0.3337","0.0323"),
]

# ── Table 6: SHAP Top-10 ─────────────────────────────────────────────────────
SHAP_DATA = {
    "PPD_binary (Binary)": [
        (1,  "Angry after latest child birth_Yes",           "0.0830", "Raw"),
        (2,  "Maternal_MentalHealth_Risk_Index_Weighted",    "0.0511", "Composite"),
        (3,  "Abuse_Yes",                                     "0.0355", "Raw"),
        (4,  "Social_Support_Index_Weighted",                "0.0334", "Composite"),
        (5,  "Economic_Stability_Index_Weighted",            "0.0235", "Composite"),
        (6,  "Feeling for regular activities_Not Reported",  "0.0184", "Raw"),
        (7,  "Neonatal_Delivery_Stress_Index_Weighted",      "0.0180", "Composite"),
        (8,  "Major changes or losses during pregnancy_Yes", "0.0169", "Raw"),
        (9,  "Relax/sleep when newborn is asleep_Yes",       "0.0159", "Raw"),
        (10, "Relationship with husband_Good",               "0.0153", "Raw"),
    ],
    "EPDS Result (3-class)": [
        (1,  "Angry after latest child birth_Yes",           "0.0552", "Raw"),
        (2,  "Maternal_MentalHealth_Risk_Index_Weighted",    "0.0473", "Composite"),
        (3,  "Social_Support_Index_Weighted",                "0.0376", "Composite"),
        (4,  "Abuse_Yes",                                     "0.0321", "Raw"),
        (5,  "Feeling for regular activities_Not Reported",  "0.0251", "Raw"),
        (6,  "Economic_Stability_Index_Weighted",            "0.0207", "Composite"),
        (7,  "Need for Support_Not Reported",                "0.0199", "Raw"),
        (8,  "Neonatal_Delivery_Stress_Index_Weighted",      "0.0162", "Composite"),
        (9,  "Relax/sleep when newborn is asleep_Yes",       "0.0144", "Raw"),
        (10, "Major changes or losses during pregnancy_Yes", "0.0142", "Raw"),
    ],
    "PHQ9 Result (5-class)": [
        (1,  "Social_Support_Index_Weighted",                "0.0338", "Composite"),
        (2,  "Angry after latest child birth_Yes",           "0.0332", "Raw"),
        (3,  "Maternal_MentalHealth_Risk_Index_Weighted",    "0.0275", "Composite"),
        (4,  "Abuse_Yes",                                     "0.0204", "Raw"),
        (5,  "Relationship with the in-laws_Good",          "0.0127", "Raw"),
        (6,  "Major changes or losses during pregnancy_Yes", "0.0120", "Raw"),
        (7,  "Relax/sleep when the newborn is asleep_Yes",  "0.0107", "Raw"),
        (8,  "Neonatal_Delivery_Stress_Index_Weighted",      "0.0104", "Composite"),
        (9,  "Trust and share feelings_Yes",                 "0.0104", "Raw"),
        (10, "Depression during pregnancy (PHQ2)_Positive",  "0.0093", "Raw"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD DOCUMENT
# ═══════════════════════════════════════════════════════════════════════════════

doc = Document()
section = doc.sections[0]
section.left_margin   = Inches(0.75)
section.right_margin  = Inches(0.75)
section.top_margin    = Inches(1.0)
section.bottom_margin = Inches(1.0)

# Title
title = doc.add_heading("PPD Prediction — Full Model Results", level=1)
for run in title.runs:
    run.font.name = "Calibri"
    run.font.size = Pt(18)

sub = doc.add_paragraph(
    "Leakage-Free Machine Learning Framework for Postpartum Depression Prediction in Bangladesh"
)
sub.runs[0].font.italic = True
sub.runs[0].font.size = Pt(11)
sub.runs[0].font.color.rgb = RGBColor(0x44, 0x44, 0x44)
add_spacer(doc)

# ───────────────────────────────────────────────────────────────────────────────
# TABLE 1: Baseline Model Performance
# ───────────────────────────────────────────────────────────────────────────────
add_heading(doc, "Table 1: Baseline Model Performance Comparison")
add_note(doc,
    "★ = adopted model. All models trained on Arm B (raw + composite indices). "
    "Lin. Reg. + Threshold is the adopted model for PHQ9 (ordinal-aware). "
    "CV F1 = 5-fold F1-macro; Test F1 = holdout F1-macro; AUC = holdout ROC-AUC.")
add_spacer(doc)

t1 = doc.add_table(rows=1, cols=10)
t1.style = "Table Grid"

# merged group headers
h = t1.rows[0].cells
h[1].merge(h[3]); h[4].merge(h[6]); h[7].merge(h[9])
set_cell_text(h[0], "Model",                   bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h[0])
set_cell_text(h[1], "PPD_binary (Binary)",      bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h[1])
set_cell_text(h[4], "EPDS Result (3-class)",    bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h[4])
set_cell_text(h[7], "PHQ9 Result (5-class)",    bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h[7])

# sub-headers
sr = t1.add_row().cells
set_cell_text(sr[0], "", bold=True); shade_cell(sr[0])
for i, lbl in enumerate(["CV F1","Test F1","AUC"] * 3):
    set_cell_text(sr[i+1], lbl, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(sr[i+1])

for row in BASELINE_ROWS:
    name, *vals, adopted = row
    dr = t1.add_row().cells
    set_cell_text(dr[0], name, bold=adopted)
    if adopted:
        shade_cell(dr[0], ADOPTED_BLUE)
    for i, v in enumerate(vals):
        set_cell_text(dr[i+1], v, align=WD_ALIGN_PARAGRAPH.CENTER, bold=adopted)
        if adopted:
            shade_cell(dr[i+1], ADOPTED_BLUE)

set_col_widths(t1, [1.8, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.8, 0.9, 0.65])
add_spacer(doc)

# ───────────────────────────────────────────────────────────────────────────────
# TABLE 2: Advanced Models (Stacking / Tuned / MLP)
# ───────────────────────────────────────────────────────────────────────────────
add_heading(doc, "Table 2: Advanced Model Comparison (Stacking Ensemble, Tuned, Deep Learning)")
add_note(doc,
    "Stacking Ensemble = meta-model over LR + RF + XGBoost + LightGBM + CatBoost. "
    "Tuned = RandomizedSearchCV hyperparameter search. MLP = Multi-Layer Perceptron (deep learning). "
    "⚠ Stacking gives slightly higher F1 but the difference is NOT statistically significant "
    "(McNemar p=0.228, Bootstrap p=0.074) and cannot be explained with SHAP.")
add_spacer(doc)

t2 = doc.add_table(rows=1, cols=10)
t2.style = "Table Grid"

h2 = t2.rows[0].cells
h2[1].merge(h2[3]); h2[4].merge(h2[6]); h2[7].merge(h2[9])
set_cell_text(h2[0], "Model",                 bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h2[0])
set_cell_text(h2[1], "PPD_binary (Binary)",    bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h2[1])
set_cell_text(h2[4], "EPDS Result (3-class)",  bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h2[4])
set_cell_text(h2[7], "PHQ9 Result (5-class)",  bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h2[7])

sr2 = t2.add_row().cells
set_cell_text(sr2[0], "", bold=True); shade_cell(sr2[0])
for i, lbl in enumerate(["CV F1","Test F1","AUC"] * 3):
    set_cell_text(sr2[i+1], lbl, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(sr2[i+1])

CAT_COLORS = {"hybrid": WARN_YELLOW, "tuned": "EDEDED", "dl": "F4CCCC"}

for row in ADVANCED_ROWS:
    name, *vals, cat = row
    color = CAT_COLORS.get(cat, "FFFFFF")
    dr = t2.add_row().cells
    set_cell_text(dr[0], name, bold=(cat == "hybrid"))
    shade_cell(dr[0], color)
    for i, v in enumerate(vals):
        set_cell_text(dr[i+1], v, align=WD_ALIGN_PARAGRAPH.CENTER)
        shade_cell(dr[i+1], color)

set_col_widths(t2, [1.8, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65])
add_spacer(doc)

# ───────────────────────────────────────────────────────────────────────────────
# TABLE 3: RF vs Stacking — Why RF was chosen
# ───────────────────────────────────────────────────────────────────────────────
add_heading(doc, "Table 3: Why Random Forest Over Stacking Ensemble?")
add_note(doc,
    "The performance difference between RF and Stacking is statistically non-significant. "
    "RF is adopted because it is fully explainable (SHAP) and clinically interpretable. "
    "n.s. = not significant.")
add_spacer(doc)

t3 = doc.add_table(rows=1, cols=4)
t3.style = "Table Grid"

h3 = t3.rows[0].cells
for i, lbl in enumerate(["Criterion", "Random Forest (Adopted ★)", "Stacking Ensemble", "Verdict"]):
    set_cell_text(h3[i], lbl, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h3[i])

for criterion, rf_val, stack_val, winner in RF_VS_STACKING:
    dr = t3.add_row().cells
    set_cell_text(dr[0], criterion, bold=True)
    rf_bold   = (winner == "rf")
    st_bold   = (winner == "stacking")
    set_cell_text(dr[1], rf_val,    bold=rf_bold,
                  color="1F7034" if rf_bold else None,
                  align=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell_text(dr[2], stack_val, bold=st_bold,
                  color="1F7034" if st_bold else None,
                  align=WD_ALIGN_PARAGRAPH.CENTER)
    verdict = "RF ✔" if winner == "rf" else "Stacking ✔"
    set_cell_text(dr[3], verdict, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
                  color="1F7034" if winner == "rf" else "C00000")
    if winner == "rf":
        shade_cell(dr[1], BEST_GREEN)
    else:
        shade_cell(dr[2], BEST_GREEN)

set_col_widths(t3, [2.2, 2.0, 2.0, 1.3])
add_spacer(doc)

# ───────────────────────────────────────────────────────────────────────────────
# TABLE 4: Composite Index Reliability
# ───────────────────────────────────────────────────────────────────────────────
add_heading(doc, "Table 4: Composite Risk Index Reliability")
add_note(doc,
    "Four composite indices built from 24 survey features using chi-square weighting. "
    "Cronbach α = internal consistency. Max VIF < 1.4 confirms no multicollinearity. "
    "All 24/24 association tests significant (p < 0.05).")
add_spacer(doc)

t4 = doc.add_table(rows=1, cols=6)
t4.style = "Table Grid"

h4 = t4.rows[0].cells
for i, lbl in enumerate(["Index","Full Name","Cronbach α","Max VIF","Sig. Assoc.","Note"]):
    set_cell_text(h4[i], lbl, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h4[i])

for abbr, full, alpha, vif, assoc, note in COMPOSITE_ROWS:
    dr = t4.add_row().cells
    set_cell_text(dr[0], abbr,   bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell_text(dr[1], full)
    set_cell_text(dr[2], alpha,  align=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell_text(dr[3], vif,    align=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell_text(dr[4], assoc,  align=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell_text(dr[5], note)

# summary row
sr4 = t4.add_row().cells
for c in sr4: shade_cell(c)
set_cell_text(sr4[0], "All 4",      bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
set_cell_text(sr4[1], "—",          align=WD_ALIGN_PARAGRAPH.CENTER)
set_cell_text(sr4[2], "0.214–0.456",align=WD_ALIGN_PARAGRAPH.CENTER)
set_cell_text(sr4[3], "< 1.4",      align=WD_ALIGN_PARAGRAPH.CENTER)
set_cell_text(sr4[4], "24 / 24",    bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
set_cell_text(sr4[5], "McNemar p=0.228; Bootstrap p=0.074 (RF vs Stacking, both n.s.)")

set_col_widths(t4, [0.55, 1.9, 0.8, 0.7, 0.85, 3.15])
add_spacer(doc)

# ───────────────────────────────────────────────────────────────────────────────
# TABLE 5: Tier 2 Closing Loop (Script 20)
# ───────────────────────────────────────────────────────────────────────────────
add_heading(doc, "Table 5: Tier 2 — Closing Loop Test (Script 20)")
add_note(doc,
    "Tests whether adding interaction features or cluster membership improves performance. "
    "Mean CV F1-macro ± std across 5 folds. "
    "RESULT: No configuration beats baseline — all deltas are smaller than fold-to-fold std. "
    "Baseline model is sufficient.")
add_spacer(doc)

t5 = doc.add_table(rows=1, cols=7)
t5.style = "Table Grid"

h5 = t5.rows[0].cells
for i, lbl in enumerate([
    "Configuration",
    "EPDS F1 (mean)","EPDS F1 (std)",
    "PPD_binary F1 (mean)","PPD_binary F1 (std)",
    "PHQ9 F1 (mean)","PHQ9 F1 (std)"
]):
    set_cell_text(h5[i], lbl, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h5[i])

for i, (cfg, ef, es, pf, ps, qf, qs) in enumerate(TIER2_ROWS):
    dr = t5.add_row().cells
    is_baseline = (i == 0)
    if is_baseline:
        for c in dr: shade_cell(c, BEST_GREEN)
    set_cell_text(dr[0], cfg,  bold=is_baseline)
    set_cell_text(dr[1], ef,   align=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell_text(dr[2], f"± {es}", align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
    set_cell_text(dr[3], pf,   align=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell_text(dr[4], f"± {ps}", align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
    set_cell_text(dr[5], qf,   align=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell_text(dr[6], f"± {qs}", align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)

set_col_widths(t5, [1.35, 1.0, 0.8, 1.1, 0.8, 1.0, 0.8])
add_spacer(doc)

# ───────────────────────────────────────────────────────────────────────────────
# TABLE 6: SHAP Top-10
# ───────────────────────────────────────────────────────────────────────────────
add_heading(doc, "Table 6: SHAP Top-10 Features per Target (Random Forest, Arm B)")
add_note(doc,
    "Mean absolute SHAP values from TreeExplainer on holdout set. "
    "'Composite' = chi-square-weighted index feature (highlighted green). "
    "'Raw' = individual survey item. "
    "Composite indices appear in top-10 for all three targets.")
add_spacer(doc)

t6 = doc.add_table(rows=1, cols=5)
t6.style = "Table Grid"

h6 = t6.rows[0].cells
for i, lbl in enumerate(["Rank","Target","Feature","Mean |SHAP|","Type"]):
    set_cell_text(h6[i], lbl, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER); shade_cell(h6[i])

for target, items in SHAP_DATA.items():
    for rank, feat, shap_val, ftype in items:
        dr = t6.add_row().cells
        is_comp = (ftype == "Composite")
        set_cell_text(dr[0], str(rank), align=WD_ALIGN_PARAGRAPH.CENTER)
        set_cell_text(dr[1], target if rank == 1 else "", bold=(rank == 1))
        set_cell_text(dr[2], feat)
        set_cell_text(dr[3], shap_val, align=WD_ALIGN_PARAGRAPH.CENTER)
        set_cell_text(dr[4], ftype, bold=is_comp, align=WD_ALIGN_PARAGRAPH.CENTER,
                      color="1F7034" if is_comp else None)
        if is_comp:
            shade_cell(dr[4], BEST_GREEN)

set_col_widths(t6, [0.45, 1.3, 4.25, 0.85, 0.85])
add_spacer(doc)

# Footer
fp = doc.add_paragraph()
fp.add_run(f"Generated: {__import__('datetime').date.today()}  |  "
           f"Total tables: 6  |  Framework: Leakage-Free PPD Prediction").font.size = Pt(8)

doc.save(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")
print("Tables created:")
print("  Table 1 — Baseline model performance (all 5 models + LinReg+Threshold)")
print("  Table 2 — Advanced models (Stacking Ensemble, Tuned, MLP)")
print("  Table 3 — RF vs Stacking: why RF was chosen")
print("  Table 4 — Composite index reliability (α, VIF, associations)")
print("  Table 5 — Tier 2 closing loop test (Script 20)")
print("  Table 6 — SHAP Top-10 features per target")
