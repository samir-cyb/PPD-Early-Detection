# PPD Project — Tests, Scoring & Visualization: Consolidated Review

*Purpose: single place to review everything already computed (Steps 1–20)
before starting the paper write-up, plus what Script 21/22 add on top.
Every number below is cross-checked against the actual CSV/txt files in
`codes/outputs/`, not just the narrative in PROJECT_STATUS.md.*

---

## 1. Composite Scoring Formulas

Four composite risk indices, each built the same way: raw sub-feature →
documented ordinal map → min-max normalize to [0,1] → combine → scale to
0–10. Two variants exist for each; the **weighted** variant is the one
adopted for modeling/reporting.

| Index | Sub-features (raw columns) | Direction |
|---|---|---|
| **Economic Stability Index (ESI)** | Current monthly income, Husband's monthly income, Occupation after childbirth, Education Level | higher = more stable |
| **Social/Family Support Index (SSI)** | Family type, Relationship with in-laws, Relationship with husband, Household size, Received Support | higher = more support |
| **Maternal Mental-Health Risk Index (MHRI)** | PHQ2 before/during pregnancy, Disease before pregnancy, Abuse, History of pregnancy loss | higher = more risk |
| **Neonatal/Delivery Stress Index (NSI)** | Mode of delivery, Birth complications, Newborn illness, Worry about newborn | higher = more stress |

**Equal-weight formula:** `index = mean(normalized sub-scores) × 10`

**Weighted formula (adopted):**
`index = Σ(normalized sub-score × wᵢ) × 10`, where
`wᵢ = -log10(p_i) / Σ(-log10(p_j))` and `p_i` is the chi-square association
p-value of that sub-feature against `PPD_binary` on its own.

Why weighted, not equal: on the first run, equal-weighting made MHRI and ESI
non-significant against some targets because averaging diluted `Abuse`
(the single strongest raw predictor, -log10(p)=24.82) with weaker
co-features. Weighting fixed this — **24/24 significance tests pass** for
the weighted indices across all 3 targets (source: `composite_index_association.csv`,
`composite_index_summary.txt`).

**Final weights actually used** (from `composite_index_summary.txt`):

| Index | Sub-feature | Normalized weight |
|---|---|---|
| ESI | education | 0.745 |
| ESI | income_now | 0.127 |
| ESI | husband_income | 0.098 |
| ESI | occupation | 0.030 |
| SSI | relationship_in_laws | 0.346 |
| SSI | relationship_husband | 0.334 |
| SSI | received_support | 0.226 |
| SSI | family_type | 0.060 |
| SSI | household_members | 0.034 |
| MHRI | abuse | 0.634 |
| MHRI | phq2_before | 0.146 |
| MHRI | phq2_during | 0.138 |
| MHRI | disease_before | 0.079 |
| MHRI | pregnancy_loss | 0.004 |
| NSI | worry_newborn | 0.525 |
| NSI | birth_complications | 0.341 |
| NSI | delivery_mode | 0.113 |
| NSI | newborn_illness | 0.021 |

**Known caveat to document in Methods/Limitations:** the `Abuse` map was
empirically flipped (`Yes→0, No→1`) because "No" answers associate with a
64.1% PPD-positive rate vs 27.0% for "Yes" — the opposite of clinical
expectation. The index stays internally consistent, but the real-world
meaning of the survey question itself is unresolved (possible translation
or question-framing artifact) — worth flagging explicitly rather than
silently normalizing over it.

---

## 2. Statistical Tests Inventory (all already run)

| Test | Purpose | Script | Key result |
|---|---|---|---|
| Chi-square / ANOVA / Kruskal-Wallis | Raw feature vs target association | `03_association_eda.py` | 28–31 of 53 features significant per target |
| Chi-square / ANOVA / Kruskal-Wallis | Composite index vs target association | `04_feature_engineering.py` | 24/24 significant (weighted variants) |
| Paired t-test + Wilcoxon signed-rank (5-fold) | Ablation arm differences (B vs A, C vs B) real or noise? | `08_ablation_study.py` | 0/30 significant — underpowered at 5 folds |
| Paired t-test + Wilcoxon (RepeatedStratifiedKFold, 5×5=25 folds) | Same question, more power | `12_catboost_native_and_repeated_cv.py` | 2/15 significant (EPDS×XGBoost p=0.0008; PHQ9×LightGBM p=0.002) |
| Quadratic Weighted Kappa (QWK) | Ordinal-aware accuracy for EPDS/PHQ9 severity bands | `10_ordinal_regression.py` | EPDS 0.557, PHQ9 0.602 (regression+threshold) |
| Precision-Recall threshold tuning + Brier score/calibration | PPD_binary decision threshold + probability trustworthiness | `11_threshold_calibration.py` | Max-F1 threshold 0.407 beats default 0.5; Brier 0.1677→0.1667 |
| Chi-square | Unsupervised cluster PPD-prevalence difference | `18_cluster_analysis.py` | p=3.73e-28 (3.6× prevalence spread) |
| Mann-Whitney U | What separates FN/TP and FP/TN (error analysis) | `19_error_analysis.py` | MHRI p=2.95e-07 (FN vs TP), p=7.2e-07 (FP vs TN) |
| SHAP interaction values | Which feature PAIRS drive predictions jointly | `17_shap_interactions.py` | Strongest pair in all 3 targets involves 2 composite indices |
| Permutation vs SHAP-additive group importance | Cross-check one-hot-hidden raw-question importance | `15_age_features_and_group_importance.py`, `16_shap_arm_comparison_and_group.py` | 3 under-rated raw variables surfaced for PHQ9 |

## 2b. New tests — Script 21 (RUN — DONE, results below)

These close 4 real gaps a reviewer of a composite-index paper would ask
about, none of which existed in scripts 01–20. **Run confirmed by Samir,
2026 — no errors.**

| New test | Result | Interpretation |
|---|---|---|
| **Cronbach's alpha** (per composite index) | ESI=0.214, SSI=0.456, MHRI=0.298, NSI=0.343 — all "unacceptable" by the conventional >0.7 rule of thumb | Sub-features inside each index are only weakly inter-correlated. **Not a bug and not a contradiction of the 24/24 predictive-validity result** — see caveat below. |
| **VIF** (4 weighted composites together) | ESI=1.04, SSI=1.32, MHRI=1.34, NSI=1.01 — all «5 | No multicollinearity. The 4 indices are safely usable together; their individual SHAP rankings can be read as independent contributions. |
| **McNemar's test** (RF vs Stacking, PPD_binary, same 160-patient holdout) | b=3 (RF right/Stacking wrong), c=8 (Stacking right/RF wrong), chi2=1.455, **p=0.228 (not significant)** | Stacking's edge is not statistically distinguishable from RF on this test set. |
| **Bootstrap 95% CI**, AUC(RF)−AUC(Stacking) | RF AUC=0.8261, Stacking AUC=0.8084, diff=+0.0177, 95% CI=[−0.0016, +0.0380], p=0.074 | CI barely straddles 0 (borderline) — consistent with McNemar: no statistically proven accuracy cost to choosing RF for explainability. |

**Important honest caveat on the low Cronbach's alpha (write this into
Methods/Limitations, don't hide it):** these composites were designed as
*content-valid, domain-coverage* risk indices (grouping clinically related
but conceptually distinct sub-features, e.g. MHRI mixes PHQ2 history +
disease history + abuse + pregnancy loss — related risk *domains*, not
repeated measurements of one underlying trait). Cronbach's alpha assumes
items ARE repeated measurements of a single latent trait, which is a
different design goal than this project's. **Report both properties
side by side and explain the distinction explicitly:** predictive validity
(association tests, 24/24 significant, p as low as 1e-35) is strong;
internal-consistency reliability (alpha) is low, because the indices were
never intended to be unidimensional psychometric scales. This is a
legitimate, defensible thing to say in a paper — but it MUST be stated
proactively, since a reviewer who checks alpha without this framing will
flag it as a weakness. Do not report alpha without this paragraph next to
it.

Files written: `composite_reliability_cronbach_alpha.csv`, `composite_vif.csv`,
`rf_vs_stacking_significance_PPD_binary.csv`,
`reliability_and_extra_tests_summary.txt`.

---

## 3. Visualization Inventory (already exists, `codes/outputs/`)

| Figure | Shows | Source script |
|---|---|---|
| `epds_class_distribution.png`, `phq9_class_distribution.png`, `missingness_bar.png` | Raw dataset overview | `01_dataset_overview.py` |
| `shap_summary_bar_<target>.png`, `shap_beeswarm_<target>.png` | Global SHAP feature importance, ×3 targets | `09_shap_explainability.py` |
| `shap_dependence_<index>_<target>.png` (×4 indices ×3 targets) | SHAP dependence plots for each composite index | `09_shap_explainability.py` |
| `shap_waterfall_highrisk_<target>.png` / `..._lowrisk_<target>.png` | Single-patient prediction breakdown | `09_shap_explainability.py` |
| `precision_recall_curve.png`, `calibration_curve.png` | PPD_binary threshold tuning + calibration | `11_threshold_calibration.py` |
| `shap_interaction_heatmap_<target>.png` | Pairwise SHAP interaction strength | `17_shap_interactions.py` |
| `cluster_pca_scatter.png` | 2D PCA view of the 3 unsupervised clusters | `18_cluster_analysis.py` |

**32 SHAP plots + 8 other charts already exist** — enough raw material for
the Results section, but every one is per-target/per-index (no single
figure puts all 3 targets or all 4 indices side by side, which is usually
what a paper figure needs).

## 3b. New polished/consolidated figures added now (Script 22 — not yet run)

| New figure | Shows | Why it's needed on top of what exists |
|---|---|---|
| `fig1_model_comparison.png` | Grouped bar: CV F1-macro, 5 models × 3 targets, one image | Existing `model_performance_*.csv` tables have never been plotted together |
| `fig2_composite_boxplots.png` | 2×2 boxplots: each weighted composite index by PPD status | No existing chart shows the composite score distributions directly |
| `fig3_ablation_comparison.png` | Grouped bar: Arm A/B/C mean F1-macro × 3 targets, with value labels | Turns the ablation *table* into the ablation *figure* readers expect |
| `fig4_cluster_prevalence.png` | Bar chart: PPD prevalence % by cluster, labeled with n | Cleaner, paper-ready alternative/complement to the PCA scatter |
| `fig5_shap_top10_consolidated.png` | 3-panel horizontal bar, top-10 SHAP features per target, composite indices color-highlighted | Existing SHAP bars are separate per target; this is the "before/after, composites rise to top" story in one glance |

Run with: `python codes/22_paper_figures.py` — outputs 5 PNGs (300 dpi) to
`codes/outputs/paper_figures/`. **Fig2 hit a matplotlib version error
(`boxplot(labels=...)` renamed in newer matplotlib) — fixed, re-run needed.**

## 3c. Figures that were MISSING from Script 22 — now covered by Script 23

Samir correctly flagged that Script 22 only covered Steps 7/8/9/18 and
skipped Step 7b, Tier 1 (10/11), Script 20, and Script 21's own new results.
`codes/23_additional_paper_figures.py` adds the 6 missing figures:

| New figure | Covers | Why it matters for the paper |
|---|---|---|
| `fig6_advanced_modeling_comparison.png` | Step 7b: baseline vs tuned vs Stacking vs MLP, ranked, ×3 targets | Shows tuning/DL gave marginal/inconsistent gains and Stacking is the best raw predictor — sets up the Novelty 6 trade-off |
| `fig7_ordinal_regression_comparison.png` | Step 10: Direct Classifier vs best Regression+Threshold, Accuracy/F1/QWK, EPDS & PHQ9 | Visualizes the one target (PHQ9) where the modeling APPROACH itself changed — a headline Novelty |
| `fig8_threshold_comparison.png` | Step 11: Default/Max-F1/Screening thresholds, Precision/Recall/F1 | Shows the free F1 gain from threshold retuning and the precision/recall trade-off of the screening option |
| `fig9_closing_loop_test.png` | Script 20: baseline vs +interactions vs +cluster vs +both, ×3 targets, with error bars | The honest "nothing beat baseline" negative result, visualized so the error bars visibly swallow every delta |
| `fig10_rf_vs_stacking.png` | Script 21 (this session): RF vs Stacking AUC with McNemar/bootstrap p in the title | Turns Section 2b's new test into a figure, not just a table row |
| `fig11_error_analysis.png` | Step 19: composite index profiles, FN vs TP and FP vs TN, PPD_binary | Shows the model's blind spot is driven by the same 2 composites in both error directions |

**Deliberately still NOT charted (kept as CSV/table-only — secondary
robustness checks, not headline results):** CatBoost native-vs-one-hot
(Step 12), SMOTE/imputation comparison (Step 14), composite
weight-learning comparison (Step 13), Age²/group-importance discrepancy
(Step 15). These read fine as compact tables in an appendix or a
"robustness checks" subsection — say the word if any of them should get
their own figure too (e.g. the Step 15 permutation-vs-SHAP discrepancy for
PHQ9 is a genuinely interesting one if there's room).

Run with: `python codes/23_additional_paper_figures.py` — outputs 6 more
PNGs to the same `codes/outputs/paper_figures/` folder (needs
`21_reliability_and_extra_tests.py` to have been run first, for fig10).

---

## 4. What's still open

1. **Script 22 needs a re-run** (fig1 done; fig2 crashed on a matplotlib
   version issue, now fixed) and **Script 23 needs its first run** — both
   pending because the sandbox on Claude's side can't execute Python.
   Script 21 is DONE (results in Section 2b above).
2. **Low Cronbach's alpha must be framed correctly in the write-up** — see
   the caveat paragraph in Section 2b. Do not report the alpha numbers
   without that explanation attached.
3. **Abuse column semantics** — still needs a real-world check against the
   original Bangla questionnaire wording (see Section 5 of
   `PROJECT_STATUS.md`).
4. Everything else (Steps 1–21) is confirmed final and locked in — no
   re-run needed.

---

## 5. Recommended order for you to run locally

```
cd codes
python 21_reliability_and_extra_tests.py
python 22_paper_figures.py
```

Paste back the console output (or just confirm no errors + list the files
in `codes/outputs/` and `codes/outputs/paper_figures/`) and this review will
be finalized before we move to the actual paper draft.
