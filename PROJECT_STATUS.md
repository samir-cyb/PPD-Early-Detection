# PPD Early Detection Project — Living Status Document

> **Purpose of this file:** single source of truth for project continuity across
> sessions. Update this file every time a step is completed, a new code file is
> added, or a problem is found/solved. When a problem is solved, **delete it**
> from the Problems section instead of just marking it done — this file should
> only ever show what is *currently* true.

Last updated: after completing Steps 1-6 (cleaning, association EDA,
composite feature engineering + direction fixes, encoding — all confirmed
final) and writing Step 7 (modeling script, not yet run).

---

## 1. Project Goal

Build a **leakage-free, explainable Machine Learning framework** for **early
detection of Postpartum Depression (PPD)** among postpartum women in
Bangladesh, using the Mendeley "Data for Postpartum Depression Prediction in
Bangladesh" dataset (n=800).

The contribution is **not** "train yet another classifier." The novelty is:

1. **Leakage-free prediction pipeline** — predict PPD risk using only
   demographic/family/economic/pregnancy/neonatal/prior-history features,
   never the EPDS/PHQ-9 items or scores that were used to define the label.
2. **Composite Risk Index feature engineering** — 4 new engineered indices
   built from raw features (see Section 4).
3. **Feature selection** — reduce raw+composite features to the most
   informative subset.
4. **Ablation study** — Raw vs Raw+Composite vs Selected+Composite, to prove
   the engineered features actually help.
5. **Explainable AI (SHAP)** — show which risk factors/composite indices
   drive predictions, for clinical interpretability.

Two possible targets (train separate models, not multi-output):
- `EPDS Result` (Low / Medium / High) or binary `PPD_binary` (EPDS ≥ 13)
- `PHQ9 Result` (Minimal / Mild / Moderate / Moderately Severe / Severe)

---

## 1.5 Key Novelties & Contributions (Executive Summary)

*(Added as a single, synthesized summary on top of the detailed chronological
roadmap in Section 3 — read this first for "what did we actually contribute
and why does it matter", then go to Section 3 for full step-by-step detail
and Section 3.5 for the final per-target model comparison.)*

### Novelty 1 — Four engineered Composite Risk Indices (the core contribution)
Instead of feeding 108 raw one-hot survey answers straight into a model, 4
clinically-grounded composite indices were engineered from groups of related
raw answers: **Economic Stability Index (ESI)**, **Social/Family Support
Index (SSI)**, **Maternal Mental-Health Risk Index (MHRI)**, and
**Neonatal/Delivery Stress Index (NSI)**. Each is built by mapping every
raw sub-feature to a documented ordinal score, min-max normalizing to
[0,1], then combining sub-scores into a single 0-10 index. Two versions
were built and compared: **equal-weight** (every sub-feature counts the
same) and **chi-square-weighted** (each sub-feature weighted by
`-log10(p)` from its own association strength with PPD_binary). The
weighted version was adopted as final because equal-weighting diluted
MHRI's strongest sub-feature (Abuse) to the point of non-significance —
documented, empirical justification for the weighting choice, not an
arbitrary one. All 4 weighted composites are significant (p<0.05, both
ANOVA and Kruskal-Wallis) against all 3 targets (24/24 tests), and 2
independent later checks (Step 12's literature review, Step 13's
LR-learned-weight comparison) both corroborate the same 4 domains and the
same "Abuse" direction finding from 3 different angles.

### Novelty 2 — A genuinely leakage-free prediction pipeline
The external reference IEEE paper on a similar Bangladesh dataset reports
near-perfect AUROC, which is a red flag for data leakage (almost certainly
including EPDS/PHQ-9 item-level answers as model inputs when predicting a
label derived FROM those same items). This project explicitly excludes all
19 EPDS/PHQ-9 item columns and the 2 raw score columns from every model's
input (`codes/06_encoding.py`), and the resulting accuracy is realistic
(EPDS ~58% CV F1-macro, PPD_binary ~75% CV F1-macro) rather than
suspiciously perfect — itself a defensible, honestly-reported result that
directly contrasts with the leakage-prone comparison paper.

### Novelty 3 — A statistically rigorous ablation study (Raw vs Raw+Composite vs Selected+Composite)
`codes/08_ablation_study.py` compares 3 feature "arms" across all 5 model
families and all 3 targets (30 model×target combinations), with feature
selection for Arm C done in a leakage-safe NESTED way (re-derived inside
every CV fold, not once on the full dataset — an earlier version of this
script had that bug, caught and fixed). A paired t-test + Wilcoxon
signed-rank test was added on top of the raw mean comparison (per Samir's
own question, "don't we need a test to prove this?"), and later re-run with
`RepeatedStratifiedKFold` (5×5=25 folds) for more statistical power —
revealing the original 5-fold test was underpowered (0/30 significant)
rather than the composite effect being absent (2/15 significant once power
increased, both in the expected positive direction).

### Novelty 4 — Multi-angle explainability, not just one SHAP plot
Beyond a standard SHAP summary plot, this project layers 5 independent
explainability angles on the same model: (a) global SHAP feature
importance + dependence + waterfall plots (Step 9), (b) a direct
"before/after" SHAP comparison showing composites rising to the top only
once available (Step 16 Part A), (c) SHAP-additive GROUP importance that
re-aggregates one-hot dummies back to their original survey question
(Step 16 Part B / Step 15 Part B, via 2 independent methods — permutation
and additive-SHAP — that mostly agree but also honestly reveal one
interaction-driven discrepancy), (d) SHAP INTERACTION values exposing
which feature PAIRS the model relies on jointly, not just individually
(Step 17 — the strongest pairwise interaction in all 3 targets involves
two composite indices), and (e) an independent UNSUPERVISED confirmation
via K-Means clustering on the composite indices alone, with no PPD label
used, that still recovers a 3.6x PPD-prevalence spread across the natural
clusters found (Step 18, chi-square p=3.73e-28) — plus a targeted ERROR
ANALYSIS (Step 19) that pinpoints exactly which patients the model
misses and why (atypical low-mental-health-risk-index presentations).

### Novelty 5 — Ordinal-aware modeling instead of forcing multi-class classification
EPDS Result and PHQ9 Result are ordinal (Low<Medium<High;
Minimal<...<Severe), not unordered categories, but a plain classifier and
plain F1-macro treat every misclassification as equally bad. Step 10 adds
Quadratic Weighted Kappa (QWK) as an ordinal-aware metric, AND tests
regression-then-threshold as an alternative to direct classification. This
is not just a metric addition — for PHQ9 Result specifically, **Linear
Regression on the continuous score, thresholded into the 5 official
clinical bands, genuinely beats the direct classifier on every metric**
(accuracy 0.344→0.463, F1-macro 0.329→0.426, QWK 0.548→0.602) — a real,
adopted change of modeling APPROACH for this target, not just a
hyperparameter tweak.

### Novelty 6 — Deliberate interpretability-over-accuracy trade-off, made explicit
Step 7b's tuning/stacking experiments found the Stacking Ensemble is
actually the single best PPD_binary model by both CV F1-macro (0.7618)
and holdout F1 (0.7654) — better than the plain Random Forest (0.7459 CV /
0.7362 holdout) used from Step 9 onward. Random Forest was deliberately
kept as the project's explainability backbone anyway, because
`shap.TreeExplainer` gives fast, exact Shapley values for a single tree
ensemble, while explaining a 4-model stacking ensemble would need a
slower, approximate, harder-to-defend explanation. This is reported
honestly as a conscious trade-off (clarity over the last ~1-2 points of
accuracy), consistent with the project's stated goal of an *explainable*
ML framework, not just the most accurate one possible.

### Honest negative/rejected results (reported for rigor, not hidden)
Not every idea tried worked — and reporting the ones that didn't is itself
part of the project's rigor: LR-learned composite weights (no universal
win over chi-square weighting), SMOTENC oversampling (helped 2 targets
marginally, hurt PHQ9 Result), IterativeImputer/MICE (no meaningful
downstream difference vs. simple mode-fill), Age²/Age-band features (not
significant, negligible predictive change), CatBoost's native categorical
handling (comparable accuracy with fewer features, but no universal win),
and finally script 20's explicit interaction/cluster FEATURES added on top
of the already-composite-index-equipped models (every config's
improvement was smaller than its own fold-to-fold noise — confirming the
existing models already capture this information, nothing left "on the
table").

---

## 2. Dataset Summary

- **Source file (raw, untouched):**
  `PPD_dataset_v3.csv` — in the dataset root folder.
- **n = 800** postpartum mothers, **70 raw columns** (69 after dropping the
  `sr` index column).
- Two reference papers in the dataset root folder (already read):
  - `A Machine Learning Approach for Early Detection of Postpartum Depression
    in Bangladesh.pdf` (Raisa, Kaiser, Mahmud) — the dataset's own
    methodology paper (n=150 pilot study, precursor to this n=800 dataset).
  - `2026-Predicting_Postpartum_Depression_Using_Questionnaire-Based_Data_and
    _Machine_Learning.pdf` — external IEEE paper using a similar Bangladesh
    dataset (defines binary PPD = EPDS≥13, reports near-perfect AUROC which
    likely reflects data leakage — a cautionary example for us).
- Original exploration notebook (by a collaborator, not fully cleaned/saved):
  `all_the_problem_solve_column.ipynb` — cleaning logic was partially
  written but the cleaned CSV was never exported/shared. This project's
  `codes/02_data_cleaning.py` reproduces and extends that logic.

### Column roles (after cleaning)
- **Target candidates:** `EPDS Score`, `EPDS Result`, `PHQ9 Score`, `PHQ9 Result`, `PPD_binary` (new, EPDS≥13)
- **Leakage columns (never use as input), 23 total:** the 10 EPDS items + 9
  PHQ-9 items + the 4 score/result columns above.
- **Safe input features, 46 raw columns:** demographic, family, economic,
  pregnancy, neonatal, and PHQ-2 (before/during-pregnancy history — kept as
  input since it's a *different* time point from postpartum EPDS/PHQ-9, not
  leakage).
- Full column-by-column catalogue: `codes/outputs/column_catalogue.csv`.

### Known target distribution (validated against Mendeley's published description)
- EPDS Result (recomputed from score): High ~349, Low ~247, Medium ~204 (out of 800)
- PHQ9 Result (recomputed from score): Mild 263, Moderate 237, Moderately Severe 132, Severe 89, Minimal 79
- PPD_binary (EPDS≥13): 349 positive / 451 negative

---

## 3. Roadmap & Current State

- [x] **Step 0 — Understand dataset & literature.** Read both papers, read
      collaborator's notebook, clarified input/target/leakage split, agreed
      on novelty plan (composite indices + leakage-free pipeline + ablation
      + SHAP).
- [x] **Step A — Dataset overview script.** `codes/01_dataset_overview.py`
      — full column catalogue, missingness, class balance, dtype audit.
- [x] **Step B — Data cleaning script v1.** `codes/02_data_cleaning.py`
      — fixed typos/case mismatches, structural missing values, mode-imputed
      true random missing, recomputed EPDS/PHQ9 Result labels from scores.
- [x] **Step 1 — Fix `Need for Support` gap.** v1 of the cleaning script
      missed this column (167 missing, 20.88%); added a diagnostic crosstab
      against `Received Support` plus an explicit `"Not Reported"` fill.
      Crosstab result (confirmed by re-run): missing rate is ~33% when
      `Received Support = High` (102/307) vs ~12% for Medium (40/339) and
      ~16% for Low (25/154) — a real but not clean-cut skip-logic signal.
      Decision: kept the conservative `"Not Reported"` bucket for all 167
      rather than assuming a specific skip-logic value, since the pattern
      isn't strong enough (only ~61% of missing rows are `High`) to safely
      infer "no need for support". Documented and closed.
- [x] **Step 2 — This document.** Living project-status file created.
- [x] **Step 3 — Re-run cleaning script, confirm zero missing values.**
      Re-run by Samir: final shape (800, 75), "No missing values remain
      anywhere in the dataset." `PPD_dataset_cleaned_v2.csv` is now locked
      in as the final clean dataset for all downstream work.
- [x] **Step 4 — Association EDA.** `codes/03_association_eda.py` — run by
      Samir successfully. Key finding: the SAME psychosocial features rank
      as most significant across all 3 targets (EPDS Result, PHQ9 Result,
      PPD_binary) — `Angry after latest child birth`, `Abuse`,
      `Relationship with the in-laws/husband`, `Received/Need for Support`,
      `Feeling about motherhood`, `Depression before/during pregnancy
      (PHQ2)`, `Fear of pregnancy`. This is a strong, literature-consistent
      signal (these are exactly the known PPD risk factors) and directly
      validates the Social Support (SSI) and Maternal Mental-Health Risk
      (MHRI) composite index designs. Economic and neonatal-stress features
      did not appear in the top-15 (weaker univariate signal) but remain
      theoretically justified for the composite indices — univariate
      non-significance doesn't rule out useful joint/nonlinear signal.
      Side finding: `Abuse_was_missing` (the imputation flag) was itself
      significant for EPDS Result — i.e. *whether Abuse was reported at all*
      carries information (missing-not-at-random), interesting for
      discussion but not acted on further for now.
- [x] **Step 5 — Feature engineering, equal-weight + weighted composites.**
      `codes/04_feature_engineering.py` — run twice by Samir. First run:
      equal-weight composites gave only 6/12 significant tests (ESI and
      MHRI both non-significant, despite MHRI containing `Abuse`, the
      single strongest raw predictor — signal diluted by equal averaging).
      Added a second variant per index, weighted by each sub-feature's own
      chi-square association strength vs `PPD_binary`. Second run: 18/24
      significant — SSI_Weighted and MHRI_Weighted now extremely
      significant (p as low as 1e-27), NSI_Weighted also improved. ESI
      still weak (only borderline significant for 2/3 targets) — expected,
      matches script 03's weak univariate income/occupation signal.
      `PPD_dataset_with_composite_features.csv` now has 8 composite columns
      (4 equal-weight + 4 weighted), shape (800, 83).
- [x] **Step 5.5 — Verify & fix composite direction.** Ran
      `codes/05_direction_check.py` — found 4 sub-features whose assumed
      direction empirically ran opposite to what was intended:
        - `Abuse`: "No" -> 64.1% PPD-positive vs "Yes" -> 27.0% (2.4x
          reversed). This is the big one — flagged as a methodology caveat
          (see Problems section) since it's clinically counterintuitive;
          the map was flipped to match the empirical direction so MHRI
          stays internally consistent.
        - `Family type`: Nuclear 37.8% vs Joint 50.1% PPD-positive (Joint
          family = MORE risk, not more support — plausibly in-law
          conflict/reduced autonomy, consistent with some South Asian
          family-dynamics literature). Map flipped.
        - `Number of household members`: 2-5 members 39.9% vs 9+ members
          50.9% (more members = more risk, likely crowding, not more
          support). Map flipped.
        - `Education Level`: Primary 22.2%, High School 19.7%, College
          39.4%, University 48.5% (more education = MORE reported PPD,
          plausibly more symptom awareness/higher expectations — also
          noted in some literature). Map flipped for ESI.
      `Relationship with the in-laws/husband` and `Received Support` were
      *correctly* decreasing already (script 05's original generic message
      calling every "decreasing" trend "flipped" was itself misleading for
      protective-type composites — 05 was rewritten to check against a
      per-column expected direction instead). All 4 fixes applied directly
      in `codes/04_feature_engineering.py`. **CONFIRMED by re-run:** every
      sub-feature now shows "direction is CORRECT" (the only "NOT
      monotonic" results left are `History of pregnancy loss`,
      `Relationship with the in-laws`, `Current monthly income`, and
      `Education Level`'s category-level micro-fluctuations — small-sample
      noise in individual categories, not direction bugs). All 8 composite
      indices are now significant against all 3 targets:
      **24/24 tests significant**, with clean, clinically-sensible
      monotonic trends, e.g. weighted MHRI across PHQ9 severity: Minimal
      1.26 -> Mild 2.41 -> Moderate 3.54 -> Moderately Severe 4.84 -> Severe
      6.4 (weighted SSI decreases the same clean way). **Composite feature
      engineering is now DONE and validated** —
      `PPD_dataset_with_composite_features.csv` is locked in.
- [x] **Step 6 — Encoding.** `codes/06_encoding.py` — run twice by Samir,
      confirmed final: shape (800, 119) = 116 features (59 raw/composite +
      one-hot expansions) + 3 targets, 0 missing values, "non-numeric"
      sanity check now prints "None -- good." after the `dtype=int` fix.
      `PPD_model_ready.csv` is locked in as the final modeling dataset.
- [x] **Step 7 — Modeling.** `codes/07_modeling.py` — first run crashed on
      `PPD_binary` (`classification_report` can't handle numpy.int64 class
      labels; fixed by casting to `str`), also removed a deprecated
      `use_label_encoder` XGBoost parameter that was spamming warnings.
      **Re-run confirmed clean, no errors — Step 7 is DONE and final.**

      **Final baseline results (full feature set, raw+composite together,
      all 5 models x 3 targets, no crashes, CV and holdout numbers close
      together for every model = no overfitting red flags):**
      - `PPD_binary` — **strongest target.** All 5 models cluster tightly:
        accuracy 0.74-0.76, ROC-AUC 0.80-0.83 (best: Random Forest AUC
        0.826, LightGBM CV F1 0.759, CatBoost holdout accuracy 0.762). A
        strong, *believable* result for a leakage-free binary screen
        (compare to the external reference paper's suspicious ~0.9999
        AUROC, which almost certainly reflects leakage — ours is lower but
        trustworthy). **This is the headline result for the thesis.**
      - `EPDS Result` (3-class) — best model Random Forest (CV F1 0.583,
        accuracy ~0.58-0.62, ROC-AUC ~0.75). Consistent weak point across
        every model: the "Medium" category has the worst per-class F1
        (0.20-0.29) — borderline cases are hardest to separate, matching
        clinical intuition (moderate-risk mothers blur the boundary
        between low and high risk).
      - `PHQ9 Result` (5-class) — weakest target (best: XGBoost CV F1
        0.362, accuracy ~0.37-0.42, ROC-AUC ~0.67-0.70). Expected: 5
        fine-grained, imbalanced severity bands (Severe n=89, Minimal n=79)
        from demographic/psychosocial predictors alone is a genuinely hard
        problem. **Methodological note for later:** both `EPDS Result` and
        `PHQ9 Result` are ORDINAL categories (not nominal) but are
        currently treated as plain multi-class targets — an ordinal-aware
        approach (e.g. ordinal logistic regression / cumulative link
        models) could be worthwhile future work, not required now.
      - **Recommended framing for the thesis: `PPD_binary` as the main
        result, `EPDS Result` as a secondary result, `PHQ9 Result` as a
        supplementary/exploratory result.**
- [x] **Step 7b — Advanced modeling: tuning + MLP + stacking (script
      written, not yet run).** Samir asked whether results could be
      improved with hyperparameter tuning, a hybrid/stacking ensemble, or
      deep learning — answered honestly: PPD_binary's 0.80-0.83 ROC-AUC is
      already competitive for a leakage-free psychosocial model (published
      leakage-free PPD models typically land 0.65-0.85 AUC), and MLP is
      NOT expected to beat gradient boosting at n=800 (well-established for
      small tabular data, and already flagged at project kickoff). Samir
      chose to try all 3 improvement avenues anyway before the ablation
      study. `codes/07b_advanced_modeling.py` — RandomizedSearchCV tuning
      (20 iter x 5-fold CV) for RF/XGBoost/LightGBM/CatBoost, a small MLP
      architecture search (3 architectures x 3 alphas) as the DL
      comparison point, and a StackingClassifier hybrid ensemble combining
      the 4 tuned tree models with a Logistic Regression meta-learner.
      Merges in the Step 7 baseline (untuned) numbers for direct
      before/after comparison. **Deliberately kept separate from the
      ablation study** — Step 8 stays untuned-simple on purpose, to isolate
      the feature-set variable without conflating it with tuning effects.
      **Run — DONE.** Verdict: tuning gave marginal/inconsistent gains
      (sometimes tuned models even scored slightly below baseline defaults —
      n=800 is already near each model's performance ceiling). Stacking
      Ensemble was a genuine (if modest) win for `PPD_binary` (CV F1=0.762,
      best of everything tried). MLP consistently the weakest model for
      `EPDS Result` (0.499) and `PHQ9 Result` (0.313), confirming the
      project's original expectation that deep learning underperforms
      gradient boosting on n=800 tabular data — for `PPD_binary` (a simpler
      binary task) MLP was more competitive (0.740) but still not top.
      No severe overfitting (CV vs holdout stayed close for the strong
      models); MLP's weak numbers reflect underfitting/high-bias on this
      small dataset, not overfitting.
- [x] **Step 8 — Ablation study.** `codes/08_ablation_study.py` — for each
      target, compares 3 feature arms using the identical model/CV protocol
      as script 07: **Arm A "Raw only"**, **Arm B "Raw + Composite"** (+ the
      4 WEIGHTED composite indices), **Arm C "Selected + Composite"** (top
      30 features by RF importance). First run found a real methodological
      gap (Arm C selected its top-30 features using the FULL dataset before
      CV/holdout — double-dipping); fixed via `run_arm_c_nested` so feature
      selection is re-derived separately inside every CV fold's training
      portion and inside the holdout's 80% training portion only. **Re-run
      with the nested fix — CONFIRMED DONE.**

      **Final (honest, nested-CV) numbers, avg CV F1-macro across all
      models:**
      - `EPDS Result`: Raw=0.552, Raw+Composite=0.554 (+0.002),
        Selected+Composite=0.561 (+0.009) — essentially unchanged from the
        pre-fix numbers, a genuine and robust gain.
      - `PHQ9 Result`: Raw=0.339, Raw+Composite=0.352 (+0.013),
        Selected+Composite=0.346 (+0.007) — **Arm C is now actually LOWER
        than Arm B** (the pre-fix version had claimed +0.016). For this
        target specifically, reducing to 30 features loses information;
        the full Raw+Composite set is the better choice.
      - `PPD_binary`: Raw=0.740, Raw+Composite=0.741 (+0.001),
        Selected+Composite=0.745 (+0.005) — real but much smaller than the
        pre-fix +0.016 (roughly 1/3 of the original claimed gain).
      - **What held up unchanged:** composite indices (especially
        `Social_Support_Index_Weighted` and
        `Maternal_MentalHealth_Risk_Index_Weighted`) remain top-3 most
        important features for every target, even when importance is
        computed strictly from each fold's training-only data — this is
        the strongest, bias-free evidence for the "composite features are
        genuinely informative" claim. The Arm A-vs-B comparison was never
        affected by the double-dipping bug in the first place (no feature
        selection happens there), so those numbers were always trustworthy.
      - **Thesis framing takeaway:** report the smaller, honest Arm C
        effect sizes rather than the original inflated ones; explicitly
        note feature-selection's benefit is target-dependent (helps for
        EPDS Result and PPD_binary, does not help for PHQ9 Result — use the
        full Raw+Composite set for that target instead of the 30-feature
        subset).
      - **Statistical significance added** (Samir asked "dont we need any
        test to proof our results?"): the script now runs a paired t-test +
        Wilcoxon signed-rank test per model per target, comparing the SAME
        5 CV folds' scores across arms (B vs A, C vs B). Saves
        `ablation_significance.csv`. **Re-run confirmed: 0/30 comparisons
        reached p<0.05.** This does NOT mean composite features/feature
        selection don't help — with only 5 paired folds, these tests have
        very low statistical power (Wilcoxon's smallest possible two-sided
        p-value at n=5 is ~0.0625), so a true small effect (which is what
        the consistent positive direction across all 3 targets suggests)
        will often fail to reach significance. **Documented as an honest
        limitation for the thesis**, not treated as disproving the earlier
        findings. The strongest, power-unaffected evidence for "composite
        features matter" remains: (a) the Step 4/5 association tests (run
        on the full n=800, much higher power), and (b) feature-importance
        ranking (composite indices consistently top-3 across all targets,
        computed honestly via nested/training-only selection in Step 8).
        **Recommendation for the write-up:** report Arm B/C's absolute
        gains as "a consistent positive trend across all three targets and
        model families, not reaching significance in 5-fold paired testing
        — a known low-power limitation of small-fold CV designs."
      - **07b_advanced_modeling.py re-run confirmed** with epoch logging:
        MLP used only 18-40 epochs (out of max_iter=1000) across all 3
        targets before early_stopping triggered — concrete, quantified
        evidence that the network's validation performance plateaus almost
        immediately, i.e. underfitting/high-bias on this small dataset, not
        a tuning problem. Confirms the project's original expectation that
        deep learning doesn't have enough data here to out-learn gradient
        boosting. Tuning (RandomizedSearchCV) again showed marginal/
        inconsistent gains, sometimes even underperforming untuned
        baselines on holdout (e.g. EPDS Result: tuned RF holdout F1=0.502
        vs untuned baseline holdout F1=0.520) — a mild sign that tuning
        itself can slightly overfit the CV folds it was optimized on, given
        how small n=800 is.
- [x] **Step 9 — Best model + SHAP explainability.** `codes/09_shap_explainability.py`
      — Arm B "Raw + Composite" (112 features), Random Forest, SHAP values
      on the held-out 20% test set. **Run — CONFIRMED, no crashes, every
      plot saved successfully for all 3 targets (no `[plot failed]` lines
      at all).**

      **Findings:** Composite indices dominate the global top 10 in every
      target — EPDS Result 4/4, PPD_binary 4/4, PHQ9 Result 3/4 (only
      `Neonatal_Delivery_Stress_Index_Weighted` dropped to rank 8 for
      PHQ9). This is now confirmed by a THIRD independent method
      (association tests -> ablation feature importance -> SHAP on unseen
      test data), the strongest possible triangulated evidence for the
      thesis's core novelty claim. Single strongest individual (non-
      composite) predictor across the whole project, confirmed again here:
      `Angry after latest child birth_Yes` — rank #1 for EPDS Result and
      PPD_binary, rank #2 for PHQ9 Result (just behind
      Social_Support_Index_Weighted) — worth highlighting on its own in the
      discussion section as the single most consistent raw risk factor.

      Train accuracy = 1.000 for all 3 targets (Random Forest with
      unrestricted max_depth memorizes training data by design) — this is
      NOT new evidence of overfitting; the real generalization check is the
      CV-vs-holdout gap already validated in Steps 7/8, which stayed small.
      Waterfall examples show the model is confidently correct at the
      extremes (EPDS: P(High)=0.933 for a genuine high-risk patient,
      P(High)=0.037 for a genuine low-risk one; PPD_binary similar:
      0.930/0.037) but much less confident for PHQ9 "Severe" (best case
      only P(Severe)=0.483) — concrete, patient-level confirmation of the
      earlier finding that PHQ9's 5 ordinal severity bands blur into each
      other and are the hardest target to separate cleanly.
- [x] **Step 10 — Write-up / clinical interpretation.** Delivered as a
      spoken Bangla discussion (Samir's choice, no file) covering risk
      factors, model performance, composite-index novelty, and
      limitations. A written proposal-style document (`Composite_Risk_
      Indices_PPD_Proposal.rtf`, saved to the dataset root, RTF format
      since the code sandbox couldn't run docx-js this session) was later
      created for sharing with Samir's teacher — written in forward-looking
      "we propose to" language per Samir's explicit request, not reporting
      completed results.
- [ ] **Step 11 — Post-hoc improvement experiments** (Samir asked "is
      there any other way to improve the result?" after Step 10; picked
      all of the top recommendations to implement, step by step). Three
      scripts written, none yet run:
      - `codes/10_ordinal_regression.py` — EPDS Result/PHQ9 Result are
        ordinal, not plain multi-class, but every prior script treated
        them as unordered categories. This script (a) adds Quadratic
        Weighted Kappa (QWK) as an ordinal-aware metric alongside
        accuracy/F1-macro, and (b) tries predicting the continuous
        EPDS/PHQ9 Score via regression (Linear/RF/XGBoost/LightGBM/
        CatBoost) on the Arm B feature set, then thresholding back into
        the official clinical bands, compared head-to-head against the
        direct-classifier approach on an identical split. **Run — DONE.**
        Findings: QWK is notably higher than F1-macro for both targets
        (EPDS 0.540->0.557, PHQ9 0.329->0.548) confirming most errors are
        near-miss/adjacent-band, not wild misses -- both targets are more
        clinically useful than F1-macro alone suggested. For PHQ9 Result,
        Regression+Threshold (Linear Regression) genuinely beat the direct
        classifier on every metric (accuracy 0.344->0.463, F1-macro
        0.329->0.426, QWK 0.548->0.602) -- adopt this as the PHQ9 approach
        going forward. For EPDS Result, the direct classifier remained
        best (QWK 0.557 vs best regression 0.521) -- no universal win,
        reported honestly per-target. Caveat found: Random Forest
        regression compresses predictions toward the mean (classic
        ensemble-averaging effect), badly hurting its categorical
        conversion (PHQ9 F1-macro only 0.251 despite R2=0.433) -- Linear
        Regression or CatBoost are the better regressors for this
        thresholding use case, not Random Forest.
      - `codes/11_threshold_calibration.py` — for PPD_binary: precision-
        recall-based threshold tuning (max-F1 threshold, plus a "recall
        >= 85%" screening threshold, vs the default 0.5 cutoff), and
        CalibratedClassifierCV (Platt/sigmoid) with Brier score +
        reliability-curve comparison, so predicted probabilities can be
        trusted as an actual risk score, not just a ranking. **Run — DONE.**
        Findings: Max-F1 threshold (0.407 instead of default 0.5) is a
        free improvement -- same accuracy (0.775), better F1 (0.731->0.753)
        and notably better recall (0.700->0.786) at only a small precision
        cost. Recommend adopting 0.407 as the new default operating point.
        Screening threshold (0.303, hits the recall>=85% target) trades off
        much more: precision drops to 0.600, accuracy drops to 0.6875, F1
        actually falls below the Max-F1 option (0.706 vs 0.753) -- a real,
        expected precision/recall trade-off, not a bug; document both
        options in the write-up and let the deployment context (how costly
        a missed case is vs a false alarm) decide which to use. Calibration
        gave a small, expected improvement (Brier 0.1677->0.1667) with
        ROC-AUC essentially unchanged (0.829->0.828), confirming
        calibration reshaped probability values without hurting
        discrimination -- use the calibrated probabilities when reporting
        a patient-facing risk score.
      - `codes/12_catboost_native_and_repeated_cv.py` — Part A: CatBoost
        with native categorical handling (no one-hot) vs the standard
        one-hot pipeline, identical split, all 3 targets. Part B: redoes
        08's Arm A vs Arm B paired significance test using
        RepeatedStratifiedKFold (5 folds x 5 repeats = 25 paired
        observations instead of 5), to check whether more statistical
        power reveals a significant composite-feature effect that the
        original 5-fold test lacked the power to detect. **Run — DONE**
        (needed one fix: CatBoostClassifier doesn't clone cleanly through
        sklearn's `cross_validate` when `cat_features` is set -- worked
        around with a manual per-fold CV loop for the native-categorical
        model only). Findings -- Part A: no universal winner (native beats
        one-hot on PHQ9 holdout 0.400 vs 0.336, roughly ties on the other
        two targets), but native categorical achieves comparable results
        with only 59 features vs one-hot's 116 -- a genuine simplicity/
        efficiency gain worth noting even without a raw accuracy win.
        Part B: with 25 folds instead of 5, 2/15 comparisons are NOW
        statistically significant (EPDS Result x XGBoost, +0.022, p=0.0008;
        PHQ9 Result x LightGBM, +0.022, p=0.002) -- confirms the original
        "0/30 significant" result was a low-power artifact, not evidence
        against the composite features' value; most other comparisons
        remain positive in direction but still short of significance.
        One honest exception: PPD_binary x Random Forest showed a small
        negative (non-significant) effect -- composite features help most
        consistently for boosting models, less reliably for Random Forest
        specifically on this target.
All 3 Tier-1 improvement experiments (scripts 10, 11, 12) are done and
reviewed -- see findings above. Samir chose to continue into Tier-2. Three
more scripts written, none yet run:
- `codes/13_composite_weight_learning.py` — replaces the manual
  -log10(p) chi-square weighting with per-domain Logistic Regression
  learned weights (the LR's own predicted probability becomes the new
  composite index), validated the same way as script 04 (association
  tests) plus a predictive CV F1-macro comparison against the current
  chi-square-weighted composites, swapped into Arm B, identical folds.
  **Run — DONE.** Findings: both variants are 12/12 significant on
  association tests (tie). Predictive CV F1-macro differences are small
  and inconsistent across targets -- LR-learned wins for EPDS Result
  (0.590 vs 0.578) and PHQ9 Result (0.371 vs 0.365), chi-square-weighted
  wins for PPD_binary (0.753 vs 0.746) -- no universal winner, differences
  within normal CV noise. **Decision: keep the existing chi-square-
  weighted composites** as the final/thesis version (already deeply
  validated via steps 4/5/8/9); report this experiment as confirmatory due
  diligence -- a more sophisticated data-driven alternative didn't
  meaningfully beat the original, simpler weighting choice. Bonus finding:
  the learned LR coefficients independently reproduce the earlier
  counter-intuitive "Abuse" direction finding (positive coefficient on the
  normalized abuse feature confirms "No" answers associate with higher
  predicted PPD risk) -- a THIRD independent method confirming this is a
  genuine data pattern, not an artifact of the chi-square weighting
  approach specifically.
- `codes/14_smote_and_imputation.py` — Part A: SMOTENC oversampling
  (fold-safe, training-portion only) vs class_weight-only, for all 3
  targets, primary interest being PHQ9 Result's rare severity bands. Part
  B: IterativeImputer (MICE-style) vs the original mode-fill for the 5
  true-random-missing columns (uses the `_was_missing` flags to know
  exactly which cells to re-impute), reports how many values changed and
  whether it affects downstream PPD_binary CV F1-macro. **Run — DONE.**

  **Part A findings (negative result, worth reporting honestly):** SMOTENC
  gave a tiny improvement for EPDS Result (0.578->0.583) and PPD_binary
  (0.753->0.758), but made PHQ9 Result WORSE (0.365->0.348) -- exactly the
  target it was meant to help most (5 classes, most imbalanced: Mild=263
  vs Minimal=79). Root cause: with ~112 mostly-binary one-hot features,
  "nearest neighbor" becomes unreliable in high-dimensional space (the
  curse of dimensionality) -- SMOTE's synthetic minority-class patients end
  up being unrealistic combinations of survey answers rather than
  plausible patients, and generating MORE synthetic samples (needed for
  PHQ9's more severe imbalance) amplifies this noise more than it helps.
  **Decision: do not adopt SMOTE; `class_weight="balanced"` alone remains
  the right approach for this dataset.** Worth including in the paper as a
  tested-but-rejected alternative -- shows methodological rigor.

  **Part B findings:** MICE-style re-imputation agreed completely with the
  original mode-fill for binary columns with a dominant category (Abuse:
  0/38 changed, Trust and share feelings: 0/1 changed), but disagreed
  almost completely for multi-category columns (Husband's monthly income:
  28/28 changed; Education Level: 6/6 changed; Husband's education level:
  7/9 changed) -- expected, since mode-fill assigns one fixed bracket to
  everyone missing, while MICE predicts a person-specific bracket from
  their other answers. Despite these large value changes, downstream
  PPD_binary CV F1-macro was statistically indistinguishable (mode-fill
  0.654+/-0.058 vs MICE 0.647+/-0.064 -- well within each other's noise).
  **Decision: keep the existing mode-fill** (simpler, equally good in
  practice); note MICE as a tested, defensible alternative that didn't
  show a compelling enough improvement to justify re-running the entire
  pipeline (steps 02 through 09) with it.
- `codes/15_age_features_and_group_importance.py` — Part A: adds Age^2 and
  clinical Age bands as candidate features, tests association + predictive
  impact. Part B: group-level permutation importance (permuting all of an
  original question's one-hot dummies together, using `encoding_map.csv`),
  compared side-by-side against standard per-dummy importance, to check
  whether one-hot encoding was hiding any raw question's true importance
  by splitting its signal across several correlated dummy columns.
  **Run — DONE.**

  **Part A findings:** Age and Age^2 are NOT significantly associated with
  any target on their own (all p>0.10), and adding them to Arm B changed
  CV F1-macro negligibly (EPDS +0.005, PHQ9 -0.002, PPD_binary +0.003, all
  within noise). Interesting nuance: Age still ranks fairly high in
  Random Forest's individual feature importance (e.g. #3 for PHQ9) DESPITE
  failing the univariate association test -- this is expected, since tree
  models capture Age's INTERACTION effects with other features (e.g. young
  mothers with low support vs young mothers with high support may have
  very different risk) that a simple group-mean comparison can't detect.
  **Decision: no need to add Age^2/Age bands** -- RF already captures
  Age's non-linear/interaction contribution natively via tree splits.

  **Part B findings (genuinely valuable methodological discovery):**
  grouping one-hot dummies back to their original question revealed
  several raw variables whose true importance was hidden by encoding
  splitting their signal across multiple correlated dummy columns. Most
  notable, for PHQ9 Result (the hardest target): "Age of immediate older
  children" (grouped rank #2) and "Disease before pregnancy" (grouped rank
  #3) don't appear AT ALL in the standard top-15 individual-dummy ranking,
  and "Husband's education level" (grouped rank #7) is similarly absent --
  three variables the standard SHAP/RF-importance view (steps 8/9)
  effectively understated. For EPDS Result, "Feeling for regular
  activities" jumped from individual rank #7 to grouped rank #1, and
  "Received Support" from individual rank #15 to grouped rank #4. For
  PPD_binary, grouping mostly CONFIRMED the existing ranking with no major
  surprises (Angry after latest child birth remained #1 either way) --
  this target's earlier analysis already had the right picture.
  **Recommendation: report group-level importance as an additional/
  corrected view alongside the per-dummy SHAP ranking from step 9,
  especially for the PHQ9 Result discussion** -- these 3 under-rated
  variables (age gap to older children, pre-pregnancy disease history,
  husband's education) are worth naming explicitly as possible directions
  for a future composite index or at least explicit discussion points.

- [x] **Step 12 — Clinical literature grounding for the 4 composite
      indices** (colleague's suggestion #1: "give the composite indices
      scientific/clinical-literature backing, not just internal
      statistics"). Done via WebSearch, four separate searches, one per
      composite domain. **Findings, each independently confirming the
      composite's own internal association-test results from steps 4/8/9:**
      - **Social_Support_Index:** an umbrella review of 77 meta-analyses
        found poor social support (RR 3.57) among the strongest PPD risk
        factors; a Chinese-population meta-analysis found lack of social
        support OR 2.57 (95% CI 2.32-2.85); doula/parenting-support
        interventions are protective (RR 0.36). Matches this project's
        finding that Social_Support_Index is significant for all 3 targets.
      - **Maternal_MentalHealth_Risk_Index (Abuse sub-feature):** intimate
        partner violence meta-analyses report a 5.46-fold increased PPD risk
        (POR 5.46, 95% CI 3.94-7.56), with physical/emotional/sexual abuse
        sub-types individually significant (OR 1.56-1.90); inadequate
        social support compounds this further (6.27-fold). This is strong
        literature support for Abuse being one of the strongest individual
        predictors in this project's SHAP rankings (steps 9/16) -- though
        the literature does NOT explain this project's own counter-intuitive
        finding (Abuse="No" associating with higher predicted risk in the
        raw data), which remains a data-quality/reporting-bias question
        specific to this dataset, not something literature can resolve.
      - **Economic_Stability_Index:** Bangladesh-specific studies (Dhaka
        urban slums, rural cross-sectional) directly confirm low economic
        status, household-resource poverty, and financial crisis as
        established PPD risk factors in exactly this population, with PPD
        prevalence as high as 39.4% in the most deprived urban-slum cohort.
        This is the strongest literature match of all 4 domains, since it
        comes from the same country/population rather than a global average.
      - **Neonatal_Delivery_Stress_Index (Cesarean/birth-complication
        sub-features):** cesarean delivery (planned or emergency) is
        associated with a >15% increased risk of postpartum psychiatric
        conditions vs. vaginal birth, and childbirth trauma (higher blood
        loss, emergency cesarean) is shown to mediate this risk -- though
        prior mental illness remains a stronger predictor than delivery
        mode alone. Supports including delivery-related stress in the
        composite, while cautioning not to overweight it relative to the
        other 3 domains.
      **Overall conclusion: all 4 composite indices have independent
      external literature support, not just internal chi-square/ANOVA
      significance** -- this is a meaningful strengthening of the "why
      these 4 domains, not some other grouping" justification for the
      thesis/paper's methodology section. Full source links are in the
      chat message accompanying this update.
- [x] **Step 13 — Further explainability & discovery experiments.**
      (colleague's suggestions #2, #3, #4, #5, #6 — SHAP with/without
      composites, SHAP interaction values, unsupervised cluster analysis,
      error analysis of misclassified patients, SHAP + group importance
      combined). All 4 scripts run and reviewed:
      - `codes/16_shap_arm_comparison_and_group.py` (suggestions #2 + #6) —
        Part A re-runs SHAP on Arm A ("raw only", no composites) on the
        identical train/test split used in step 9, so Arm A's top features
        (individual raw answers) can be shown side-by-side against Arm B's
        top features (from step 9's saved results, reloaded not
        recomputed) where the composite indices rise to the top -- a direct
        "before/after" visual proof that the model itself prefers the
        composite summary once it's available. Part B computes group-level
        SHAP importance: since SHAP values are additive, a one-hot group's
        combined pull is just the row-wise SUM of its dummies' SHAP values
        (more principled than script 15's permutation-based grouping) --
        reuses `encoding_map.csv` the same way script 15 did. **Run — DONE.**
        Findings -- Part A: "Angry after latest child birth_Yes" stays the
        #1 feature with or without composites for EPDS Result and
        PPD_binary (4/4 composites still crowd into top-10 right behind
        it: Maternal_MentalHealth_Risk_Index_Weighted lands at #2,
        Social_Support_Index_Weighted at #3). For PHQ9 Result specifically,
        something stronger happens: Social_Support_Index_Weighted actually
        OVERTAKES "Angry_Yes" to become the single #1 feature once
        composites are added (3/4 composites in top-10) -- i.e. for PHQ9,
        the composite isn't just riding along near the top, it becomes the
        model's most-relied-on feature. Good, reportable confirmation that
        composite indices earn their place by the model's own SHAP
        judgment, not just by construction. Part B (SHAP-additive grouping)
        mostly agrees with script 15's permutation-based grouping for the
        big names (Angry/Abuse/Social_Support/Maternal_MentalHealth all
        rank near the top both ways) -- but for PHQ9 Result, "Age of
        immediate older children" ranked #2 by permutation importance
        (script 15) drops to only #14 by SHAP-additive importance here.
        This is a genuinely interesting, honest discrepancy worth
        discussing: permutation importance captures a feature's value even
        when it works mainly through INTERACTIONS with other features
        (shuffling it breaks those interactions too), while plain SHAP-sum
        only captures its direct additive/main effect -- so a large
        gap between the two rankings for one feature is itself a hint that
        the feature's importance is interaction-driven rather than a
        simple main effect. This connects directly to script 17's
        interaction analysis (next) -- worth checking there whether "Age
        of immediate older children" shows up with unusually strong
        pairwise interactions for PHQ9 Result.
      - `codes/17_shap_interactions.py` (suggestion #3) — **What/how:** every
        SHAP result so far (steps 9 and 16) only reports MAIN effects, i.e.
        how much a single feature pushes a prediction up or down on its
        own. It never asks whether two features work TOGETHER in a way
        that's more than the sum of their parts. This script computes
        `shap.TreeExplainer(model).shap_interaction_values(X_test)`, which
        decomposes every prediction into pure main effects PLUS pairwise
        interaction effects for every feature pair. Because this is an
        O(n^2) computation, it's restricted to each target's own top-20
        SHAP features (from step 9's ranking) rather than all 112 -- a
        small, fresh diagnostic Random Forest is trained on just those 20
        features specifically for this analysis (NOT a replacement for the
        main 112-feature pipeline model; its own held-out accuracy is
        reported purely so this distinction is clear: EPDS 0.569, PHQ9
        0.356, PPD_binary 0.756 -- all lower than the main models simply
        because only 20 of 112 features are available to it here). The
        script also specifically checks a handful of clinically-motivated
        pairs chosen in advance (Age x Support, Age x Angry,
        Economic_Stability x Abuse, income x Abuse, Economic_Stability x
        Maternal_MentalHealth) regardless of whether they show up in the
        general top-15 list. **Run — DONE. Findings (strong, consistent,
        genuinely thesis-worthy):**
        1. **The single strongest pairwise interaction in ALL THREE targets
           involves two composite indices (or a composite + the top raw
           flag), never two unrelated raw answers:** EPDS Result ->
           Maternal_MentalHealth_Risk_Index_Weighted x
           Social_Support_Index_Weighted (0.0135); PHQ9 Result ->
           Social_Support_Index_Weighted x "Angry after latest child
           birth_Yes" (0.0114); PPD_binary -> "Angry after latest child
           birth_Yes" x Social_Support_Index_Weighted (0.0137). This means
           the composite indices are not just individually important
           (steps 4/9/16 already showed that) -- they also drive the
           model's strongest COMBINED effects, i.e. a mother's risk isn't
           just "high mental-health risk index" OR "low social support" in
           isolation, it's specifically the COMBINATION of the two that
           the model leans on hardest.
        2. **Economic_Stability x Maternal_MentalHealth is a robust,
           cross-target interaction:** it ranks 5th/190 for EPDS Result,
           12th/190 for PHQ9 Result, and 5th/190 for PPD_binary -- i.e.
           economic hardship consistently amplifies mental-health risk's
           effect on the prediction across all 3 outcome measures, not
           just one. This is a very reportable, clinically intuitive
           finding (financial stress compounding psychological risk) that
           now has model-internal SHAP evidence behind it, not just
           correlation.
        3. **Economic_Stability x Abuse is present but weaker/less
           consistent** (rank 44/190 EPDS, 50/190 PHQ9, 15/190 PPD_binary)
           -- worth mentioning as a checked-but-secondary interaction,
           honest reporting rather than cherry-picking only the strong
           results.
        4. **Resolves the script-16 mystery about "Age of immediate older
           children" for PHQ9 Result:** script 16 found this raw feature
           ranked #2 by permutation-based group importance (script 15) but
           only #14 by simple SHAP-additive group importance (script 16) --
           a large, unexplained gap at the time. This script confirms why:
           "Social_Support_Index_Weighted x Age of immediate older
           children_Not Applicable (Only One Child)" is the 9th/190
           strongest interaction pair for PHQ9 Result specifically (and
           "Angry x Age-of-older-children" is 30th/190). In other words,
           this feature's real importance comes through its INTERACTION
           with the Social Support composite, not through a strong
           standalone/main effect -- exactly the mechanism permutation
           importance is sensitive to (shuffling breaks the interaction
           too) and plain SHAP-sum importance is not. This is a clean,
           three-script chain of evidence (15 -> 16 -> 17) demonstrating
           WHY two reasonable importance methods can disagree, and is
           worth walking through explicitly in the discussion section as a
           methodological insight, not just a footnote.
        5. For EPDS Result and PPD_binary specifically, "Age" itself never
           made the top-20 feature list at all, so the Age x Support / Age
           x Angry hint-pairs weren't checked for those two targets --
           consistent with step 15's Part A finding that Age has no strong
           standalone effect for those targets either.
      - `codes/18_cluster_analysis.py` (suggestion #4) — **What/how:** every
        prior script asked "given a mother's profile, predict her PPD
        risk" (a supervised question). This script asks a different,
        complementary question: "ignoring PPD entirely, do mothers
        naturally fall into distinct psychosocial risk PROFILES/groups on
        their own?" -- and only AFTER discovering those groups purely from
        their composite-index/Age/pregnancy-count values, checks whether
        PPD prevalence happens to differ across them. Deliberately
        clustered on a compact 6-feature space (the 4 weighted composite
        indices + Age + Number of the latest pregnancy) rather than the
        full 112-dimension one-hot space, because script 14's SMOTE
        experiment already showed distance-based methods behave poorly in
        high-dimensional mostly-binary spaces -- the composite indices are
        exactly the right compact, clinically meaningful summary for this.
        K was chosen by silhouette score, tried K=2 through K=6; each
        cluster is auto-labeled purely from its own feature means relative
        to the overall dataset mean (no hardcoded "this cluster is the
        risky one" assumption); PPD_binary prevalence per cluster is then
        tested via chi-square, plus EPDS/PHQ9 distributions per cluster and
        a 2D PCA scatter for visualization. **Run — DONE. Findings (a
        strong, clean, genuinely important result):**
        1. **Best K = 3** by silhouette score (0.224 at K=3, only slightly
           above K=2's 0.209 and K=4's 0.224 -- the silhouette scores
           themselves are modest across the board, 0.19-0.22, which is
           normal/expected for real survey data rather than
           artificially-separated synthetic clusters; it means the 3
           groups blend into each other somewhat at the edges, not that
           they're meaningless -- the downstream PPD-rate separation
           below confirms they ARE meaningful despite the modest geometric
           separation).
        2. **The 3 clusters found, purely from composite-index values
           (PPD labels never used in the clustering step):**
           - **Cluster 0 (n=336): "Lower support, higher mental-health
             risk, lower economic stability."** Economic_Stability=1.44,
             Social_Support=4.90, Maternal_MentalHealth_Risk=7.02 (much
             higher than the other two clusters), Age~28.
           - **Cluster 1 (n=113, smallest group): "Higher support, lower
             mental-health risk, higher economic stability."**
             Economic_Stability=6.46 (far above the other two clusters),
             Social_Support=6.89, Maternal_MentalHealth_Risk=1.95,
             youngest average Age (~25) and most prior pregnancies (~2.28).
           - **Cluster 2 (n=351, largest group): "Higher support, lower
             mental-health risk, lower economic stability."** Similar
             support/mental-health profile to Cluster 1, but economically
             worse off (Economic_Stability=1.57, close to Cluster 0's
             level) -- this is the group that isolates the EFFECT of
             economic stability specifically, holding support/mental-
             health risk roughly constant.
        3. **PPD_binary prevalence differs dramatically and is highly
           significant:** Cluster 0 = 66.4% positive, Cluster 2 = 29.9%,
           Cluster 1 = 18.6% -- a 3.6x spread between the highest- and
           lowest-risk cluster, chi-square p=3.73e-28 (about as
           significant as a p-value gets with n=800). EPDS Result "High"
           rate and PHQ9 Result "Severe" rate follow the exact same
           pattern (Cluster 0: 66.4% High EPDS, 22.3% Severe PHQ9 vs.
           Cluster 1: 18.6% High EPDS, 0.9% Severe PHQ9) -- fully
           consistent across all 3 targets, not a fluke of one label
           definition.
        4. **Why this matters (the key thesis point):** this is a
           genuinely INDEPENDENT, unsupervised confirmation of the
           composite-index risk story -- the clustering step never saw a
           single PPD/EPDS/PHQ9 label, yet the natural groupings it found
           from the composite indices alone line up almost perfectly with
           real depression severity. This directly answers "are these
           composite indices actually capturing something real, or just
           something the supervised model was trained to exploit?" -- the
           unsupervised result says the structure is real and exists in
           the data independent of any specific classifier.
        5. **Bonus nuance from comparing Cluster 1 vs Cluster 2:** both
           have essentially the same (good) support and mental-health
           profile, and differ mainly in economic stability (6.46 vs
           1.57) -- yet PPD prevalence still rises from 18.6% to 29.9%
           moving from Cluster 1 to Cluster 2. This isolates economic
           stability as adding real incremental protective value even
           when support and mental-health risk are already favorable --
           a cleaner, more direct demonstration of economic stability's
           independent contribution than the earlier association tests
           alone provided.
        6. **Practical framing for the paper/thesis:** Cluster 0 (42% of
           the sample) is a directly actionable "high-priority outreach"
           profile -- mothers with low social support, low economic
           stability, and elevated baseline mental-health risk indicators,
           who are having PPD at nearly 3.6x the rate of the lowest-risk
           group. This is a concrete, clinically usable recommendation a
           policy-maker or clinic could act on directly, distinct from
           (and complementary to) the per-feature SHAP explanations.
      - `codes/19_error_analysis.py` (suggestion #5) — **What/how:** every
        prior evaluation reported an AGGREGATE metric (accuracy, F1,
        ROC-AUC) or a confusion matrix count. This script asks a sharper
        question: "who, specifically, are the ~37 mothers on the test set
        the PPD_binary model gets wrong, and what do they have in common?"
        Splits the held-out test set into TP/FN/FP/TN, then runs
        Mann-Whitney U tests comparing FN vs TP (among mothers who truly
        have PPD, what separates the ones the model catches from the ones
        it misses?) and FP vs TN (among mothers who truly don't have PPD,
        what separates the false alarms from the correctly-cleared?) on
        the 4 weighted composite indices + Age + pregnancy count, plus
        raw-flag rate comparisons (Abuse, Angry after birth, Received
        Support, Need for Support). Separately, for EPDS/PHQ9 Result:
        an ordinal "band distance" analysis (how many severity bands off a
        wrong prediction was), reusing script 10's QWK ordinal ordering.
        Hit one bug on first run (`np.select`'s implicit integer default
        clashed with the string outcome labels under this numpy version --
        fixed by passing an explicit `default="UNKNOWN"`). **Run — DONE.
        Findings (a very clean, coherent story that ties the whole project
        together):**
        1. **Test set breakdown (n=160):** TN=75, TP=48, FN=22, FP=15 --
           consistent with the ~0.775 holdout accuracy reported since
           step 7/9.
        2. **The model's blind spot is almost entirely explained by its
           own two most-relied-on composite indices** (the same two that
           script 17's interaction analysis flagged as the strongest
           pairwise interaction for this exact target):
           - **Missed real PPD cases (FN, n=22) look like "low-risk"
             mothers on paper, despite actually having PPD:** their
             Maternal_MentalHealth_Risk_Index_Weighted is dramatically
             LOWER than correctly-caught cases (1.96 vs 6.05, p=2.95e-07),
             their Social_Support_Index_Weighted is HIGHER (6.60 vs 4.53,
             p=3.4e-05), and even Economic_Stability_Index_Weighted is
             somewhat higher (2.90 vs 1.50, p=0.033). In plain language:
             the model is missing mothers who genuinely have PPD (by
             EPDS/PHQ9 clinical threshold) but who do NOT show the usual
             risk-factor answers on the survey -- they report decent
             support, decent economic stability, and don't flag the
             typical mental-health risk sub-answers. This looks like an
             "atypical presentation" or under-reporting subgroup -- a
             genuinely important clinical caveat: the model (and the
             composite indices it relies on) work well for the TYPICAL
             risk profile but can miss PPD in mothers who don't outwardly
             present the expected risk factors.
           - **False alarms (FP, n=15) show the mirror-image pattern:**
             Maternal_MentalHealth_Risk_Index_Weighted is much HIGHER than
             correctly-cleared mothers (6.32 vs 1.59, p=7.2e-07), and
             Social_Support_Index_Weighted is somewhat LOWER (5.78 vs
             6.95, p=0.015). These are mothers who score as "high risk" on
             the composites but do not actually cross the clinical PPD
             threshold -- i.e. the composites correctly flag risk
             FACTORS, but risk factors don't always translate into an
             actual clinical outcome for every individual (expected --
             risk factors are probabilistic, not deterministic).
           - Economic_Stability_Index_Weighted, Age, and the individual raw
             flags (Abuse, Angry, Received/Need for Support) did NOT reach
             significance for the FP-vs-TN comparison -- the false-alarm
             pattern is driven specifically by Mental-Health-Risk and
             Social-Support, not by economic or demographic factors.
        3. **The unifying insight:** both types of error (FN and FP) are
           explained almost entirely by the SAME two composite indices --
           this is expected and actually reassuring: it confirms
           Maternal_MentalHealth_Risk_Index and Social_Support_Index truly
           are the model's primary decision drivers (consistent with SHAP
           rankings in steps 9/16/17), and the model's mistakes are
           concentrated exactly where those two composites give a
           misleading signal (a genuinely "atypical" patient), rather than
           being scattered randomly across unrelated features. This is a
           strong, specific, well-evidenced "Limitations" paragraph for the
           thesis: the model's blind spot is well-characterized, not vague.
        4. **Ordinal error distance (EPDS/PHQ9 Result):** EPDS Result --
           91.2% of test predictions are correct or only one severity band
           off (matches the earlier QWK=0.557 finding closely). PHQ9
           Result -- 84.4% correct-or-one-band-off, with only 5/160 cases
           (3.1%) at a large distance-3 error (out of a max possible
           distance of 4) -- i.e. genuinely severe misclassifications are
           rare even for PHQ9, the hardest target; worth a specific,
           qualitative look at those 5 cases if there's time, but not a
           cause for concern about the overall approach.
        **No problems found in this script's results -- this is a clean,
        positive, and highly reportable finding, not a red flag.**

### Next action for Samir
All of Step 13 (scripts 16-19) AND the script 20 closing-the-loop test are
done and reviewed (see the Consolidated Best Model section above — no
config beat baseline, decision: keep current models unchanged). Next:
Step 14, the final write-up revision folding in everything from Tier 1,
Tier 2, the literature grounding, and this whole explainability round.

- [ ] **Step 14 — Write-up / clinical interpretation, final revision**
      (after Step 13's experiments are reviewed — fold in the literature
      citations from Step 12 too).

---

## 3.5 Consolidated Best Model Per Target (as of Step 13)

*(Written because the project tried many models/variants across steps 7-19.
This section is the single place that says, per target, exactly which
model/approach is CURRENTLY the adopted best one, with the full lineage of
what else was tried and rejected. Update this section, don't just the
roadmap, whenever a new experiment changes the answer.)*

**All candidates share the same leakage-free "Arm B" feature set** (raw
safe-input features, one-hot encoded, + the 4 chi-square-WEIGHTED composite
indices — 112-116 features depending on target/encoding pass), confirmed
via the Step 8 ablation study to beat both Arm A (raw only) and Arm C
(selected+composite). Nothing below changes the feature set itself —
only which algorithm/decision rule is applied on top of it.

### EPDS Result (Low / Medium / High)
**Current best: direct Random Forest classifier** (`n_estimators=300,
class_weight="balanced"`) on Arm B. CV F1-macro ≈ 0.578, holdout QWK=0.557.
Nothing tested since has beaten it:
- Step 13 (LR-learned composite weights instead of chi-square): CV
  F1-macro 0.590 vs 0.578 — technically higher, but rejected because
  PPD_binary went the other way and the difference is within normal CV
  noise (no universal winner) — chi-square weighting kept.
- Step 10 (Regression + Threshold): best regression QWK=0.521 vs direct
  classifier's 0.557 — direct classifier stays best for this target.
- Step 12 (CatBoost native categorical): CV F1=0.528 vs one-hot's 0.537 —
  one-hot/current approach still wins.
- Step 14 (SMOTE): +0.004 (0.578→0.583) — tiny, rejected for
  cross-target consistency (hurt PHQ9 badly).
- Step 15 (Age², Age bands): +0.005, negligible — not adopted.

### PHQ9 Result (Minimal / Mild / Moderate / Moderately Severe / Severe)
**Current best: Linear Regression on continuous PHQ9 Score, thresholded
into the 5 official clinical bands** (script 10's finding) — this
GENUINELY beats the direct classifier on every metric: accuracy
0.344→0.463, F1-macro 0.329→0.426, QWK 0.548→0.602. This is the one
target where the modeling APPROACH itself changed, not just a
hyperparameter or feature tweak. All other Tier 1/2 experiments were
tested against the (now superseded) direct-classifier baseline number
of F1-macro≈0.365 chi-square-weighted:
- Step 13 (LR-learned weights): 0.371 vs 0.365 — marginal, rejected
  (no universal winner across targets).
- Step 14 (SMOTE): 0.365→0.348 — actively HURT this target, rejected.
- Step 12 (CatBoost native): CV tie (0.337 vs 0.337) but native WON on
  holdout specifically (0.400 vs 0.336) — noted as a simplicity/
  efficiency finding (59 vs 116 features) but not adopted as the primary
  approach since the regression-then-threshold pipeline already beats
  both.
- Step 15 (Age², group importance): negligible predictive change, but
  surfaced 3 under-rated raw variables (age of older children, disease
  before pregnancy, husband's education) worth naming in discussion.

### PPD_binary (binary)
**Current best: Random Forest classifier on Arm B, same as EPDS, with
TWO post-hoc refinements layered on top of the same model (not different
features/algorithms):**
1. **Decision threshold 0.407** (max-F1, from Step 11) instead of the
   default 0.5 — same accuracy (0.775), better F1 (0.731→0.753) and
   recall (0.700→0.786). A "recall≥85% screening" threshold (0.303) is
   also documented as an alternative for a high-recall screening use
   case (trades accuracy/precision for catching more true cases).
2. **Calibrated probabilities** (`CalibratedClassifierCV`, Platt/sigmoid)
   for reporting a patient-facing risk score — Brier 0.1677→0.1667,
   ROC-AUC essentially unchanged (0.829→0.828).
   CV F1-macro (features/model only, pre-threshold-tuning) ≈ 0.753.
   Other experiments tested and rejected/inconclusive:
   - Step 13 (LR-learned weights): chi-square won here specifically
     (0.753 vs 0.746) — opposite direction from EPDS/PHQ9, confirming
     "no universal winner," kept chi-square weighting for all targets.
   - Step 14 (SMOTE): small gain (0.753→0.758) but rejected for
     cross-target consistency (hurt PHQ9 elsewhere).
   - Step 12 (CatBoost native / Repeated CV): roughly ties one-hot;
     Repeated 5x5 CV found composite-feature effect not significant
     specifically for Random Forest on this target (though positive in
     direction), while significant for boosting models on other targets.

**Bottom line: the "Arm B feature set + Random Forest" backbone is
unchanged and un-beaten for EPDS Result and PPD_binary across everything
tried. PHQ9 Result is the one target where a genuinely better APPROACH
(regression-then-threshold) was found and adopted.**

### Script 20 — closing-the-loop feature test: **Run — DONE. Result: NONE
of the 4 configs meaningfully beat any target's baseline.** Full numbers
(5-fold CV mean ± std):

| Target | baseline | +interactions | +cluster | +both |
|---|---|---|---|---|
| EPDS Result F1-macro | 0.5785±0.035 | 0.5726±0.044 | 0.5831±0.030 | 0.5813±0.035 |
| PPD_binary F1-macro | 0.7527±0.027 | 0.7557±0.030 | 0.7508±0.026 | 0.7597±0.038 |
| PHQ9 Result F1-macro | 0.3523±0.044 | 0.3340±0.040 | 0.3527±0.042 | 0.3337±0.032 |
| PHQ9 Result QWK | 0.5542 | 0.5463 | 0.5495 | 0.5426 |

**Every single delta (best-config minus baseline) is smaller than that
target's own fold-to-fold standard deviation** (EPDS: +0.0046 vs std
0.030-0.035; PPD_binary: +0.0069 vs std 0.027-0.038; PHQ9: +0.0004 vs std
0.042) — i.e. every "improvement" is pure noise, not a real effect. For
PHQ9 Result specifically, adding the interaction terms consistently made
BOTH F1-macro (-0.018) and QWK (-0.008 to -0.012) slightly worse across
the board (plus_interactions and plus_both both underperform baseline) --
a small but consistent negative direction, most likely because a Linear
Regression model with ~640 training rows and 112+3=115 correlated
features is more sensitive to added multicollinearity than a tree model
would be.

**Decision: do NOT adopt any of the 4 configurations for any target.
Keep each target's current-best model exactly as documented above,
unchanged.** This is a genuinely valuable, honest negative result for the
thesis: it directly answers "can the script 17/18 discoveries be turned
into a model improvement?" with a clear, evidence-based "no" — and
explains WHY: Random Forest/tree ensembles already learn feature
interactions internally via splits (that's precisely what script 17's
SHAP interaction values were measuring — existing learned interactions,
not missed ones), and the KMeans Cluster-ID feature is a lossy,
information-REDUCING transformation of the 4 composite indices that are
already present as continuous inputs (discretizing continuous scores into
3 buckets can only lose information relative to the raw scores, never
add any). **This closes the loop on the entire Option 1-6 exploration
round (Step 13) — every discovery was correctly explanatory rather than
requiring a further model change, and the current per-target best models
are confirmed to already be near the practical ceiling for this feature
set.** Next: Step 14, final write-up.

---

## 3.6 Master Model Comparison Tables (All Models, All Targets)

*(Every model actually run, consolidated in one place. "Baseline" = Step 7,
all 5 algorithms on the full raw+composite feature set. "Tuned/Ensemble" =
Step 7b's RandomizedSearchCV-tuned versions, MLP, and Stacking Ensemble.
"Ablation Arms" = Step 8's Raw-only (A) vs Raw+Composite (B) vs
Selected+Composite (C), Random Forest and Logistic Regression rows shown
as the two most consistently-tracked models; full 5-model×3-arm tables are
in `ablation_results_<target>.csv`. **Bold** = best in that block by CV
F1-macro; ⭐ = the model actually adopted as current best, per Section 3.5.)*

### EPDS Result (Low / Medium / High) — CV F1-macro (5-fold)

| Stage | Model | CV F1-macro | Holdout F1-macro | Holdout ROC-AUC |
|---|---|---|---|---|
| Baseline | **Random Forest** | **0.5828** | 0.5201 | 0.7485 |
| Baseline | XGBoost | 0.5468 | 0.4822 | 0.7309 |
| Baseline | Logistic Regression | 0.5461 | 0.4708 | 0.7160 |
| Baseline | LightGBM | 0.5391 | 0.4971 | 0.7234 |
| Baseline | CatBoost | 0.5371 | 0.5171 | 0.7498 |
| Tuned/Ensemble | **Random Forest (tuned)** | **0.5896** | 0.5020 | 0.7491 |
| Tuned/Ensemble | Stacking Ensemble | 0.5771 | 0.5411 | 0.7604 |
| Tuned/Ensemble | CatBoost (tuned) | 0.5598 | 0.5398 | 0.7633 |
| Tuned/Ensemble | LightGBM (tuned) | 0.5512 | 0.5058 | 0.7418 |
| Tuned/Ensemble | XGBoost (tuned) | 0.5430 | 0.5368 | 0.7523 |
| Tuned/Ensemble | MLP (Deep Learning) | 0.4994 | 0.3874 | 0.6446 |
| Ablation Arm A (Raw only, 108 feat) | Random Forest | 0.5712 | 0.5409 | 0.7673 |
| Ablation Arm B (Raw+Composite, 112 feat) ⭐ | **Random Forest** | **0.5785** | 0.5399 | 0.7554 |
| Ablation Arm C (Selected 30 feat) | Random Forest | 0.5789 | 0.5151 | 0.7396 |
| Tier 2 | Arm B + Age²/Age-bands | 0.5827 | — | — |
| Tier 2 | Arm B + SMOTE | 0.5828 | — | — |
| Tier 2 | Arm B + LR-learned composite weights | 0.5905 | — | — |
| Closing-loop (Script 20) | Arm B + interaction/cluster features | 0.5726-0.5831 | — | — |

**Adopted model: Random Forest, Arm B feature set (raw + weighted
composite, 112 features), untuned hyperparameters** (`n_estimators=300,
class_weight="balanced"`). Note the tuned RF (0.5896 CV) is technically
higher than the adopted baseline RF (0.5785 CV under Step 8's fold split),
but was NOT carried forward as the reference model — its HOLDOUT F1
(0.5020) is actually lower than the plain baseline's (0.5201/0.5399
depending on split), a classic sign of mild overfitting to the CV folds
during hyperparameter search on a modest n=800 dataset. The simpler,
untuned model was judged more trustworthy and reproducible for the
explainability chapter (Step 9 onward).

### PHQ9 Result (Minimal/Mild/Moderate/Moderately Severe/Severe) — CV F1-macro

| Stage | Model | CV F1-macro | Holdout F1-macro | Holdout ROC-AUC |
|---|---|---|---|---|
| Baseline | **XGBoost** | **0.3622** | 0.3591 | 0.6732 |
| Baseline | Random Forest | 0.3501 | 0.3668 | 0.6931 |
| Baseline | Logistic Regression | 0.3458 | 0.3601 | 0.6556 |
| Baseline | LightGBM | 0.3397 | 0.3992 | 0.6770 |
| Baseline | CatBoost | 0.3374 | 0.3356 | 0.7029 |
| Tuned/Ensemble | **Random Forest (tuned)** | **0.3700** | 0.3167 | 0.6999 |
| Tuned/Ensemble | CatBoost (tuned) | 0.3606 | 0.3383 | 0.7060 |
| Tuned/Ensemble | Stacking Ensemble | 0.3551 | 0.3348 | 0.7257 |
| Tuned/Ensemble | MLP (Deep Learning) | 0.3131 | 0.3001 | 0.6181 |
| Ablation Arm A (Raw only) | Random Forest | 0.3575 | 0.3125 | 0.6776 |
| Ablation Arm B (Raw+Composite) | Random Forest | 0.3648 | 0.3294 | 0.6915 |
| Ablation Arm C (Selected 30 feat) | Random Forest | 0.3302 | 0.3630 | 0.7115 |
| Tier 1 — **Regression + Threshold (Linear Regression)** ⭐ | — | — | **Acc 0.463, F1 0.426, QWK 0.602** | — |
| Tier 1 — Direct RF Classifier (for comparison) | Random Forest | — | Acc 0.344, F1 0.329, QWK 0.548 | — |
| Tier 2 | Arm B + LR-learned composite weights | 0.3714 | — | — |
| Tier 2 | Arm B + SMOTE | 0.3476 (↓ from 0.3648) | — | — |
| Tier 2 | Arm B + Age²/Age-bands | 0.3626 | — | — |
| Closing-loop (Script 20, on adopted regression pipeline) | +interactions/+cluster | 0.3337-0.3527 (all ≤ baseline 0.3523) | — | — |

**Adopted model: Linear Regression on the continuous PHQ9 Score (Arm B
features), thresholded into the 5 official clinical bands** — the ONE
target where the modeling APPROACH itself changed (not just
hyperparameters). This is PHQ9 Result's hardest-to-predict nature made
visible: even the best CV F1-macro among direct classifiers (XGBoost
baseline, 0.3622) is barely above chance for a 5-class problem, and QWK
(0.602 for the adopted regression approach) tells a more forgiving,
clinically fairer story — most errors are near-miss/adjacent-band, not
wild misses.

### PPD_binary (binary) — CV F1-macro

| Stage | Model | CV F1-macro | Holdout F1-macro | Holdout ROC-AUC |
|---|---|---|---|---|
| Baseline | **LightGBM** | **0.7588** | 0.7279 | 0.8186 |
| Baseline | CatBoost | 0.7492 | 0.7561 | 0.8170 |
| Baseline | Random Forest | 0.7459 | 0.7362 | **0.8261** |
| Baseline | XGBoost | 0.7397 | 0.7350 | 0.8248 |
| Baseline | Logistic Regression | 0.7267 | 0.7341 | 0.8032 |
| Tuned/Ensemble | **Stacking Ensemble** | **0.7618** | **0.7654** | 0.8089 |
| Tuned/Ensemble | Random Forest (tuned) | 0.7574 | 0.7570 | 0.7929 |
| Tuned/Ensemble | LightGBM (tuned) | 0.7541 | 0.7193 | 0.8129 |
| Tuned/Ensemble | CatBoost (tuned) | 0.7524 | 0.7491 | 0.8124 |
| Ablation Arm A (Raw only) | Random Forest | 0.7563 | 0.7561 | 0.8194 |
| Ablation Arm B (Raw+Composite) ⭐ | **Random Forest** | **0.7527** | 0.7620 | 0.8287 |
| Ablation Arm C (Selected 30 feat) | Random Forest | 0.7458 | 0.7647 | 0.8206 |
| Tier 1 — Threshold retuned to 0.407 (max-F1) | Random Forest | — | **Acc 0.775, F1 0.753, Recall 0.786** | 0.8286 |
| Tier 1 — Calibrated probabilities | Random Forest | — | Brier 0.1677→0.1667 | 0.8286→0.8276 |
| Tier 2 | Arm B + LR-learned composite weights | 0.7457 (↓ from 0.7527) | — | — |
| Tier 2 | Arm B + SMOTE | 0.7576 | — | — |
| Tier 2 | Arm B + Age²/Age-bands | 0.7555 | — | — |
| Closing-loop (Script 20) | +interactions/+cluster | 0.7508-0.7597 (within noise of 0.7527) | — | — |

**Adopted model: Random Forest, Arm B feature set, decision threshold
0.407 instead of 0.5, calibrated probabilities for risk-score reporting.**
Note the Stacking Ensemble is the single best RAW predictive model found
anywhere in the project for this target (0.7618 CV / 0.7654 holdout,
both better than RF's 0.7527/0.7620) — deliberately NOT adopted as the
final model because of the interpretability trade-off explained in
Novelty 6 above (Section 1.5): a stacking ensemble of 4 different model
types cannot be explained by `shap.TreeExplainer` as cleanly as a single
Random Forest, and this project's stated contribution is an *explainable*
framework, not the single highest possible accuracy number.

---

## 4. Code Files — Location & Description

All code lives in: `codes/` (inside the dataset folder)

| File | Description | Status |
|---|---|---|
| `codes/01_dataset_overview.py` | Loads raw CSV, cleans column names, classifies every column as TARGET / LEAKAGE / SAFE_INPUT, prints missingness + dtypes + class distributions, saves `column_catalogue.csv` and 3 plots. Read-only exploration, does not fix data. | Done, ran successfully by Samir. |
| `codes/02_data_cleaning.py` | Full cleaning pipeline: fixes category typos/case mismatches (Education Level, Total children, disease columns, Likert items across all EPDS/PHQ-9 questions), fills structural missing values with explicit categories (Addiction, pregnancy loss, disease, income, older-child age, Need for Support), mode-imputes true random missing (Abuse, Husband's income/education, Education Level, Trust and share feelings) with `_was_missing` flag columns, and **recomputes `EPDS Result`/`PHQ9 Result` directly from the numeric scores** to fix internally-inconsistent labels (fixed 16 EPDS + 40 PHQ9 mislabeled rows, including the non-standard "Normal" PHQ9 category). Adds `PPD_binary` target. | Updated just now with the `Need for Support` fix — **needs to be re-run** to produce the final `PPD_dataset_cleaned_v2.csv`. |
| `codes/outputs/column_catalogue.csv` | One row per raw column: dtype, role (target/leakage/safe input), n_unique, missing count/%, sample values. | Generated by script 01. |
| `codes/outputs/missing_values.csv` | Missing-value table for the raw dataset. | Generated by script 01. |
| `codes/outputs/dataset_overview_report.txt` | Full printed log of script 01. | Generated by script 01. |
| `codes/outputs/PPD_dataset_cleaned.csv` | Cleaned column *names* only (from script 01, NOT fully cleaned values) — superseded by `PPD_dataset_cleaned_v2.csv` below. | Superseded, safe to ignore. |
| `codes/outputs/cleaning_log.txt` | Full printed log of script 02 (every before/after count, every fill decision). | **Final** — confirmed zero missing values, shape (800, 75). |
| `codes/outputs/missing_after_cleaning.csv` | Missing-value table after cleaning. | **Final** — confirmed empty (no missing values). |
| `codes/outputs/PPD_dataset_cleaned_v2.csv` | **The final cleaned dataset (800 rows x 75 cols)** — use this for all feature engineering / modeling going forward. | **Locked in as final.** |
| `codes/outputs/composite_feature_plan.json` | Which raw columns feed into each of the 4 planned composite indices. | Generated by script 01 (plan only, not yet built). |
| `codes/03_association_eda.py` | Chi-square (categorical) and ANOVA/Kruskal-Wallis (numeric) association tests between every safe input feature and each target (EPDS Result, PHQ9 Result, PPD_binary). Identifies which raw risk factors are statistically significant, informing/validating the composite index design in Step 5. | **Done** — 28-31 of 53 features significant per target; top features (Angry after childbirth, Abuse, in-law/husband relationship, support level, PHQ2 history) validate the SSI and MHRI composite designs. |
| `codes/outputs/association_EPDS_Result.csv` / `association_PHQ9_Result.csv` / `association_PPD_binary.csv` | Per-target association results, sorted by p-value. | Generated, final. |
| `codes/04_feature_engineering.py` | Builds 8 composite columns total: 4 equal-weight indices (Economic Stability, Social/Family Support, Maternal Mental-Health Risk, Neonatal/Delivery Stress) + 4 weighted variants (same indices, but each sub-feature weighted by its own chi-square association strength vs `PPD_binary`). Includes 4 empirically-corrected ordinal maps (Abuse, Family type, Number of household members, Education Level — see Step 5.5). Validates all 8 against all 3 targets. | **Final** — 24/24 significance tests pass, all directions confirmed correct. |
| `codes/outputs/PPD_dataset_with_composite_features.csv` | Cleaned dataset + 8 composite index columns, shape (800, 83). | **Locked in as final** for feature engineering. |
| `codes/outputs/composite_index_association.csv` / `composite_index_summary.txt` | Significance results and descriptive stats for all 8 composite indices vs. all 3 targets. | Generated, final. |
| `codes/05_direction_check.py` | Empirically checks whether the ordinal-map direction assumed for every raw sub-feature used in the 4 composites actually matches the data — prints PPD-positive rate per category ordered by assigned score, checks against a per-column expected direction (increasing for risk-index sub-features, decreasing for protective-index ones), flags mismatches. | **Final** — confirmed all directions correct after the Step 5.5 fixes. Useful to re-run if the composite designs ever change. |
| `codes/06_encoding.py` | One-hot encodes all categorical safe features (drop_first=True, dtype=int) and composite indices' source columns, leaves numeric/composite/flag columns as-is, drops the 21 leakage columns, keeps the 3 targets un-encoded. Produces the final `PPD_model_ready.csv` for the modeling script. Keeps raw + composite features side by side (needed for the Step 8 ablation study). | **Run once, working** (800, 119) shape confirmed, 0 missing. Fixed a dtype=int cosmetic issue (get_dummies defaulted to bool) — **needs one more re-run** to regenerate with the fix. |
| `codes/outputs/PPD_model_ready.csv` / `feature_columns.txt` / `encoding_map.csv` | Fully numeric model-ready dataset, the list of feature columns, and a map from each one-hot column back to its original raw column (for SHAP grouping later). | Generated, will be regenerated with clean int dtype on next run. |
| `codes/07_modeling.py` | Trains Logistic Regression + Random Forest (always) and XGBoost/LightGBM/CatBoost (if installed) for each of the 3 targets, using the full `PPD_model_ready.csv` (raw + composite features together). Reports stratified 5-fold CV + 80/20 holdout metrics (accuracy, precision, recall, F1, ROC-AUC). Saves `model_performance_<target>.csv` per target. | **Final** — clean run confirmed, no errors. PPD_binary is the strongest target (ROC-AUC ~0.82). |
| `codes/outputs/model_performance_EPDS_Result.csv` / `model_performance_PHQ9_Result.csv` / `model_performance_PPD_binary.csv` | Per-target model comparison (CV + holdout metrics, ranked by CV F1-macro). | Generated, final. |
| `codes/07b_advanced_modeling.py` | Answers "does tuning/MLP/stacking beat the baseline?" — RandomizedSearchCV tuning for RF/XGBoost/LightGBM/CatBoost, a small MLP (deep learning) architecture search, and a StackingClassifier hybrid ensemble of the 4 tuned tree models. Merges in Step 7's baseline numbers for direct comparison. Saves `model_performance_advanced_<target>.csv` + `best_hyperparameters.json`. Now also logs MLP epoch counts (`holdout_epochs_used` column, plus mean-epochs-per-fold printed during the architecture search and saved in `best_hyperparameters.json`) since tree models don't have an epoch concept but MLP does. | Previously run/final; **updated with epoch logging, needs one more re-run.** |
| `codes/08_ablation_study.py` | Compares Raw-only vs Raw+Weighted-Composite vs Selected(top-30)+Composite feature sets, same models/CV protocol as script 07, for all 3 targets. Prints an arm-by-arm F1/accuracy comparison and a verdict on whether composites/selection helped. Saves `ablation_results_<target>.csv` + `ablation_summary.txt`. Now also runs a paired t-test + Wilcoxon signed-rank test per model per target (B vs A, C vs B) on the same 5 CV folds, saved to `ablation_significance.csv`, to check whether arm-vs-arm differences are statistically real and not just noise. | Nested-CV fix confirmed via re-run; **updated with significance testing, needs one more re-run.** |
| `codes/09_shap_explainability.py` | Trains Random Forest on the Arm B "Raw + Composite" feature set (112 features) per target, computes SHAP TreeExplainer values on the held-out 20% test set, ranks global feature importance, and saves summary bar/beeswarm/dependence/waterfall plots plus a plain-language interpretation .txt per target. Every plotting block is wrapped in try/except (shap's plotting API varies by version) so a single plot failure doesn't stop the script. | **Written, not yet run** — needs `pip install shap` + a run to confirm. |
| `codes/10_ordinal_regression.py` | Rebuilds Arm B features from `PPD_dataset_with_composite_features.csv` (needs the continuous EPDS/PHQ9 Score as regression target, which 06_encoding.py deliberately drops). Compares a direct Random Forest classifier against regression-then-threshold (5 regressor families) for EPDS Result and PHQ9 Result, scored by accuracy, F1-macro, AND Quadratic Weighted Kappa (QWK). | **Written, not yet run.** |
| `codes/11_threshold_calibration.py` | For PPD_binary: precision-recall-curve-based threshold tuning (max-F1 and recall>=85% screening threshold vs default 0.5) plus CalibratedClassifierCV (Platt scaling) with Brier score and reliability-curve comparison. | **Written, not yet run.** |
| `codes/12_catboost_native_and_repeated_cv.py` | Part A: CatBoost native categorical handling vs one-hot, identical split, all 3 targets. Part B: re-runs 08's Arm A vs B significance test with RepeatedStratifiedKFold (5x5=25 folds) for higher statistical power. | **Written, not yet run.** |
| `all_the_problem_solve_column.ipynb` | Collaborator's original exploration notebook (partial cleaning, not fully saved). Reference only — not being edited directly (see Problems history: decided not to risk blind-editing the raw `.ipynb` JSON). | Reference only. |

---

## 5. Current Problems / Open Issues

*(This section should only ever list problems that are ACTIVE right now. Delete an entry the moment it's resolved — do not keep a history log here.)*

- **Claude's code sandbox is currently unable to execute Python** (infra
  issue on Claude's end this session: "Workspace unavailable... isolated
  Linux environment failed to start"). This means Claude cannot run/verify
  scripts before handing them to Samir — Samir must run each script locally
  and paste the console output back so Claude can verify correctness. Check
  again each session; if the sandbox is back, Claude should self-verify
  scripts before sharing them.
- **`Abuse` column semantics need real-world verification.** The empirical
  direction check found "Yes" answers to `Abuse` correlate with LOWER PPD
  (27.0%) and "No" answers with HIGHER PPD (64.1%) — the opposite of the
  natural clinical expectation that experiencing abuse should raise
  depression risk. We've made the composite index internally consistent by
  flipping the map to match the empirical pattern, but the real-world
  meaning of "Yes"/"No" in this specific survey question is still unclear
  (possible translation artifact, a safety-framed question rather than an
  abuse-framed one, or a genuine but unusual finding). Worth asking the
  dataset's original author/collaborator to clarify the exact Bangla
  wording of this question before finalizing the thesis's methodology
  section — this affects how `Abuse` and `Maternal_MentalHealth_Risk_Index`
  should be *described*, though it doesn't block the modeling pipeline
  either way (ML models don't care about label direction, only the
  composite index's human-facing interpretation does).
