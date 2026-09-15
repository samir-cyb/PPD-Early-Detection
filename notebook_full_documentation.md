# 📓 Notebook Full Documentation
## `all_the_problem_solve_column (1).ipynb`
### Postpartum Depression (PPD) Prediction — Final Version (After Professor's Changes)

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Loading](#3-dataset-loading)
4. [Data Cleaning — Phase 1 (Category Fixes)](#4-data-cleaning--phase-1-category-fixes)
5. [Data Cleaning — Phase 2 (Column Names & Duplicates)](#5-data-cleaning--phase-2-column-names--duplicates)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7. [Numerical Feature Handling & PHQ9 Banding](#7-numerical-feature-handling--phq9-banding)
8. [Encoding & PPD_binary Creation](#8-encoding--ppd_binary-creation)
9. [Chi-2 Feature Ranking (65 Features)](#9-chi-2-feature-ranking-65-features)
10. [Chi-2 Backward Elimination → 17 Optimal Features](#10-chi-2-backward-elimination--17-optimal-features)
11. [⚠️ DATA LEAKAGE Warning](#11-️-data-leakage-warning)
12. [15-Model Comparison](#12-15-model-comparison)
13. [Feature Category Mapping (6 Groups)](#13-feature-category-mapping-6-groups)
14. [Category-wise Model Performance](#14-category-wise-model-performance)
15. [Correlation Analysis (17 Features)](#15-correlation-analysis-17-features)
16. [Multinomial LR Coefficients](#16-multinomial-lr-coefficients)
17. [RF + XGB Feature Importance](#17-rf--xgb-feature-importance)
18. [All-Feature Summary with Leakage Labels](#18-all-feature-summary-with-leakage-labels)
19. [CELL 7 — Within-Category Feature Importance](#19-cell-7--within-category-feature-importance)
20. [CELL 8 — Risk Factor Analysis via Kendall's Tau](#20-cell-8--risk-factor-analysis-via-kendalls-tau)
21. [Figure Generation (fig1–fig12)](#21-figure-generation-fig1fig12)
22. [Chart 1–4 (Advanced Visualizations)](#22-chart-14-advanced-visualizations)
23. [Complete Cell Index (71 Cells)](#23-complete-cell-index-71-cells)

---

## 1. Project Overview

**গবেষণার বিষয়:** বাংলাদেশের প্রসবোত্তর বিষণ্নতা (Postpartum Depression — PPD) নির্ণয়ের জন্য মেশিন লার্নিং মডেল তৈরি

**Platform:** Google Colab  
**Drive Path:** `/content/drive/MyDrive/thesis_ppd/`  
**Dataset:** `PPD_dataset_v3.csv` → 800 rows × 70 columns  
**Final saved file:** `PPD_data_updated_v34.csv`  
**Output figures:** `fig1.jpg` through `fig12_3d_clustered.jpg`

**গবেষণার মূল প্রশ্নগুলো:**
- কোন features PPD নির্ণয়ে সবচেয়ে গুরুত্বপূর্ণ?
- Screening questionnaire (EPDS/PHQ9) ছাড়া শুধু demographic/familial/health features দিয়ে কত ভালো prediction করা যায়?
- Data leakage ছাড়া বাস্তবসম্মত accuracy কত?

---

## 2. Environment Setup

```python
# Cell 1: Google Drive Mount
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2: All Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')
```

---

## 3. Dataset Loading

```python
# Cell 3: Load dataset
df = pd.read_csv('/content/drive/MyDrive/thesis_ppd/PPD_dataset_v3.csv')
print(df.shape)  # Output: (800, 70)
```

**Output:** `(800, 70)` — 800 জন মায়ের ডেটা, 70টি column

```python
# Cell 4: Drop 'sr' (serial number) column
df.drop(columns=['sr'], inplace=True)
# → (800, 69) columns
```

---

## 4. Data Cleaning — Phase 1 (Category Fixes)

### 4.1 Column Rename
```python
# Cell 5: Rename column + strip special characters
df.columns = (
    df.columns
    .str.strip()
    .str.replace("'", "'", regex=False)
)
df.rename(columns={
    "Recieved Support": "Received Support",
}, inplace=True)
```

### 4.2 Education Level Fix
```python
# Inconsistent values fixed:
df['Education Level'] = df['Education Level'].replace({
    # Various spelling inconsistencies standardized
})
df["Husband's education level"] = df["Husband's education level"].replace({...})
```

### 4.3 Total Children Fix
```python
df['Total children'] = df['Total children'].replace('More than two', 'More than Two')
```

### 4.4 History of Pregnancy Loss Fix
```python
df['History of pregnancy loss'] = df['History of pregnancy loss'].replace(
    'Still-born delivery', 'Still-born Delivery'
)
```

### 4.5 Disease Before Pregnancy Fix (Line 15595)
```python
# Remove invalid numeric entry '1.2' — replace with NaN
col_target = 'Disease before pregnancy'
df[col_target] = df[col_target].replace('1.2', np.nan)
```
**কারণ:** Frequency analysis-এ দেখা গেছে 1টি row-এ `'1.2'` — clearly wrong/typo → NaN করা হয়েছে

### 4.6 Fix Inconsistent Category Values — Batch Replace (Line 15851)
```python
# Fix inconsistent category values across many columns
df = df.replace({
    # Various inconsistencies across categorical columns
    # Standardizes Yes/No capitalization, spelling errors, etc.
})
```

### 4.7 Diseases During Pregnancy Fix (Line 15885)
```python
# Standardize values in 'Diseases during pregnancy' column
# e.g., spelling errors, case mismatches
```

### 4.8 Column Name & Label Cleaning (Line 16132)
```python
# Clean column names — strip extra spaces
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Replace inconsistent labels
replacement_map = {
    # Map inconsistent string values to standard ones
}
df = df.replace(replacement_map)
```

### 4.9 Occupation Column Fix (Line 16334)
```python
# Fix 'House wife' → 'Housewife' (found in both occupation columns)
cols_to_fix = ['Occupation before latest pregnancy', 'Occupation After Your Latest Childbirth']
for col in cols_to_fix:
    df[col] = df[col].replace('House wife', 'Housewife')
```
**কারণ:** EDA-তে দেখা গেছে ১টি row-এ 'House wife' (space সহ) → same as 'Housewife'

---

## 5. Data Cleaning — Phase 2 (Column Names & Duplicates)

### 5.1 Duplicate Check (Line 16841)
```python
# Check duplicates
duplicate_count = df.duplicated().sum()
print("Duplicate rows:", duplicate_count)
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)
```
**Output:**
```
Duplicate rows: 0
Shape after removing duplicates: (800, 69)
```
✅ কোনো duplicate নেই — সব 800 row unique

### 5.2 Unique Values Verification (Line 16869–16878)
```python
# Check unique values for all columns
for col in df.columns:
    print(f"\nUnique values for {col}:")
    print(df[col].unique())
```
**Notable outputs:**
- Occupation before latest pregnancy: Housewife, Service, Business, etc.
- Received Support: Yes, No
- Need for Support: Yes, No

### 5.3 Post-Cleaning Frequency Verification (Line 17057)
- সব column-এর frequency আবার run করা হয়েছে cleaning-এর পর
- Confirm যে কোনো inconsistency নেই
- 'House wife' এখন আর নেই — শুধু 'Housewife' আছে

---

## 6. Exploratory Data Analysis (EDA)

### 6.1 Frequency Analysis (Cell at line 739)
```python
# Frequency and Percentage for all columns
for col in df.columns:
    freq = df[col].value_counts()
    pct = df[col].value_counts(normalize=True) * 100
    result = pd.DataFrame({'Frequency': freq, 'Percentage': pct})
    display(result)
```

**Key findings from EDA:**

| Column | Notable Values |
|--------|---------------|
| Occupation before latest pregnancy | Housewife: 390 (48.75%), House wife: 1 (0.125%) — fixed |
| Occupation After Childbirth | Housewife: 534 (66.75%), House wife: 1 (0.125%) — fixed |
| Disease before pregnancy | Had `1.2` (1 row) — invalid, replaced with NaN |
| Received Support | Yes: ~79%, No: ~21% |
| Need for Support | Yes: ~79%, No: ~21% |
| EPDS Result | PPD: ~30%, No PPD: ~70% |

### 6.2 Missing Values (Cell at line 31241)
```python
# Missing value table
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_table = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
display(missing_table[missing_table['Missing Count'] > 0])
```

---

## 7. Numerical Feature Handling & PHQ9 Banding

### 7.1 PHQ9 Band Function (Cell at line 31558)
```python
def phq9_band(score):
    if score <= 4:
        return "Minimal"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    elif score <= 19:
        return "Moderately Severe"
    else:
        return "Severe"

df['PHQ9 Band'] = df['PHQ9 Score'].apply(phq9_band)
```

**PHQ9 Score Interpretation:**

| Score Range | Severity |
|-------------|----------|
| 0–4 | Minimal |
| 5–9 | Mild |
| 10–14 | Moderate |
| 15–19 | Moderately Severe |
| 20+ | Severe |

### 7.2 Descriptive Statistics (Cell at line 31665)
```python
print(df.describe())
```

### 7.3 EPDS Score Distribution (Cell at line 31894)
```python
# Distribution of EPDS scores
# PPD threshold: EPDS Score >= 13
print(df['EPDS Score'].describe())
print(df['EPDS Result'].value_counts())
```

### 7.4 Save Updated Dataset (Cell at line 31869)
```python
df.to_csv('/content/drive/MyDrive/thesis_ppd/PPD_data_updated_v34.csv', index=False)
print("Saved: PPD_data_updated_v34.csv")
```

---

## 8. Encoding & PPD_binary Creation

### 8.1 Load & Encode (Cell at line 32741)
```python
# Load v34 and encode all categorical columns
df_encoded = pd.read_csv('/content/drive/MyDrive/thesis_ppd/PPD_data_updated_v34.csv')
le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
```

### 8.2 Create PPD_binary Target (Cell at line 32904)
```python
# Load numeric CSV
df = pd.read_csv('/content/drive/MyDrive/thesis_ppd/PPD_data_updated_v34.csv')

# Create binary target
df["PPD_binary"] = (df["EPDS Score"] >= 13).astype(int)
# 0 = No PPD (EPDS < 13)
# 1 = PPD (EPDS >= 13)
```

**EPDS Audit:**
```
[EPDS AUDIT] 43 / 800 rows had inconsistent labels.
[PHQ9 AUDIT] 0 / 800 rows had inconsistent labels.
```
→ 43 row-এ EPDS Score ও EPDS Result-এর মধ্যে inconsistency ছিল → score দিয়ে correct করা হয়েছে

### 8.3 Exclude Columns for Feature Selection
```python
exclude_cols = ['EPDS Result', 'EPDS Score', 'PHQ9 Score', 'PHQ9 Result', 'PPD_binary']
# বাকি সব = feature candidates
```

---

## 9. Chi-2 Feature Ranking (65 Features)

### Cell at line 33411
```python
from sklearn.feature_selection import chi2, SelectKBest

X = df_encoded.drop(columns=exclude_cols)
y = df_encoded['PPD_binary']

# Chi2 ranking — সব 65 feature
chi2_selector = SelectKBest(chi2, k='all')
chi2_selector.fit(X, y)
chi2_scores = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_selector.scores_,
    'P-Value': chi2_selector.pvalues_
}).sort_values('Chi2 Score', ascending=False)
```

**Top features from Chi2 ranking (selected highlights):**

| Rank | Feature | Chi2 Score | p-value |
|------|---------|-----------|---------|
| 1 | You have felt sad or miserable | ~High | <0.001 |
| 2 | Feeling down, depressed, or hopeless | ~High | <0.001 |
| 3 | Things have been getting to you | ~High | <0.001 |
| ... | (other EPDS/PHQ9 items) | ... | ... |
| 19 | Occupation before latest pregnancy | 0.8750 | 0.8397 |
| 39 | Received Support | 0.8838 | 0.8510 |
| 40 | Need for Support | 0.8912 | 0.8606 |

**Key insight:** EPDS/PHQ9 questionnaire items dominate the top ranks. Demographic/familial features rank low in Chi2.

---

## 10. Chi-2 Backward Elimination → 17 Optimal Features

### Cell at line 34564
```python
# Start with all 65 features
# Remove one feature at a time (lowest Chi2 score)
# At each step: 5-fold CV with Gradient Boosting
# Track accuracy and F1

best_step = 48  # Remove 48 features
# Best result: 17 features remaining

# Best metrics:
# Accuracy = 0.9125 (±0.0079)
# F1 = 0.8859
```

**Backward Elimination Steps:**

| Step | Features Left | Accuracy | F1 |
|------|--------------|----------|-----|
| 0 | 65 | ~0.89 | ~0.86 |
| 10 | 55 | ~0.90 | ~0.87 |
| 20 | 45 | ~0.91 | ~0.88 |
| 48 | **17** | **0.9125** | **0.8859** |
| 60 | 5 | ~0.85 | ~0.82 |

### The 17 "Optimal" Features (ALL LEAKAGE — see Section 11)
```python
optimal_features = [
    'Feeling down, depressed, or hopeless',
    'You have felt sad or miserable',
    'You have been anxious or worried for no good reason',
    'Trouble concentrating on things',
    'Things have been getting to you',
    'Feeling bad about yourself...',
    'You have blamed myself unnecessarily...',
    'You have been so unhappy that you have been crying',
    'You have been able to laugh and see the funny side of things',
    'You have looked forward with enjoyment to things',
    'You have felt scared or panicky for no good reason',
    'Little interest or pleasure in doing things',
    'Feeling tired or having little energy',
    'Thoughts that you would be better off dead, or of hurting yourself',
    'The thought of harming yourself has occurred',
    'Trouble falling or staying asleep, or sleeping too much',
    'You have been so unhappy that you have had difficulty sleeping',
]
```

---

## 11. ⚠️ DATA LEAKAGE Warning

> **এটি এই গবেষণার সবচেয়ে গুরুত্বপূর্ণ সীমাবদ্ধতা!**

### সমস্যাটি কী?

**Target variable:** `PPD_binary = (EPDS Score >= 13).astype(int)`

**17টি "optimal" feature:** সবগুলোই EPDS বা PHQ-9 questionnaire-এর প্রশ্ন।

**ফলাফল:** Model আসলে EPDS Score দিয়েই EPDS Score predict করছে — এটা circular reasoning!

### উদাহরণ
```
প্রশ্ন ৩ (EPDS): "আপনি কি বিনা কারণে উদ্বিগ্ন বা চিন্তিত হয়েছেন?"
→ এটি EPDS এর অংশ → EPDS Score নির্ধারণ করে
→ PPD_binary = EPDS Score >= 13
→ সুতরাং এই feature থেকে target predict করা = cheating
```

### Leakage Features (excluded from safe analysis)
```python
LEAKAGE_COLUMNS = ['EPDS Score', 'PHQ9 Score', 'PHQ9 Result', 'PPD_binary']
# এই column থেকে তৈরি যেকোনো feature = leakage
# 17টি "optimal" feature = সব EPDS/PHQ items = সব leakage
```

**সতর্কতা:** `'Leakage = YES'` label-এর features-এর correlation স্বাভাবিকভাবেই অনেক বেশি আসবে — এটা সত্যিকারের predictive power নয়।

### Real Accuracy (No Leakage)
- Screening Items category (contains EPDS/PHQ): **91%** ← FAKE (leakage)
- Neonatal Health features only: **61%** ← REAL
- Personal Health only: **52%** ← REAL
- Familial only: **47%** ← REAL
- Demographic only: **40%** ← REAL

---

## 12. 15-Model Comparison

### Initial 6-Model Comparison (Cell at line 34905)
```python
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'KNN': KNeighborsClassifier(),
    'MLP': MLPClassifier(),
}
# 5-fold CV on 17 optimal features
```

### Full 15-Model Comparison (Cell at line 35579)
```python
!pip install lightgbm catboost

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
```

**All 15 Models & Results (5-fold CV, 17 features, PPD_binary target):**

| Rank | Model | Accuracy | F1 Score |
|------|-------|----------|----------|
| 1 | **Gradient Boosting** | **0.926** | ~0.90 |
| 2 | LightGBM | ~0.920 | ~0.89 |
| 3 | XGBoost | ~0.918 | ~0.89 |
| 4 | Hist Gradient Boosting | ~0.916 | ~0.88 |
| 5 | Random Forest | ~0.912 | ~0.88 |
| 6 | CatBoost | ~0.910 | ~0.88 |
| 7 | Extra Trees | ~0.908 | ~0.87 |
| 8 | 1D CNN | ~0.900 | ~0.87 |
| 9 | Decision Tree | ~0.895 | ~0.86 |
| 10 | AdaBoost | ~0.890 | ~0.85 |
| 11 | MLP | ~0.885 | ~0.85 |
| 12 | Deep MLP | ~0.880 | ~0.84 |
| 13 | SVM | ~0.875 | ~0.84 |
| 14 | KNN | ~0.865 | ~0.83 |
| 15 | Logistic Regression | ~0.858 | ~0.82 |

> ⚠️ মনে রাখতে হবে: এই সব accuracy data leakage এর কারণে inflated।

---

## 13. Feature Category Mapping (6 Groups)

### Cell at line 36298 / 36300
```python
# CELL 2 (UPDATED): Feature Category Mapping (exact column names from Age.txt)

feature_categories = {
    # --- DEMOGRAPHIC (8 features) ---
    'Age': 'Demographic',
    'Education Level': 'Demographic',
    "Husband's education level": 'Demographic',
    'Occupation before latest pregnancy': 'Demographic',
    'Occupation After Your Latest Childbirth': 'Demographic',
    'Monthly income before latest pregnancy': 'Demographic',
    'Current monthly income': 'Demographic',
    'Residence': 'Demographic',
    
    # --- FAMILIAL (9 features) ---
    'Family type': 'Familial',
    'Marital status': 'Familial',
    'Total children': 'Familial',
    'Husband\'s attitude': 'Familial',
    'Trust and share feelings': 'Familial',
    'Abuse': 'Familial',
    'Family history of mental illness': 'Familial',
    'Feeling about motherhood': 'Familial',
    'Anger after latest child birth': 'Familial',
    
    # --- PERSONAL HEALTH (8 features) ---
    'Disease before pregnancy': 'Personal Health',
    'History of pregnancy loss': 'Personal Health',
    'History of depression': 'Personal Health',
    'History of anxiety': 'Personal Health',
    'Depression before pregnancy (PHQ2)': 'Personal Health',
    'Stress before pregnancy': 'Personal Health',
    'Major changes or losses during pregnancy': 'Personal Health',
    'Fear of pregnancy': 'Personal Health',
    
    # --- NEONATAL HEALTH (21 features) ---
    'Number of the latest pregnancy': 'Neonatal Health',
    'Pregnancy length': 'Neonatal Health',
    'Pregnancy plan': 'Neonatal Health',
    'Diseases during pregnancy': 'Neonatal Health',
    'Depression during pregnancy (PHQ2)': 'Neonatal Health',
    'Delivery method': 'Neonatal Health',
    'Birth complications': 'Neonatal Health',
    'Infant health status': 'Neonatal Health',
    'Baby\'s feeding method': 'Neonatal Health',
    'Received Support': 'Neonatal Health',
    'Need for Support': 'Neonatal Health',
    # ... (21 total)
    
    # --- SCREENING ITEMS (19 features) ---
    # All EPDS questions (10 items) + PHQ-9 questions (9 items)
    'Feeling down, depressed, or hopeless': 'Screening Items',
    'Little interest or pleasure in doing things': 'Screening Items',
    # ... (19 total EPDS + PHQ9 question items)
    
    # --- TARGET/SCORE (5 columns — excluded from features) ---
    'EPDS Score': 'Target/Score',
    'EPDS Result': 'Target/Score',
    'PHQ9 Score': 'Target/Score',
    'PHQ9 Result': 'Target/Score',
    'PPD_binary': 'Target/Score',
}
```

**Category Summary:**

| Category | Feature Count | Notes |
|----------|--------------|-------|
| Demographic | 8 | Age, Education, Occupation, Income, Residence |
| Familial | 9 | Family type, Spouse attitude, Abuse, Anger |
| Personal Health | 8 | Disease history, Depression/Anxiety history |
| Neonatal Health | 21 | Pregnancy, Birth, Infant, Support |
| Screening Items | 19 | ⚠️ EPDS (10) + PHQ-9 (9) questions = LEAKAGE |
| Target/Score | 5 | Excluded from features |
| **Total** | **70** | |

---

## 14. Category-wise Model Performance

### Cell at line 36460
```python
# For each category (excluding Target/Score):
# X = only that category's features
# y = PPD_binary
# Model: Gradient Boosting, 5-fold CV
```

**Results:**

| Category | Accuracy | Interpretation |
|----------|----------|---------------|
| **Screening Items** | **91%** | ⚠️ DATA LEAKAGE — circular |
| Neonatal Health | **61%** | ✅ Realistic — best real predictor |
| Personal Health | **52%** | ✅ Reasonable |
| Familial | **47%** | ✅ Moderate predictive power |
| Demographic | **40%** | ✅ Low (near baseline) |

**গুরুত্বপূর্ণ সিদ্ধান্ত:** যদি Screening Items বাদ দেওয়া হয়, Neonatal Health সবচেয়ে ভালো predictor। এই 4টি category দিয়েই thesis-এর real contribution।

---

## 15. Correlation Analysis (17 Features)

### Cell at line 36816
```python
# Correlation of 17 optimal features with EPDS Score
# Three methods: Pearson, Spearman, Kendall

correlations = pd.DataFrame()
for feature in optimal_features:
    pearson = df[feature].corr(df['EPDS Score'], method='pearson')
    spearman = df[feature].corr(df['EPDS Score'], method='spearman')
    kendall_tau, _ = kendalltau(df[feature], df['EPDS Score'])
    correlations = pd.concat([correlations, pd.DataFrame({
        'Feature': [feature],
        'Pearson': [pearson],
        'Spearman': [spearman],
        'Kendall': [kendall_tau],
    })])
```

**Notable output for safe features (Occupation before pregnancy):**
- Pearson: -0.010525, p = 0.194295
- Spearman: -0.183769, p = 0.129530
- Leakage: `no`

---

## 16. Multinomial LR Coefficients

### Cell at line 37373
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(multi_class='multinomial', max_iter=1000)
lr.fit(X_train, y_train)

coef_df = pd.DataFrame(lr.coef_, columns=X.columns)
# Coefficients show direction and magnitude of each feature's contribution
```

**Output includes per-class coefficients for all 17 features (or all 65 features in extended analysis).**

---

## 17. RF + XGB Feature Importance

### Cell at line 37719
```python
# Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'RF_Importance': rf.feature_importances_
}).sort_values('RF_Importance', ascending=False)

# XGBoost Feature Importance
xgb = XGBClassifier(random_state=42)
xgb.fit(X, y)
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'XGB_Importance': xgb.feature_importances_
}).sort_values('XGB_Importance', ascending=False)
```

---

## 18. All-Feature Summary with Leakage Labels

### Cell at line 38825
```python
# All 65 features: LR + RF + XGB importance + Leakage label
# For each feature, compute combined importance and mark:
# Leakage = "YES" → EPDS/PHQ question items
# Leakage = "NO"  → Demographic/Familial/Personal/Neonatal features

summary_df = pd.DataFrame({
    'Feature': all_features,
    'LR_Coef': [...],
    'RF_Importance': [...],
    'XGB_Importance': [...],
    'Leakage': ['YES' if f in screening_features else 'NO' for f in all_features]
})
```

**Sample output:**
```
Rank | Feature                              | RF_Imp | XGB_Imp | Leakage
-----|--------------------------------------|--------|---------|--------
  1  | You have felt sad or miserable       | High   | High    | YES
  2  | Feeling down, depressed...           | High   | High    | YES
...
 28  | Need for Support                     | 69.50  | ...     | NO
 29  | Received Support                     | 62.50  | ...     | NO
 30  | Feeling about motherhood             | ~50    | ...     | NO
```

---

## 19. CELL 7 — Within-Category Feature Importance

### Cell at line 39743 (NEW — Professor's Addition)

**উদ্দেশ্য:** প্রতিটি category-র ভেতরে কোন features সবচেয়ে গুরুত্বপূর্ণ?

```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

TARGET_CATS = ['Demographic', 'Familial', 'Personal Health', 'Neonatal Health']
# Note: Screening Items বাদ (leakage)

scaler = MinMaxScaler()

within_importance = {}

for cat in TARGET_CATS:
    cat_features = [f for f in feature_categories if feature_categories[f] == cat]
    X_cat = df_encoded[cat_features]
    y = df_encoded['PPD_binary']
    
    # Method 1: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_cat, y)
    rf_imp = rf.feature_importances_
    
    # Method 2: XGBoost
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_cat, y)
    xgb_imp = xgb.feature_importances_
    
    # Method 3: Logistic Regression (absolute coefficients)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_cat, y)
    lr_imp = np.abs(lr.coef_[0])
    
    # Method 4: Mutual Information
    mi = mutual_info_classif(X_cat, y, random_state=42)
    
    # Normalize each to [0, 1] and combine
    rf_norm = scaler.fit_transform(rf_imp.reshape(-1, 1)).flatten()
    xgb_norm = scaler.fit_transform(xgb_imp.reshape(-1, 1)).flatten()
    lr_norm = scaler.fit_transform(lr_imp.reshape(-1, 1)).flatten()
    mi_norm = scaler.fit_transform(mi.reshape(-1, 1)).flatten()
    
    combined = (rf_norm + xgb_norm + lr_norm + mi_norm) / 4
    
    within_importance[cat] = pd.DataFrame({
        'Feature': cat_features,
        'RF': rf_norm,
        'XGB': xgb_norm,
        'LR': lr_norm,
        'MI': mi_norm,
        'Combined': combined
    }).sort_values('Combined', ascending=False)

# Save output
within_importance_df = pd.concat(within_importance.values())
within_importance_df.to_csv(
    '/content/drive/MyDrive/thesis_ppd/within_category_importance.csv', 
    index=False
)
print("✅ Saved: within_category_importance.csv")
```

**Output file:** `within_category_importance.csv`

**Top features per category (sample):**

| Category | Top Feature | Combined Score |
|----------|------------|---------------|
| Familial | Anger after latest child birth | Highest in Familial |
| Familial | Feeling about motherhood | High |
| Familial | Abuse | Moderate-High |
| Neonatal Health | Depression during pregnancy (PHQ2) | Highest in Neonatal |
| Demographic | Age | Moderate |
| Personal Health | History of depression | Highest in Personal |

---

## 20. CELL 8 — Risk Factor Analysis via Kendall's Tau

### Cell at line 40718 (NEW — Professor's Addition)

**উদ্দেশ্য:** 4টি safe category (no screening items) থেকে সবচেয়ে গুরুত্বপূর্ণ risk factors কোনগুলো?

```python
from scipy.stats import kendalltau

TARGET_CATS = ['Demographic', 'Familial', 'Personal Health', 'Neonatal Health']
# Screening Items EXCLUDED (leakage)

results = []

for cat in TARGET_CATS:
    cat_features = [f for f in feature_categories if feature_categories[f] == cat]
    
    for feature in cat_features:
        # Correlation with EPDS Score (continuous)
        tau_score, p_score = kendalltau(df[feature], df['EPDS Score'])
        # Correlation with EPDS Result (binary: PPD/No PPD)
        tau_result, p_result = kendalltau(df[feature], df['PPD_binary'])
        
        results.append({
            'Category': cat,
            'Feature': feature,
            'Kendall_tau_vs_Score': tau_score,
            'p_value_vs_Score': p_score,
            'Kendall_tau_vs_Result': tau_result,
            'p_value_vs_Result': p_result,
        })

risk_df = pd.DataFrame(results).sort_values(
    'Kendall_tau_vs_Score', key=abs, ascending=False
)
risk_df.to_csv(
    '/content/drive/MyDrive/thesis_ppd/risk_factor_kendall_tau.csv', 
    index=False
)
print("✅ Saved: risk_factor_kendall_tau.csv")
```

**Output file:** `risk_factor_kendall_tau.csv`

**Key Risk Factor Findings (sorted by |τ|):**

| Rank | Feature | Category | τ (vs EPDS Score) | p-value |
|------|---------|----------|-------------------|---------|
| 1 | **Anger after latest child birth** | Familial | **+0.418** | 1.4×10⁻⁴⁵ |
| 2 | **Feeling about motherhood** | Familial | **+0.280** | 5.3×10⁻²² |
| 3 | **Abuse** | Familial | **-0.223** | 1.1×10⁻¹⁴ |
| 4 | **Depression during pregnancy (PHQ2)** | Neonatal Health | **+0.195** | 3.3×10⁻¹¹ |
| 5 | **Trust and share feelings** | Familial | **-0.177** | 1.8×10⁻⁹ |
| 6 | Need for Support | Neonatal Health | +0.061 | ~0.05 |
| ... | Other demographic features | Demographic | ~0.01–0.05 | >0.05 |

**ব্যাখ্যা:**
- **+τ** = feature বাড়লে EPDS Score বাড়ে (PPD risk বাড়ে)
- **-τ** = feature বাড়লে EPDS Score কমে (protective factor)
- `Anger after childbirth` = সবচেয়ে শক্তিশালী risk factor (τ=0.418)
- `Abuse` ও `Trust and share feelings` = protective factor (negative τ)

---

## 21. Figure Generation (fig1–fig12)

### 21.1 Main Figure Cell (line 41831) — fig1 through fig8

#### fig1 — Category Performance Bar Chart
```python
# Bar chart: Accuracy per category
# X-axis: 5 categories (Demographic, Familial, Personal Health, Neonatal Health, Screening Items)
# Y-axis: Accuracy (0–1)
# Colors: Different per category
plt.savefig('/content/drive/MyDrive/thesis_ppd/fig1_category_performance.jpg', dpi=300)
```

#### fig2 — Kendall's Tau Risk Factors
```python
# Horizontal bar chart
# Top N risk factors from CELL 8
# Color: Red = positive tau (risk), Blue = negative tau (protective)
plt.savefig('/content/drive/MyDrive/thesis_ppd/fig2_kendall_tau.jpg', dpi=300)
```

#### fig3 — LR Coefficients (Non-Screening Features)
```python
# Logistic Regression coefficients
# Only non-screening, non-leakage features
plt.savefig('/content/drive/MyDrive/thesis_ppd/fig3_lr_coefficients.jpg', dpi=300)
```

#### fig4 — RF Feature Importance
```python
# Random Forest feature importance (non-screening features)
plt.savefig('/content/drive/MyDrive/thesis_ppd/fig4_rf_importance.jpg', dpi=300)
```

#### fig5 — XGB Feature Importance
```python
# XGBoost feature importance (non-screening features)
plt.savefig('/content/drive/MyDrive/thesis_ppd/fig5_xgb_importance.jpg', dpi=300)
```

#### fig6 — Within-Category Heatmap
```python
# Heatmap: rows = features, cols = methods (RF, XGB, LR, MI)
# Shows normalized scores per method for each feature
# Based on CELL 7 output
plt.savefig('/content/drive/MyDrive/thesis_ppd/fig6_within_category_heatmap.jpg', dpi=300)
```

#### fig7 — Screening vs Others Comparison
```python
# Side-by-side comparison: Screening Items accuracy vs safe categories
# Visual showing the leakage gap (91% vs 40–61%)
plt.savefig('/content/drive/MyDrive/thesis_ppd/fig7_screening_vs_others.jpg', dpi=300)
```

#### fig8 — Correlation Heatmap
```python
# Full correlation heatmap of selected features
# Shows inter-feature correlations
plt.savefig('/content/drive/MyDrive/thesis_ppd/fig8_correlation_heatmap.jpg', dpi=300)
```

### 21.2 Data Gathering Summary (Cell at line 42301)
```
DATA GATHERING SUMMARY:
- Total features evaluated: 65
- Categories analyzed: 5 (+ Target/Score excluded)
- Best model: Gradient Boosting (0.926 with 17 features — leakage)
- Best real accuracy: Neonatal Health → 61%
- Risk factors identified via Kendall's tau: 46 (4 safe categories)
- Within-category importance computed: 4 categories
```

---

## 22. Chart 1–4 (Advanced Visualizations)

### CHART 1 — fig9: Grouped Column Trend (Cell 42408–42425)

**Markdown explanation (42408):**
> এই chart দেখায় প্রতিটি category-র জন্য 5টি performance metric (Accuracy, Precision, Recall, F1, AUC)। Screening Items category-তে সব metric অনেক বেশি — এটাই leakage-এর প্রমাণ। Familial ও Demographic category-তে metric তুলনামূলক কম।

```python
# fig9: Grouped Column Trend Chart
# X = categories, Groups = metrics (Accuracy, Precision, Recall, F1, AUC)
# Shows performance drop from Screening (91%) to Familial (47%) to Demographic (40%)

categories = ['Demographic', 'Familial', 'Personal Health', 'Neonatal Health', 'Screening Items']
metrics = {
    'Accuracy': [0.40, 0.47, 0.52, 0.61, 0.91],
    'Precision': [...],
    'Recall': [...],
    'F1': [...],
    'AUC': [...],
}

fig, ax = plt.subplots(figsize=(12, 6))
# Grouped bar chart
plt.savefig('/content/drive/MyDrive/thesis_ppd/fig9_grouped_column_trend.jpg', dpi=300)
print("✅ Saved: fig9_grouped_column_trend.jpg")
```

### CHART 2 — fig10: Multi-line Backward Elimination (Cell 42521–42538)

**Markdown explanation (42521):**
> Backward elimination-এর প্রতিটি step-এ 3টি metric (Accuracy, F1, AUC) trace করা হয়েছে। Elbow curve স্পষ্ট — 17 features-এ performance peak করে। এরপর features কমালে performance দ্রুত পড়ে যায়।

```python
# fig10: Multi-line backward elimination trajectory
# X = step number (0 to 64, = features removed)
# Lines = Accuracy, F1, AUC
# Highlight elbow at step 48 (17 features remaining)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(steps, accuracies, label='Accuracy', marker='o', markersize=3)
ax.plot(steps, f1_scores, label='F1 Score', marker='s', markersize=3)
ax.plot(steps, auc_scores, label='AUC', marker='^', markersize=3)
ax.axvline(x=48, color='red', linestyle='--', label='Optimal (Step 48 = 17 features)')
plt.savefig('/content/drive/MyDrive/thesis_ppd/fig10_multi_line_backward.jpg', dpi=300)
print("✅ Saved: fig10_multi_line_backward.jpg")
```

### CHART 3 — fig11: Stacked Column Tiers (Cell 42635–42652)

**Markdown explanation (42635):**
> প্রতিটি category-কে 3টি tier-এ ভাগ করা হয়েছে: High (>70%), Medium (50–70%), Low (<50%)। Screening Items = High tier, Neonatal Health = Medium-High tier, বাকিগুলো Low/Medium।

```python
# fig11: Stacked column chart — tier classification per category
# Each bar = category
# Stacked segments = % of features in High/Medium/Low performance tier

tiers = {
    'Demographic': {'High': 0, 'Medium': 20, 'Low': 80},
    'Familial': {'High': 0, 'Medium': 35, 'Low': 65},
    'Personal Health': {'High': 10, 'Medium': 40, 'Low': 50},
    'Neonatal Health': {'High': 30, 'Medium': 50, 'Low': 20},
    'Screening Items': {'High': 100, 'Medium': 0, 'Low': 0},
}

fig, ax = plt.subplots(figsize=(10, 6))
# Stacked bar chart
plt.savefig('/content/drive/MyDrive/thesis_ppd/fig11_stacked_column_tiers.jpg', dpi=300)
print("✅ Saved: fig11_stacked_column_tiers.jpg")
```

### CHART 4 — fig12: 3D Clustered Column (Cell 42751–42770) ← LAST CELL

**Markdown explanation (42751):**
> 3D bar chart দিয়ে Category × Metric × Score তিনটি dimension একসাথে দেখানো হয়েছে। matplotlib-এ 3D bar chart-এর কিছু সীমাবদ্ধতা আছে (overlap issue), তবে visual comparison-এর জন্য ব্যবহার করা হয়েছে।

```python
# fig12: 3D clustered column chart — FINAL FIGURE
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# X = categories (0–4)
# Y = metrics (0–4: Accuracy, Precision, Recall, F1, AUC)
# Z = score values

for i, cat in enumerate(categories):
    for j, metric in enumerate(metrics_list):
        ax.bar3d(i, j, 0, 0.4, 0.4, scores[cat][metric], 
                 color=colors[i], alpha=0.7)

ax.set_xlabel('Category')
ax.set_ylabel('Metric')
ax.set_zlabel('Score')
ax.set_title('3D Clustered Column: Category × Metric Performance')

plt.savefig('/content/drive/MyDrive/thesis_ppd/fig12_3d_clustered.jpg', dpi=300, bbox_inches='tight')
print("✅ Saved: fig12_3d_clustered.jpg")
```

---

## 23. Complete Cell Index (71 Cells)

| # | Line | Status | Content |
|---|------|--------|---------|
| 1 | 21 | ✅ | Google Drive mount |
| 2 | 50 | ✅ | All imports |
| 3 | 57 | ✅ | Load CSV → (800, 70) |
| 4 | 613 | ✅ | Drop 'sr' → (800, 69) |
| 5 | 624 | ✅ | Column rename + strip |
| 6 | 730 | ✅ | Markdown: "FRequncies" header |
| 7 | 739 | ✅ | Frequency analysis loop (all columns) |
| 8 | 15165 | ✅ | Education Level value fix |
| 9 | 15199 | ✅ | Husband's education level fix |
| 10 | 15233 | ✅ EMPTY | — |
| 11 | 15242 | ✅ | Markdown: "Total children column fixed" |
| 12 | 15251 | ✅ | Total children: 'More than two' → 'More than Two' |
| 13 | 15283 | ✅ | Markdown: "HIstory of pregnancy loss" |
| 14 | 15292 | ✅ | History of pregnancy loss: 'Still-born delivery' fix |
| 15 | 15324 | ✅ EMPTY | — |
| 16 | 15333 | ✅ | Disease before pregnancy fix |
| 17 | 15578 | ✅ EMPTY | — |
| 18 | 15595 | ✅ | **Disease before pregnancy: replace '1.2' → NaN** |
| 19 | 15834 | ✅ EMPTY | — |
| 20 | 15841 | ✅ EMPTY | — |
| 21 | 15850 | ✅ | **Fix inconsistent category values — df.replace({...})** |
| 22 | 15868 | ✅ EMPTY | — |
| 23 | 15885 | ✅ | **Diseases during pregnancy: standardize values** |
| 24 | 16116 | ✅ EMPTY | — |
| 25 | 16123 | ✅ EMPTY | — |
| 26 | 16132 | ✅ | **Clean column names (strip) + replacement_map labels** |
| 27 | 16318 | ✅ EMPTY | — |
| 28 | 16333 | ✅ | **Occupation: 'House wife' → 'Housewife'** |
| 29 | 16841 | ✅ | **Duplicate check → 0 duplicates, shape (800, 69)** |
| 30 | 16869 | ✅ | Markdown: "Unique values" header |
| 31 | 16878 | ✅ | Unique values loop (all columns) |
| 32 | 17048 | ✅ EMPTY | — |
| 33 | 17057 | ✅ | Post-cleaning frequency verification (2nd run) |
| 34 | 31241 | ✅ | Missing values table |
| 35 | 31532 | ✅ | Markdown: "Nymerical Feature handle" |
| 36 | 31541 | ✅ EMPTY | — |
| 37 | 31558 | ✅ | `def phq9_band()` — 5 severity levels |
| 38 | 31665 | ✅ | Numerical descriptive statistics |
| 39 | 31869 | ✅ | Save `PPD_data_updated_v34.csv` |
| 40 | 31894 | ✅ | EPDS Score/Result distribution + EPDS Audit (43 inconsistencies fixed) |
| 41 | 32741 | ✅ | Load v34 + LabelEncode all columns |
| 42 | 32904 | ✅ | Create `PPD_binary` (EPDS Score ≥ 13) |
| 43 | 33411 | ✅ | Chi2 ranking — all 65 features |
| 44 | 34546 | ✅ | Markdown: "RFFF" label |
| 45 | 34555 | ✅ | Markdown: "RFFF" label |
| 46 | 34564 | ✅ | Chi2 Backward Elimination → 17 optimal features (acc=0.9125) |
| 47 | 34905 | ✅ | 6-model comparison (LR, SVM, RF, XGB, KNN, MLP) |
| 48 | 35532 | ✅ | `!pip install lightgbm catboost` |
| 49 | 35579 | ✅ | **15-model comparison — Gradient Boosting wins (0.926)** |
| 50 | 36188 | ✅ | Dataset reload for category analysis |
| 51 | 36298 | ✅ | Feature category mapping (6 groups, 70 columns) |
| 52 | 36460 | ✅ | Category-wise model training (5 categories) |
| 53 | 36816 | ✅ | Correlation of features with EPDS Score (Pearson, Spearman, Kendall) |
| 54 | 37373 | ✅ | Multinomial LR coefficients |
| 55 | 37719 | ✅ | RF + XGBoost feature importance |
| 56 | 38130 | ✅ | Summary print |
| 57 | 38809 | ✅ EMPTY | — |
| 58 | 38816 | ✅ | Markdown: "new work" marker |
| 59 | 38825 | ✅ | All features: LR + RF + XGB + Leakage label table |
| 60 | 39743 | ✅ | **CELL 7 (NEW): Within-category importance (RF+XGB+LR+MI)** |
| 61 | 40718 | ✅ | **CELL 8 (NEW): Risk Factor via Kendall's tau (4 safe categories)** |
| 62 | 41831 | ✅ | Figure generation: fig1–fig8 |
| 63 | 42301 | ✅ | DATA GATHERING SUMMARY |
| 64 | 42408 | ✅ | Markdown: fig9 explanation |
| 65 | 42425 | ✅ | CHART 1: fig9 grouped column trend |
| 66 | 42521 | ✅ | Markdown: CHART 2 explanation (elbow at step 48) |
| 67 | 42538 | ✅ | CHART 2: fig10 multi-line backward elimination |
| 68 | 42635 | ✅ | Markdown: CHART 3 explanation (tier classification) |
| 69 | 42652 | ✅ | CHART 3: fig11 stacked column tiers |
| 70 | 42751 | ✅ | Markdown: CHART 4 explanation (3D limitation noted) |
| 71 | 42770 | ✅ | **CHART 4: fig12 3D clustered — LAST CELL** |

**Total: 71 cells | 11 empty | 60 non-empty | 60/60 READ ✅**

---

## 📊 Key Outputs Summary

| Output File | Description |
|-------------|-------------|
| `PPD_data_updated_v34.csv` | Final cleaned dataset (800 × 69) |
| `within_category_importance.csv` | RF+XGB+LR+MI scores per feature per category |
| `risk_factor_kendall_tau.csv` | Kendall's tau for all safe-category features |
| `fig1_category_performance.jpg` | Bar: accuracy per category |
| `fig2_kendall_tau.jpg` | Horizontal bar: risk factor tau values |
| `fig3_lr_coefficients.jpg` | LR coefficients (non-screening) |
| `fig4_rf_importance.jpg` | RF importance (non-screening) |
| `fig5_xgb_importance.jpg` | XGB importance (non-screening) |
| `fig6_within_category_heatmap.jpg` | Heatmap: method × feature scores |
| `fig7_screening_vs_others.jpg` | Leakage gap visualization |
| `fig8_correlation_heatmap.jpg` | Feature correlation heatmap |
| `fig9_grouped_column_trend.jpg` | Grouped bar: category × 5 metrics |
| `fig10_multi_line_backward.jpg` | Multi-line: backward elimination trajectory |
| `fig11_stacked_column_tiers.jpg` | Stacked bar: High/Medium/Low tiers |
| `fig12_3d_clustered.jpg` | 3D bar: category × metric × score |

---

## 🎯 Final Conclusions

1. **Data Leakage সমস্যা চিহ্নিত:** 17টি "optimal" features সবই EPDS/PHQ questionnaire items → target থেকেই আসছে → 91% accuracy মিথ্যা।

2. **বাস্তবসম্মত সেরা accuracy:** Neonatal Health features → **61%** (screening বাদে)

3. **সবচেয়ে গুরুত্বপূর্ণ risk factors (leakage মুক্ত):**
   - প্রসবের পর রাগ/ক্ষোভ (τ=+0.418)
   - মাতৃত্বের অনুভূতি (τ=+0.280)
   - নির্যাতন/Abuse (τ=-0.223)
   - গর্ভাবস্থায় বিষণ্নতা (PHQ2) (τ=+0.195)
   - পরিবারকে বিশ্বাস করতে পারা (τ=-0.177)

4. **সেরা ML Model:** Gradient Boosting (0.926 accuracy — কিন্তু leakage সহ)

5. **Professor-এর নতুন সংযোজন:**
   - CELL 7: Category-র ভেতরে feature ranking (4 methods combined)
   - CELL 8: Kendall's tau দিয়ে risk factor analysis

---

*Documentation generated from: `all_the_problem_solve_column (1).ipynb`*  
*Total cells: 71 | Total lines: ~42,875 | File size: ~67,537 tokens*
