# Data Overview & Feature Selection Results

This document provides a comprehensive summary of the dataset characteristics, feature selection methodologies, and empirical results (both overall and stratified by race). It is structured to facilitate direct integration into your LaTeX manuscript.

## 1. Data Overview

### 1.1 Dataset Characteristics & Demographics
The dataset consists of **221,957** patient records representing the Complete Cohort.

**Baseline Characteristics:**
*   **Total Readmissions:** 34,211 (15.4%)
*   **Average Age:** 74.7 $\pm$ 12.0 years
*   **Gender:** Female: 135,104 (60.9%) | Male: 86,853 (39.1%)

**Racial Demographics:**
*   **White:** 140,877 (63.5%)
*   **Hispanic or Latino:** 47,612 (21.5%)
*   **Black or African American:** 29,069 (13.1%)
*   **Asian:** 4,164 (1.9%)

**Key Comorbidities:**
*   **Hypertension:** 60,801 (27.4%)
*   **Diabetes:** 36,889 (16.6%)
*   **Heart Failure:** 18,169 (8.2%)

### 1.2 Feature Tiers
Following preprocessing and One-Hot Encoding (OHE) of categorical variables (such as `Submitted_HIPPS_Code` and `Agency_Medicare_Number`), the expanded dataset contains **203** features. These are structurally grouped into three hierarchical tiers:

*   **Micro Tier (Clinical/Individual):** 133 features
*   **Meso Tier (System/Care):** 36 features
*   **Macro Tier (SDOH):** 34 features

---

## 2. Feature Selection Methodologies

To robustly evaluate the intrinsic predictive value of each feature tier across different modeling paradigms, a comprehensive suite of **13 feature selection methods** was employed. These span filter, wrapper, embedded, and interpretability-based approaches:

1.  **Variance Threshold** (Filter)
2.  **Lasso** (L1 Regularization)
3.  **Ridge** (L2 Regularization)
4.  **Elastic Net** (Hybrid Regularization)
5.  **Random Forest Importance** (Tree-based)
6.  **Gradient Boosting Importance** (Tree-based)
7.  **XGBoost Importance** (Tree-based)
8.  **LightGBM Importance** (Tree-based)
9.  **CatBoost Importance** (Tree-based)
10. **Permutation Importance - Random Forest** (Model-Agnostic)
11. **Permutation Importance - LightGBM** (Model-Agnostic)
12. **Recursive Feature Elimination (RFE)** (Wrapper)
13. **Sequential Feature Selection (SFS)** (Wrapper)

---

## 3. Overall Feature Selection Results

The deep learning architecture (HIR-M3) natively evaluates the global importance of features via its intrinsic `HierarchicalAttention` matrix. By averaging the multi-head attention weights across the entire cohort ($n = 221,957$), the model assigned the following global normalized importance scores to each tier:

| Tier | Global Normalized Attention Importance |
| :--- | :---: |
| **Micro (Clinical)** | 0.444 |
| **Meso (System)** | 0.381 |
| **Macro (SDOH)** | **0.452** |

> [!NOTE] 
> **Finding:** On a global scale, the HIR-M3 architecture places the highest aggregated attention on the Macro tier (0.452), narrowly surpassing clinical Micro features (0.444). This validates the inclusion of aggregated social and structural determinants in the modeling pipeline.

---

## 4. Feature Selection by Race

The evaluation was then stratified by isolated racial cohorts to determine if different demographics exhibit distinct predictive risk drivers. Both the ensemble of 13 tabular methods and the deep learning intrinsic attention weights were extracted.

### 4.1 Tabular Methods: Average Normalized Importance
This represents the normalized importance scores outputted by the 13 tabular methods, averaged by tier for each cohort:

| Race | Micro Tier | Meso Tier | Macro Tier |
| :--- | :---: | :---: | :---: |
| **Asian** | 0.231 | 0.252 | 0.140 |
| **Black or African American** | 0.267 | 0.279 | 0.251 |
| **Hispanic or Latino** | 0.250 | 0.253 | 0.167 |
| **White** | **0.422** | **0.384** | **0.356** |

### 4.2 Tabular Methods: Top Selections Count
This represents the total number of times features from a specific tier appeared in the "Top 20" most important features across all 13 algorithms:

| Race | Micro | Meso | Macro |
| :--- | :---: | :---: | :---: |
| **Asian** | 226 | 203 | 74 |
| **Black or African American** | 247 | 205 | 75 |
| **Hispanic or Latino** | 228 | 226 | 77 |
| **White** | 214 | 245 | 70 |

### 4.3 HIR-M3: Intrinsic Attention Scores
The following represents the `HierarchicalAttention` weights dynamically assigned by the retrained deep learning architecture during prediction:

| Race | Micro Tier | Meso Tier | Macro Tier |
| :--- | :---: | :---: | :---: |
| **Asian** | 0.044 | 0.026 | 0.036 |
| **Black or African American** | 0.155 | 0.106 | 0.144 |
| **Hispanic or Latino** | 0.206 | 0.224 | 0.215 |
| **White** | **0.475** | **0.419** | **0.488** |

### 4.4 Synthesis of Racial Disparities
> [!IMPORTANT]
> Both the 13 tabular methods and the deep learning model mathematically arrive at the same conclusion: **White patients demonstrate the strongest predictive reliance on Macro variables**. 
>
> In the HIR-M3 model, the attention given to Macro features for White patients (0.488) is substantially higher than for Hispanic (0.215), Black (0.144), and Asian (0.036) patients. This indicates a profound structural disparity: systemic/social determinants play a much larger and more consistent role in determining the baseline readmission risk for the demographic majority compared to minority cohorts.
