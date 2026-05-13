# HIR-M3: Hierarchical Interaction Regularization Model

This repository contains the code, data, and documentation for the HIR-M3 model, an advanced framework for SDOH-Enhanced Readmission Prediction. This research focuses on quantifying the predictive value of Social Determinants of Health (SDOH) features on readmission models using hierarchical structural tiers.

## Data Overview

The primary data source for this research utilizes CMS OASIS-E (Outcome and Assessment Information Set) assessment data, focusing specifically on a Texas cohort. Evaluating this cohort illuminated systemic baseline differences when benchmarked against National CMS averages. We identified massive systematic under-coding in the Texas clinical data, including substantial relative drops in reported Diabetes (-38.4%), Heart Failure (-41.5%), and Hypertension (-52.7%). This statistical void crippled models relying solely on basic ICD codes, making the integration of demographic and geographic structural datasets mandatory.

## Methodology & Model Architecture

### Micro-Meso-Macro Hierarchy
To combat the shortcomings of conventional "flat" predictive models, features are explicitly grouped into three structural tiers:
*   **Micro**: Individual patient-level clinical variables (e.g., patient age, ICD diagnoses, basic demographics).
*   **Meso**: Proximal community/geographic contexts (e.g., total county poverty index, county-level Social Determinants of Health).
*   **Macro**: Broader systemic and healthcare disparities.

### HIR-M3 Architecture
The **HIR-M3** (Hierarchical Interaction Regularization) model is built on a foundation of Constraint-Aware Self-Attention. By using Multi-Head Self-Attention (MHSA), HIR-M3 inherently maps out how elements interact across the predefined structural environment.

### Hierarchical Interaction Regularization
The fundamental innovation within HIR-M3 is the implementation of the **HIR Penalty (λ=0.5)**. The HIR penalty acts mathematically to:
1.  **Suppress** attention explicitly directed *within* the same tier (intra-tier, such as Meso-to-Meso overfitting).
2.  **Reward** attention that crosses between tiers (cross-tier, such as evaluating how a patient's clinical state explicitly interacts with their community SDOH environment).

### Hybrid Ensemble Approach
The state-of-the-art final approach combined predictions from gradient boosting (LightGBM) and HIR-M3 via a strict weighted alpha parameter (90% LightGBM, 10% HIR-M3). This unifies the raw, elite precision of tabular gradient boosting with the structural probability calibration of regularized neural methods.

## Results

Below is the definitive evaluation of performance tracking from initial generation tests to the final optimized ensemble:

| Model Generation | Model / Strategy | ROC-AUC | PR-AUC | Brier Score | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Baseline Random Forest | 0.7385 | 0.3319 | 0.1177 | 0.2896 | 0.6442 | 0.3992 |
| **Baseline** | Baseline LightGBM | 0.7504 | 0.3545 | 0.2033 | 0.3040 | 0.6223 | 0.4082 |
| **Optimized** | **LightGBM (The Precision Engine)** | 0.8058 | 0.4295 | 0.1867 | 0.3623 | 0.6174 | 0.4566 |
| **Optimized** | **HIR-M3 (The Structural Expert)** | 0.8007 | 0.4120 | **0.1090** | 0.3584 | 0.6260 | 0.4558 |
| **Final** | **Hybrid Ensemble (10% HIR-M3 + 90% LGBM)**| **0.8065** | **0.4200** | 0.1206 | 0.3581 | 0.6320 | 0.4572 |

### Key Findings
*   **The Flat Tabular Blindspot**: While LightGBM achieved a high 0.8058 AUC, it yielded a Brier Score of 0.1867, indicating severe probability miscalibration.
*   **The HIR-M3 Calibration Mechanism**: Due to structural anchors, HIR-M3 successfully mapped cross-tier drivers and achieved a robust Brier Score of 0.1090.
*   **The Secret Sauce of the Ensemble**: A 10% HIR-M3 and 90% LightGBM weighting structure provided precisely the calibration metric the tabular engine was lacking, pushing the overall AUC to 0.8065 while maintaining realistic probability predictions.