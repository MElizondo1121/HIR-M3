# M3HKAN Technical Specification
**Multi-Modal Multi-Scale Hierarchical Knowledge-Aware Network**

## 1. Overview
M3HKAN is a hierarchical deep neural network designed to predict patient readmission risk by synthesizing data from three distinct levels of influence: **Micro** (Patient), **Meso** (Community), and **Macro** (System). It utilizes a specialized **Residual Bottleneck Architecture** to learn complex feature interactions while maintaining training stability.

## 2. Input Layers (Multi-Modal Feature Splitting)
The model receives a single patient vector and automatically routes features into three disjoint sub-networks:

| Stream | Description | Input Features (Examples) |
| :--- | :--- | :--- |
| **Micro (Patient)** | Biological & Clinical traits | Age, Gender, BMI, Comorbidities, ICD Clusters, Clinical utilization |
| **Meso (Community)** | Socioeconomic & Environmental | County Name, Poverty Rate, Education Level, Urban/Rural Status |
| **Macro (System)** | Healthcare Infrastructure | Home Health Agency ID, Facility Ownership Type, CMS Quality Ratings |

---

## 3. Sub-Network Architecture (Hierarchical Feature Extraction)
Each stream processes its features independently to learn level-specific latent representations before fusion.

### A. Micro Stream (Patient Level)
*   **Input Dimension:** $D_{micro}$ (288 features)
*   **Block 1 (Residual):** $D_{micro} \xrightarrow{\text{expand}} 128 \xrightarrow{\text{project}} 64$
*   **Activation:** SiLU $\rightarrow$ Dropout (0.1)
*   **Block 2 (Residual):** $64 \xrightarrow{\text{expand}} 64 \xrightarrow{\text{project}} 32$
*   **Output:** 32-dimensional patient embedding ($Z_{micro}$)

### B. Meso Stream (Community Level)
*   **Input Dimension:** $D_{meso}$ (90 features)
*   **Block 1 (Residual):** $D_{meso} \xrightarrow{\text{expand}} 64 \xrightarrow{\text{project}} 32$
*   **Activation:** SiLU $\rightarrow$ Dropout (0.1)
*   **Block 2 (Residual):** $32 \xrightarrow{\text{expand}} 32 \xrightarrow{\text{project}} 16$
*   **Output:** 16-dimensional community embedding ($Z_{meso}$)

### C. Macro Stream (System Level)
*   **Input Dimension:** $D_{macro}$ (42 features)
*   **Block 1 (Residual):** $D_{macro} \xrightarrow{\text{expand}} 64 \xrightarrow{\text{project}} 32$
*   **Activation:** SiLU $\rightarrow$ Dropout (0.1)
*   **Block 2 (Residual):** $32 \xrightarrow{\text{expand}} 32 \xrightarrow{\text{project}} 16$
*   **Output:** 16-dimensional system embedding ($Z_{macro}$)

---

## 4. The M3ResBlock (Core Component)
Every "Block" above follows a specific **Narrow $\rightarrow$ Wide $\rightarrow$ Narrow** logic with Residual connections:

1.  **Input:** $x$
2.  **Expansion (Linear):** Projects $x$ to $2 \times x$ dimensions.
3.  **Normalization:** LayerNorm.
4.  **Activation:** SiLU (Sigmoid Linear Unit).
5.  **Regularization:** Dropout (0.1).
6.  **Projection (Linear):** Projects back down to target output dimension.
7.  **Normalization:** LayerNorm.
8.  **Residual Add:** $Output = F(x) + Projection(x)$ (if dims change) OR $F(x) + x$.

---

## 5. Fusion & Output
The three latent representations are concatenated to form a holistic patient profile.

1.  **Concatenation:** $[Z_{micro}, Z_{meso}, Z_{macro}]$
    *   Dimensions: $32 + 16 + 16 = 64$ units.
2.  **Regularization:** Inter-block Dropout (0.15).
3.  **Fusion Block (Residual):**
    *   Input: 64 $\rightarrow$ Output: 32.
    *   Learns non-linear cross-level interactions (e.g., *High Risk Patient* + *Low Resource County*).
4.  **Output Head:**
    *   Single Linear Unit (32 $\rightarrow$ 1).
    *   Output: Logits (Raw Score).
5.  **Final Probability:** Sigmoid Function ($P(y=1)$).

---

## 6. Training Strategy
*   **Loss Function:** `BCEWithLogitsLoss` with `pos_weight` (Calculated dynamically to slightly down-weight the majority class for precision).
*   **Optimizer:** AdamW (Weight Decay enabled for regularization).
*   **Scheduling:** Cosine Annealing (Learning rate starts high and decays following a cosine curve).
*   **Early Stopping:** Monitors Validation F1 Score with a patience of 15 epochs.
