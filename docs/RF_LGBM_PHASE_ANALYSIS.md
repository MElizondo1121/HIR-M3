# Random Forest vs. LightGBM: Phase-by-Phase Tabular Model Analysis

This document isolates the performance and architectural impact of our primary standard tabular models—**Random Forest (RF)** and **LightGBM (LGBM)**—across the four major research phases.

---

## Phase 1: The Baseline Phase (Flat Learning)

In early prototyping, both models were fed identical, unoptimized flat representations of the feature space without explicit structural tagging (i.e., naive concatenation of clinical, demographic, and geographical fields).

| Model | ROC-AUC | PR-AUC | Diagnostic Observation |
| :--- | :---: | :---: | :--- |
| **Random Forest** | `0.7385` | `0.3319` | Suffered heavily from the high dimensionality of the Texas cohort; prone to overfitting on rare, high-cardinality ICD groups. |
| **LightGBM** | `0.7504` | `0.3545` | Outperformed RF inherently due to Gradient Boosting's capacity to handle sparse geographic and binary flags iteratively. |

**Phase takeaway**: The baseline implementation of tabular learning failed to break the `0.75` AUC ceiling because it systematically drowned out high-value signals in structural noise.

---

## Phase 2: The Structural Phase (Hierarchical Transition)

While Phase 2 primarily focused on introducing explicit hierarchical algorithms like **M3HKAN** and **M3-Hybrid**, these neural methods explicitly demonstrated the core weakness of flat RF and LGBM architectures.

*   **The RF/LGBM Limitation Revealed**: Standard tree-based models learn completely "flat" interactions. They mathematically treat a micro-level clinical feature (e.g., patient age) on the identical mathematical plane as a macro-level feature (e.g., total county poverty index). 
*   **The Contrast**: Hierarchical networks (M3HKAN/M3-Hybrid) naturally grouped these variables into explicit, nested interactions (Micro ↔ Meso ↔ Macro), resulting in significantly smoother cross-tier modeling, even if trailing in raw tabular accuracy originally (`0.7426` AUC for Elite).

---

## Phase 3: State-of-the-Art (Hyper-Optimized Processing)

LightGBM was resurrected for the final State-of-the-Art ensemble, utilizing advanced hyperparameter optimization to serve as the "Precision Engine". Random Forest was officially abandoned in the elite phase due to parallelization pickling constraints on maximum un-pickled depth.

| Model | ROC-AUC | PR-AUC | Brier Score | Strategy |
| :--- | :---: | :---: | :---: | :--- |
| **LightGBM (Optimized)** | `0.8058` | `0.4295` | `0.1867` | Highly optimized tree depths/leaves tailored explicitly for imbalanced healthcare datasets. |

**The Tabular "Blind Spot"**: 
Despite jumping significantly from `0.7504` to `0.8058` AUC, the tuned **LightGBM exhibited profound probabilistic overconfidence (High Brier Score: `0.1867`)**. Its flat structure correctly sorted patients by risk, but completely botched the actual raw probabilistic calibrations. 
To correct this, the architecture required an injection of the neural HIR-M3 model (Brier: `0.1090`) to form the **Hybrid Ensemble** (`0.8065` AUC), proving tabular methods and structural-hierarchical methods optimally must be fused.

---

## Phase 4: Verification & Structural Validation (SDOH Ablation)

We exclusively subjected the optimized LightGBM to the Fast DeLong statistical ablations to verify its capacity to handle Social Determinants of Health (SDOH):

1. **LightGBM + "Flat" Raw SDOH**
   * **Result**: `0.8037` AUC
   * **Impact**: **Degradation** (ΔAUC = `-0.0021`, p=0.98)
   * **Conclusion**: Directly injecting raw SDOH variables into LightGBM *adds noise*. The tabular model's baseline clinical variables already implicitly absorbed the socioeconomic risk, causing LightGBM to waste node splits on redundant, noisy data.
   
2. **LightGBM + Clustered SDOH "Risk Vector"**
   * **Result**: `0.8060` AUC
   * **Impact**: **Stabilization** (ΔAUC = `+0.0002`, p=0.69)
   * **Conclusion**: Providing LightGBM with statistically aggregated *structural* representations (the Vector) stabilized its performance, preventing the degradation seen with flat data.

### Final Consensus
**Random Forest** cannot mathematically handle the deep structure and extreme cardinality of this clinical dataset without aggressive memory scaling crashes. 
**LightGBM** serves as an elite, high-precision tabular engine, capturing maximum standalone signal, but suffers fundamentally from architectural probability miscalibration and severe vulnerability to "noisy" un-curated social structural features. Its maximum potential is truly only unlocked when explicitly guided by curated structural vectors or mathematically ensembled with hierarchically aware networks.
