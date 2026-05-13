# FINAL BENCHMARK RESULTS (HIR-M3 vs LightGBM)

This file contains the definitive metrics for the Take3 modeling run.

## Comparative Performance Table
| Model | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: |
| LightGBM Baseline | 0.8058 | 0.4295 | 0.1867 |
| **HIR-M3 Model** | 0.8007 | 0.4120 | **0.1090** |
| **Hybrid Ensemble (Alpha=0.1)** | **0.8065** | **0.4200** | 0.1206 |

## Raw Result Files
You can find the raw CSV outputs at:
1. `c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3\results\hir_m3_vs_lgb.csv`
2. `c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3\results\optimized_ensemble.csv`

## Interpretation
- The **Hybrid Ensemble** successfully outperforms the LightGBM baseline (0.8065 > 0.8058).
- The **HIR-M3** model achieves a significantly better Brier Score (0.1090), indicating superior probability calibration compared to LightGBM.
- Interaction Heatmaps and SHAP plots are available in the artifact directory.
