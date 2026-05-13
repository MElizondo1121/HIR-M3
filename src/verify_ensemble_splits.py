import os
import sys
import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from hir_m3_model import HIRModel

# File configuration
DATA_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\dashboardData_v2.csv'
HIR_MODEL_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3\results\best_hir_m3.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    target = 'ever_readmitted'
    df = df.dropna(subset=[target])
    y = (df[target] > 0).astype(int).values
    
    exclude = [target, 'Patient_ID', 'episode_id', 'has_heart_failure', 'has_diabetes', 'ever_readmitted']
    X_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    X = df[X_cols].fillna(0).values
    
    # Scale dataset for training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Loading Pre-Trained HIR-M3 Model (The Structural Regularizer)...")
    hir_model = HIRModel(num_features=len(X_cols)).to(DEVICE)
    hir_model.load_state_dict(torch.load(HIR_MODEL_PATH, map_location=DEVICE))
    hir_model.eval()

    # Set up Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Tabular Precision Engines to Test
    tabular_models = {
        'LightGBM': lambda: lgb.LGBMClassifier(is_unbalance=True, n_estimators=100, random_state=42, verbose=-1),
        'XGBoost': lambda: XGBClassifier(scale_pos_weight=(len(y)-sum(y))/max(1, sum(y)), n_estimators=100, random_state=42, eval_metric='logloss'),
        'CatBoost': lambda: CatBoostClassifier(auto_class_weights='Balanced', iterations=100, random_state=42, verbose=0)
    }

    # Ensemble Splits to cross-validate (Alpha = Weight given to the tabular engine)
    # 0.9 = 90% Tabular / 10% HIR-M3
    alphas = [1.0, 0.9, 0.8, 0.5] 
    
    results = []

    print("\nStarting 5-Fold Cross-Validation verifying optimal ensemble splits...\n")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        print(f"--- FOLD {fold+1} ---")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Get structural probabilties from HIR-M3
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            logits, _ = hir_model(X_test_t)
            prob_hir = torch.sigmoid(logits).squeeze().cpu().numpy()

        for model_name, model_fn in tabular_models.items():
            print(f"    Training {model_name}...")
            clf = model_fn()
            clf.fit(X_train, y_train)
            
            prob_tab = clf.predict_proba(X_test)[:, 1]
            
            for alpha in alphas:
                prob_ensemble = alpha * prob_tab + (1 - alpha) * prob_hir
                
                auc = roc_auc_score(y_test, prob_ensemble)
                pr_auc = average_precision_score(y_test, prob_ensemble)
                brier = brier_score_loss(y_test, prob_ensemble)
                
                results.append({
                    'Fold': fold + 1,
                    'Model Strategy': model_name,
                    'Split Ratio (Tabular% - HIR%)': f"{int(alpha*100)}% - {int((1-alpha)*100)}%",
                    'Alpha': alpha,
                    'AUC': auc,
                    'PR-AUC': pr_auc,
                    'Brier': brier
                })
                
    # Aggregate outputs into a summary table
    df_res = pd.DataFrame(results)
    df_summary = df_res.groupby(['Model Strategy', 'Split Ratio (Tabular% - HIR%)', 'Alpha'])[['AUC', 'PR-AUC', 'Brier']].mean().reset_index()
    
    # Sort to show the best performing split per model
    df_summary = df_summary.sort_values(by=['Model Strategy', 'AUC'], ascending=[True, False]).drop(columns=['Alpha'])
    
    print("\n================ FINAL ENSEMBLE SPLIT VERIFICATION ================\n")
    print(df_summary.to_markdown(index=False))
    
    os.makedirs('results', exist_ok=True)
    out_path = 'results/ensemble_splits_verification.csv'
    df_summary.to_csv(out_path, index=False)
    print(f"\nSaved raw aggregated cross-validation metrics to {out_path}")

if __name__ == "__main__":
    main()
