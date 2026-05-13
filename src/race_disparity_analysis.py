import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import shap

from hir_m3_model import HIRModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\dashboardData_v2.csv'
RESULTS_DIR = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\results'
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3\results\best_hir_m3.pth'

class SHAPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        logits, _ = self.model(x)
        return logits

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    target = 'ever_readmitted'
    
    df = df.dropna(subset=[target])
    y = (df[target] > 0).astype(int).values
    
    exclude = [target, 'Patient_ID', 'episode_id', 'has_heart_failure', 'has_diabetes', 'ever_readmitted']
    X_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    
    print(f"Total features: {len(X_cols)}")
    X = df[X_cols].fillna(0).values
    
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Load Model
    print("Loading HIR-M3 model...")
    model = HIRModel(num_features=len(X_cols), embed_dim=64, num_heads=8, hidden_dim=128).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Model not found at", MODEL_PATH)
        print("Please run run_hir_m3.py first to train the model.")
        return
        
    model.eval()
    
    # Identify Race Columns
    race_cols = [c for c in X_cols if c in ['Asian', 'Black_or_African_American', 'White', 'Hispanic_or_Latino']]
    
    print("\n--- 1. Base Rate & Sample Size Audit ---")
    base_rates = []
    for rc in race_cols:
        subset = df_test[df_test[rc] == 1]
        n = len(subset)
        if n == 0: continue
        br = subset[target].mean()
        base_rates.append({'Race': rc, 'N_Test': n, 'True_Readmission_Rate': br})
    
    br_df = pd.DataFrame(base_rates)
    print(br_df)
    br_df.to_csv(os.path.join(RESULTS_DIR, 'race_base_rates.csv'), index=False)
    
    print("\n--- 2. Confusion Matrix Deconstruction ---")
    # Get Predictions
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        preds = (probs > 0.26).astype(int) # Using optimized threshold 0.26
        
    cm_metrics = []
    for rc in race_cols:
        idx = np.where(df_test[rc] == 1)[0]
        if len(idx) == 0: continue
        y_true_sub = y_test[idx]
        y_pred_sub = preds[idx]
        
        cm = confusion_matrix(y_true_sub, y_pred_sub, labels=[0, 1])
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn+fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn+tp) > 0 else 0
            
            cm_metrics.append({
                'Race': rc, 'Sensitivity': sensitivity, 'Specificity': specificity,
                'FPR': fpr, 'FNR': fnr
            })
            
    cm_df = pd.DataFrame(cm_metrics)
    print(cm_df)
    cm_df.to_csv(os.path.join(RESULTS_DIR, 'race_confusion_matrix.csv'), index=False)
    
    print("\n--- 3. Data Quality & Missingness Check ---")
    # Missingness in raw data before fillna for top clinical features
    # Let's check missingness on raw df_test for a few common features
    check_cols = [c for c in X_cols if 'Diagnosis' in c or 'Util' in c][:10]
    miss_res = []
    for rc in race_cols:
        subset = df_test[df_test[rc] == 1]
        if len(subset) == 0: continue
        for cc in check_cols:
            if cc in subset.columns:
                pct_miss = subset[cc].isnull().mean()
                miss_res.append({'Race': rc, 'Feature': cc, 'Pct_Missing': pct_miss})
                
    if len(miss_res) > 0:
        miss_df = pd.DataFrame(miss_res).pivot(index='Race', columns='Feature', values='Pct_Missing')
        print("Missingness by race:")
        print(miss_df)
        miss_df.to_csv(os.path.join(RESULTS_DIR, 'race_missingness.csv'))
    
    print("\n--- 4. Subgroup SHAP Analysis ---")
    try:
        wrapped_model = SHAPWrapper(model)
        # Background
        bg_idx = np.random.choice(len(X_train), 100, replace=False)
        background = torch.tensor(X_train[bg_idx], dtype=torch.float32).to(DEVICE)
        
        explainer = shap.DeepExplainer(wrapped_model, background)
        
        for rc in race_cols:
            idx = np.where(df_test[rc] == 1)[0]
            if len(idx) == 0: continue
            
            # Sample max 100 for SHAP to avoid memory/time issues
            sub_idx = np.random.choice(idx, min(100, len(idx)), replace=False)
            X_sub = torch.tensor(X_test[sub_idx], dtype=torch.float32).to(DEVICE)
            
            shap_values = explainer.shap_values(X_sub)
            
            # For BCE logits, shap_values might be a list or array
            sv = shap_values[0] if isinstance(shap_values, list) else shap_values
            
            plt.figure()
            shap.summary_plot(sv, X_test[sub_idx], feature_names=X_cols, show=False)
            plt.title(f"SHAP Summary - {rc}")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'shap_summary_{rc}.png'))
            plt.close()
            print(f"Saved SHAP plot for {rc}")
            
    except Exception as e:
        print(f"SHAP explanation failed: {e}")

if __name__ == '__main__':
    main()
