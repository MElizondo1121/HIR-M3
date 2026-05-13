import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Adjust paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline_utils import split_features_by_level

from hir_m3_model import HIRModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\dashboardData_v2.csv'
MODEL_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3\results\best_hir_m3.pth'
RESULTS_DIR = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3\results'

def get_attention_importance(model, X, batch_size=512):
    """
    Passes data through HIR-M3 and averages the Multi-Head Attention matrices.
    Returns normalized importance scores for each feature.
    """
    model.eval()
    all_attn = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            # HIRModel returns (logits, attn_weights)
            _, attn = model(batch) 
            # attn shape: (Batch, N, N)
            # We average over the batch dimension
            avg_attn_batch = attn.mean(dim=0).cpu().numpy()
            all_attn.append(avg_attn_batch * len(batch))
            
    # Compute global weighted average of attention matrices
    global_attn_matrix = np.sum(all_attn, axis=0) / len(X)
    
    # Feature importance is the sum of attention it pays to other features (or receives)
    # Since it's symmetric-ish or at least row-sums to 1 usually, we can sum over columns 
    # to see how much total attention a feature receives from all other features.
    # We will use sum across rows (axis=0) which means "how much attention this feature receives"
    feature_importance = global_attn_matrix.sum(axis=0)
    
    # Normalize 0-1
    if np.max(feature_importance) > np.min(feature_importance):
        feature_importance = (feature_importance - np.min(feature_importance)) / (np.max(feature_importance) - np.min(feature_importance))
    else:
        feature_importance = np.ones_like(feature_importance)
        
    return feature_importance

def main():
    import pickle
    import os
    cache_path = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3\results\processed_data.pkl'
    if os.path.exists(cache_path):
        print("Loading preprocessed data from cache...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Loading data using pipeline_utils...")
        from pipeline_utils import load_and_preprocess_data
        target = 'ever_readmitted'
        data = load_and_preprocess_data(DATA_PATH, "Complete Cohort", target)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            
    if data is None:
        print("Data loading failed.")
        return
    X_train_df, X_test_df, y_train_s, y_test_s = data
    
    # We will compute feature importance on the entire set
    X_df = pd.concat([X_train_df, X_test_df])
    X_cols = X_train_df.columns.tolist()
    
    # We don't need to scale again because load_and_preprocess_data already scales
    X_scaled = X_df.values
    
    # Map features to tiers
    mic_idx, mes_idx, mac_idx = split_features_by_level(X_cols)
    idx_to_tier = {}
    for i in mic_idx: idx_to_tier[i] = 'Micro'
    for i in mes_idx: idx_to_tier[i] = 'Meso'
    for i in mac_idx: idx_to_tier[i] = 'Macro'
    
    print("Loading HIR-M3 Model...")
    model = HIRModel(num_features=len(X_cols)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    groups = ['Overall', 'Asian', 'Black_or_African_American', 'Hispanic_or_Latino', 'White']
    results = []
    
    for group in groups:
        print(f"\nEvaluating Attention Importance for: {group}")
        
        if group == 'Overall':
            X_subset = X_scaled
        else:
            if group not in X_cols:
                print(f"Skipping {group}, column not found in OHE.")
                continue
            subset_mask = X_df[group].values > 0.5
            if subset_mask.sum() < 10:
                print(f"Skipping {group}, insufficient samples.")
                continue
            X_subset = X_scaled[subset_mask]
            
        print(f"Sample size: {len(X_subset)}")
        
        # Get importances
        importances = get_attention_importance(model, X_subset)
        
        for idx, score in enumerate(importances):
            results.append({
                'Cohort': group,
                'Feature Index': idx,
                'Feature Name': X_cols[idx],
                'Tier': idx_to_tier.get(idx, 'Unknown'),
                'Normalized Importance': score
            })
            
    # Aggregate
    res_df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res_df.to_csv(os.path.join(RESULTS_DIR, 'hir_attention_importances_raw.csv'), index=False)
    
    summary_df = res_df.groupby(['Cohort', 'Tier'])['Normalized Importance'].mean().reset_index()
    pivot_df = summary_df.pivot(index='Cohort', columns='Tier', values='Normalized Importance')
    
    print("\n=======================================================")
    print("   HIR-M3 Feature Attention (Importance) by Tier")
    print("=======================================================")
    print(pivot_df.to_string())
    
    pivot_df.to_csv(os.path.join(RESULTS_DIR, 'hir_attention_importances_summary.csv'))
    print(f"\nResults saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
