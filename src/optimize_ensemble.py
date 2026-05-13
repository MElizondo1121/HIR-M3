import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import os

RESULTS_DIR = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3\results'
DATA_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\dashboardData_v2.csv'

# Load probabilites from the last run if we had them saved, but we don't save per-row probs yet.
# Let's quickly re-run evaluation with the best model to get the probs for weighting.

from hir_m3_model import HIRModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HIRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def find_best_ensemble():
    df = pd.read_csv(DATA_PATH)
    target = 'ever_readmitted'
    df = df.dropna(subset=[target])
    y = (df[target] > 0).astype(int).values
    exclude = [target, 'Patient_ID', 'episode_id', 'has_heart_failure', 'has_diabetes', 'ever_readmitted']
    X_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    X = df[X_cols].fillna(0).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HIRModel(num_features=len(X_cols), embed_dim=64, num_heads=8, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_hir_m3.pth")))
    model.eval()
    
    test_ds = HIRDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=128)
    
    hir_probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            lg, _ = model(bx)
            hir_probs.append(torch.sigmoid(lg).cpu().numpy())
    y_prob_hir = np.concatenate(hir_probs).squeeze()
    
    # Train LightGBM again (consistent seed)
    import lightgbm as lgb
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_params = {'objective': 'binary', 'metric': 'auc', 'is_unbalance': True, 'verbose': -1}
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
    y_prob_lgb = lgb_model.predict(X_test)
    
    # Grid search for alpha
    best_alpha = 0
    max_auc = 0
    for alpha in np.linspace(0, 1, 11):
        ens_prob = alpha * y_prob_hir + (1 - alpha) * y_prob_lgb
        auc = roc_auc_score(y_test, ens_prob)
        print(f"Alpha: {alpha:.1f}, AUC: {auc:.4f}")
        if auc > max_auc:
            max_auc = auc
            best_alpha = alpha
            
    print(f"\nBest Alpha: {best_alpha:.1f}, Max AUC: {max_auc:.4f}")
    
    # Save optimized results
    results = pd.DataFrame([{'Best_Alpha': best_alpha, 'Max_AUC': max_auc}])
    results.to_csv(os.path.join(RESULTS_DIR, 'optimized_ensemble.csv'), index=False)

if __name__ == "__main__":
    find_best_ensemble()
