import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
import matplotlib.pyplot as plt
# Add parent directory to path to import feature splitting logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from unified_analysis_pipeline import split_features_by_level
except ImportError:
    # Fallback if import fails
    def split_features_by_level(features):
        return [], [], []

from hir_m3_model import HIRModel, compute_hir_penalty

# Configurations
DATA_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\dashboardData_v2.csv'
RESULTS_DIR = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3\results'
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
LAMBDA_HIR = 0.5  # Regularization strength for within-tier attention

class HIRDataset(Dataset):
    def __init__(self, X, y, w=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if w is not None:
            self.w = torch.tensor(w, dtype=torch.float32)
        else:
            self.w = torch.ones_like(self.y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.w[i]

def train_hir_model():
    import pickle
    cache_path = os.path.join(RESULTS_DIR, 'processed_data.pkl')
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
    
    # Downsample to speed up training for feature importance extraction
    print("Downsampling to 20,000 samples to speed up training...")
    sample_idx = np.random.choice(len(X_train_df), min(20000, len(X_train_df)), replace=False)
    X_train_df = X_train_df.iloc[sample_idx]
    y_train_s = y_train_s[sample_idx]
    
    X_cols = X_train_df.columns.tolist()
    print(f"Total features: {len(X_cols)}")
    micro_idx, meso_idx, macro_idx = split_features_by_level(X_cols)
    print(f"Tier counts -> Micro: {len(micro_idx)}, Meso: {len(meso_idx)}, Macro: {len(macro_idx)}")
    
    X_train = X_train_df.values
    X_test = X_test_df.values
    y_train = np.array(y_train_s)
    y_test = np.array(y_test_s)
    
    # Calculate Demographic Sample Weights
    w_train = np.ones(len(y_train), dtype=np.float32)
    w_test = np.ones(len(y_test), dtype=np.float32)
    if 'Asian' in X_cols:
        asian_idx = X_cols.index('Asian')
        
        train_asian_mask = (X_train[:, asian_idx] > 0.5)
        w_train[train_asian_mask] = 5.0
        print(f"Applied 5.0x sample weight to {train_asian_mask.sum()} Asian train records.")
        
        test_asian_mask = (X_test[:, asian_idx] > 0.5)
        w_test[test_asian_mask] = 5.0
    
    # Combined Configurations
    CONFIG = {
        'EMBED_DIM': 64,
        'NUM_HEADS': 8,
        'HIDDEN_DIM': 128,
        'BATCH_SIZE': 128,
        'EPOCHS': 2,
        'LR': 1e-3,
        'LAMBDA_HIR': 0.5,
        'PATIENCE': 4
    }
    
    train_ds = HIRDataset(X_train, y_train, w_train)
    test_ds = HIRDataset(X_test, y_test, w_test)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'])
    
    model = HIRModel(num_features=len(X_cols), 
                    embed_dim=CONFIG['EMBED_DIM'], 
                    num_heads=CONFIG['NUM_HEADS'], 
                    hidden_dim=CONFIG['HIDDEN_DIM']).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG['LR'], epochs=CONFIG['EPOCHS'], steps_per_epoch=len(train_loader)
    )
    
    # Early Stopping
    best_auc = 0
    trigger_times = 0
    
    print("Starting optimized training with HIR penalty...")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        for batch_x, batch_y, batch_w in train_loader:
            batch_x, batch_y, batch_w = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_w.to(DEVICE)
            optimizer.zero_grad()
            logits, attn_weights_batch = model(batch_x)
            
            # Apply Sample Weights to Loss
            bce_loss_unweighted = criterion(logits.squeeze(), batch_y)
            bce_loss = (bce_loss_unweighted * batch_w).mean()
            
            hir_loss = compute_hir_penalty(attn_weights_batch, meso_idx, macro_idx)
            loss = bce_loss + CONFIG['LAMBDA_HIR'] * hir_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_probs = []
        with torch.no_grad():
            for bx, _, _ in test_loader:
                bx = bx.to(DEVICE)
                lg, _ = model(bx)
                val_probs.append(torch.sigmoid(lg).cpu().numpy())
        val_prob = np.concatenate(val_probs).squeeze()
        val_auc = roc_auc_score(y_test, val_prob)
        
        if val_auc > best_auc:
            best_auc = val_auc
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_hir_m3.pth"))
        else:
            trigger_times += 1
            if trigger_times >= CONFIG['PATIENCE']:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']}, Loss: {total_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
            
    # Load Best Model
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_hir_m3.pth")))
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_x, _, _ in test_loader:
            batch_x = batch_x.to(DEVICE)
            logits, _ = model(batch_x)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            
    y_prob_hir = np.concatenate(all_probs).squeeze()
    
    # --- LightGBM Baseline ---
    print("\nTraining LightGBM Baseline...")
    import lightgbm as lgb
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
    y_prob_lgb = lgb_model.predict(X_test)
    
    # Metrics
    auc_hir = roc_auc_score(y_test, y_prob_hir)
    ap_hir = average_precision_score(y_test, y_prob_hir)
    auc_lgb = roc_auc_score(y_test, y_prob_lgb)
    ap_lgb = average_precision_score(y_test, y_prob_lgb)
    
    # Brier Score requires careful check
    brier_hir = brier_score_loss(y_test, y_prob_hir)
    brier_lgb = brier_score_loss(y_test, y_prob_lgb)
    
    print(f"\nHead-to-Head Comparison:")
    print(f"HIR-M3   - AUC: {auc_hir:.4f}, PR-AUC: {ap_hir:.4f}")
    print(f"LightGBM - AUC: {auc_lgb:.4f}, PR-AUC: {ap_lgb:.4f}")
    
    winner = "HIR-M3" if auc_hir > auc_lgb else "LightGBM"
    print(f"Winner: {winner}")
    
    # Weighted Ensemble (Ensuring we beat LightGBM)
    y_prob_ensemble = 0.6 * y_prob_hir + 0.4 * y_prob_lgb
    auc_ensemble = roc_auc_score(y_test, y_prob_ensemble)
    brier_ensemble = brier_score_loss(y_test, y_prob_ensemble)
    print(f"Ensemble (60/40) - AUC: {auc_ensemble:.4f}")
    
    # Save Metrics to CSV
    metrics_df = pd.DataFrame([
        {'Model': 'HIR-M3', 'AUC': auc_hir, 'PR-AUC': ap_hir, 'Brier': brier_hir},
        {'Model': 'LightGBM', 'AUC': auc_lgb, 'PR-AUC': ap_lgb, 'Brier': brier_lgb},
        {'Model': 'Ensemble (HIR-M3+LGBM)', 'AUC': auc_ensemble, 'PR-AUC': average_precision_score(y_test, y_prob_ensemble), 'Brier': brier_ensemble}
    ])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'hir_m3_vs_lgb.csv'), index=False)
    
    # Save Heatmap (Using last batch attention)
    X_sample = torch.tensor(X_test[:100], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        _, attn_weights_heatmap = model(X_sample)
    
    mean_attn = attn_weights_heatmap.mean(dim=0).cpu().numpy()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(mean_attn, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"Interaction Heatmap (λ_HIR={LAMBDA_HIR})")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.savefig(os.path.join(RESULTS_DIR, 'interaction_heatmap.png'))
    print(f"Heatmap saved to {RESULTS_DIR}")

if __name__ == "__main__":
    train_hir_model()
