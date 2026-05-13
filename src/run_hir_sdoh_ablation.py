import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, recall_score, f1_score, precision_recall_curve, precision_score

# Add parent directory to path to import feature splitting logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from unified_analysis_pipeline import split_features_by_level
except ImportError:
    # Fallback
    def split_features_by_level(features):
        return [], [], []

from hir_m3_model import HIRModel, compute_hir_penalty

# Configurations
DATA_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\dashboardData_v2.csv'
RESULTS_DIR = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3\results'
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_hir(X_train, y_train, w_train, X_test, y_test, w_test, X_cols, config):
    micro_idx, meso_idx, macro_idx = split_features_by_level(X_cols)
    
    train_ds = HIRDataset(X_train, y_train, w_train)
    test_ds = HIRDataset(X_test, y_test, w_test)
    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'])
    
    model = HIRModel(num_features=len(X_cols), 
                    embed_dim=config['EMBED_DIM'], 
                    num_heads=config['NUM_HEADS'], 
                    hidden_dim=config['HIDDEN_DIM']).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config['LR'], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['LR'], epochs=config['EPOCHS'], steps_per_epoch=len(train_loader)
    )
    
    best_auc = 0
    trigger_times = 0
    best_model_path = os.path.join(RESULTS_DIR, "temp_best_hir.pth")
    
    for epoch in range(config['EPOCHS']):
        model.train()
        for batch_x, batch_y, batch_w in train_loader:
            batch_x, batch_y, batch_w = batch_x.to(DEVICE), batch_y.to(DEVICE), batch_w.to(DEVICE)
            optimizer.zero_grad()
            logits, attn_weights_batch = model(batch_x)
            
            bce_loss_unweighted = criterion(logits.squeeze(), batch_y)
            bce_loss = (bce_loss_unweighted * batch_w).mean()
            
            hir_loss = compute_hir_penalty(attn_weights_batch, meso_idx, macro_idx)
            loss = bce_loss + config['LAMBDA_HIR'] * hir_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
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
            torch.save(model.state_dict(), best_model_path)
        else:
            trigger_times += 1
            if trigger_times >= config['PATIENCE']:
                break
                
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_x, _, _ in test_loader:
            batch_x = batch_x.to(DEVICE)
            logits, _ = model(batch_x)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        
    return np.concatenate(all_probs).squeeze()

def train_lgb(X_train, y_train, X_test, y_test):
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
        'bagging_freq': 5,
        'random_state': 42
    }
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
    return lgb_model.predict(X_test)

def bootstrap_metrics(y_true, y_prob, n_bootstraps=1000):
    rng = np.random.RandomState(42)
    aucs, praucs, briers, precisions_boot, recalls, f1s = [], [], [], [], [], []
    
    # Find optimal threshold for F1 on the full test set
    precisions, recalls_pr, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls_pr) / (precisions + recalls_pr + 1e-8)
    opt_idx = np.argmax(f1_scores)
    # the threshold array has one less element than precisions/recalls
    opt_thresh = thresholds[opt_idx] if opt_idx < len(thresholds) else 0.5
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        y_true_b = y_true[indices]
        y_prob_b = y_prob[indices]
        y_pred_b = (y_prob_b > opt_thresh).astype(int)
        
        aucs.append(roc_auc_score(y_true_b, y_prob_b))
        praucs.append(average_precision_score(y_true_b, y_prob_b))
        briers.append(brier_score_loss(y_true_b, y_prob_b))
        precisions_boot.append(precision_score(y_true_b, y_pred_b, zero_division=0))
        recalls.append(recall_score(y_true_b, y_pred_b))
        f1s.append(f1_score(y_true_b, y_pred_b))
        
    def get_ci(metric_list):
        mean = np.mean(metric_list)
        lower = np.percentile(metric_list, 2.5)
        upper = np.percentile(metric_list, 97.5)
        return f"{mean:.4f} [{lower:.4f}-{upper:.4f}]"
        
    return get_ci(aucs), get_ci(praucs), get_ci(briers), get_ci(precisions_boot), get_ci(recalls), get_ci(f1s)

def main():
    import pickle
    cache_path = os.path.join(RESULTS_DIR, 'processed_data.pkl')
    if os.path.exists(cache_path):
        print("Loading preprocessed data from cache...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Loading data using pipeline_utils...")
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from pipeline_utils import load_and_preprocess_data
        data = load_and_preprocess_data(DATA_PATH, "Complete Cohort", 'ever_readmitted')
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            
    if data is None:
        print("Data loading failed.")
        return
        
    X_train_df, X_test_df, y_train_s, y_test_s = data
    
    # Downsample
    print("Downsampling to 20,000 samples to speed up training...")
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_train_df), min(20000, len(X_train_df)), replace=False)
    X_train_df = X_train_df.iloc[sample_idx]
    y_train = np.array(y_train_s)[sample_idx]
    y_test = np.array(y_test_s)
    
    all_features = X_train_df.columns.tolist()
    
    # Define feature subsets
    baseline_feats = [c for c in all_features if not c.startswith('ACS_') and 'POP' not in c and 'COUNTY' not in c.upper()]
    flat_sdoh_feats = [c for c in all_features if c.startswith('ACS_') or 'POP' in c]
    baseline_plus_flat = baseline_feats + flat_sdoh_feats
    
    high_level_sdoh_keywords = [
        'ACS_PCT_PERSON_INC_BELOW99', 'ACS_PCT_POV_WHITE', 'ACS_PCT_POV_BLACK', # Poverty
        'ACS_PCT_LT_HS', 'ACS_PCT_BACHELOR_DGR', # Education
        'ACS_PCT_HH_NO_INTERNET', # Digital divide / Housing
        'ACS_PCT_HH_FOOD_STMP_BLW_POV', # Food security
        'ACS_PCT_HH_1PERS', 'ACS_PCT_HH_ALONE_ABOVE65', # Isolation
        'POPPCT_RUR' # Rurality
    ]
    sdoh_vector_feats = [c for c in all_features if any(k in c for k in high_level_sdoh_keywords)]
    baseline_plus_vector = baseline_feats + sdoh_vector_feats
    
    feature_sets = {
        'Baseline': baseline_feats,
        'Base+Flat': baseline_plus_flat,
        'Base+Vector': baseline_plus_vector
    }
    
    config = {
        'EMBED_DIM': 64,
        'NUM_HEADS': 8,
        'HIDDEN_DIM': 128,
        'BATCH_SIZE': 128,
        'EPOCHS': 5,
        'LR': 1e-3,
        'LAMBDA_HIR': 0.5,
        'PATIENCE': 2
    }
    
    results = []
    
    for name, feats in feature_sets.items():
        print(f"\nEvaluating Feature Set: {name} ({len(feats)} features)", flush=True)
        
        X_train_sub = X_train_df[feats].values
        X_test_sub = X_test_df[feats].values
        
        # Sample weights
        w_train = np.ones(len(y_train), dtype=np.float32)
        w_test = np.ones(len(y_test), dtype=np.float32)
        if 'Asian' in feats:
            asian_idx = feats.index('Asian')
            w_train[X_train_sub[:, asian_idx] > 0.5] = 5.0
            w_test[X_test_sub[:, asian_idx] > 0.5] = 5.0
            
        print("Training LightGBM...")
        y_prob_lgb = train_lgb(X_train_sub, y_train, X_test_sub, y_test)
        
        print("Training HIR-M3...")
        y_prob_hir = train_hir(X_train_sub, y_train, w_train, X_test_sub, y_test, w_test, feats, config)
        
        # Eval LGB
        lgb_auc_ci, lgb_prauc_ci, lgb_brier_ci, lgb_prec_ci, lgb_rec_ci, lgb_f1_ci = bootstrap_metrics(y_test, y_prob_lgb)
        results.append({
            'Feature Set': name,
            'Model': 'LightGBM',
            'ROC-AUC [95% CI]': lgb_auc_ci,
            'PR-AUC [95% CI]': lgb_prauc_ci,
            'Brier [95% CI]': lgb_brier_ci,
            'Precision [95% CI]': lgb_prec_ci,
            'Recall [95% CI]': lgb_rec_ci,
            'F1 Score [95% CI]': lgb_f1_ci
        })
        
        # Eval HIR
        hir_auc_ci, hir_prauc_ci, hir_brier_ci, hir_prec_ci, hir_rec_ci, hir_f1_ci = bootstrap_metrics(y_test, y_prob_hir)
        results.append({
            'Feature Set': name,
            'Model': 'HIR-M3',
            'ROC-AUC [95% CI]': hir_auc_ci,
            'PR-AUC [95% CI]': hir_prauc_ci,
            'Brier [95% CI]': hir_brier_ci,
            'Precision [95% CI]': hir_prec_ci,
            'Recall [95% CI]': hir_rec_ci,
            'F1 Score [95% CI]': hir_f1_ci
        })
        
        # Eval Ensemble (90% LGBM, 10% HIR-M3)
        y_prob_ens = 0.9 * y_prob_lgb + 0.1 * y_prob_hir
        ens_auc_ci, ens_prauc_ci, ens_brier_ci, ens_prec_ci, ens_rec_ci, ens_f1_ci = bootstrap_metrics(y_test, y_prob_ens)
        results.append({
            'Feature Set': name,
            'Model': 'Ensemble (90/10)',
            'ROC-AUC [95% CI]': ens_auc_ci,
            'PR-AUC [95% CI]': ens_prauc_ci,
            'Brier [95% CI]': ens_brier_ci,
            'Precision [95% CI]': ens_prec_ci,
            'Recall [95% CI]': ens_rec_ci,
            'F1 Score [95% CI]': ens_f1_ci
        })

    results_df = pd.DataFrame(results)
    print("\n--- RESULTS ---")
    print(results_df.to_string(index=False))
    
    out_path = os.path.join(RESULTS_DIR, 'hir_sdoh_ablation_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

if __name__ == '__main__':
    main()
