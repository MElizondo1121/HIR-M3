import os
import sys
import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score, f1_score, precision_recall_curve
from sklearn.utils import resample
from tqdm import tqdm

# Add parent dir to sys.path to import pipeline_utils
sys.path.append(os.path.abspath('.'))
from pipeline_utils import M3HKAN, split_features_by_level, M3Dataset

# Path configurations
DATA_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\dashboardData_v2.csv'
HIR_MODEL_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3\results\best_hir_m3.pth'
HIR_MODEL_DIR = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\modeling\take3'
sys.path.append(HIR_MODEL_DIR)
from hir_m3_model import HIRModel

DEVICE = torch.device("cpu") # Force CPU for stability in this environment

def bootstrap_metrics(y_true, y_prob, n_iterations=1000, model_name="Model"):
    stats = []
    
    # Calculate best threshold for F1 on the whole set first
    p, r, th = precision_recall_curve(y_true, y_prob)
    f1s = 2*r*p/(r+p+1e-10)
    best_thresh = th[np.argmax(f1s)] if len(th) > 0 else 0.5
    
    print(f"Bootstrapping {model_name}...")
    for _ in tqdm(range(n_iterations), desc=f"Bootstrap {model_name}"):
        indices = np.random.randint(0, len(y_true), len(y_true))
        y_true_resamp = y_true[indices]
        y_prob_resamp = y_prob[indices]
        
        if len(np.unique(y_true_resamp)) < 2:
            continue
            
        auc = roc_auc_score(y_true_resamp, y_prob_resamp)
        brier = brier_score_loss(y_true_resamp, y_prob_resamp)
        
        y_pred_resamp = (y_prob_resamp >= best_thresh).astype(int)
        recall = recall_score(y_true_resamp, y_pred_resamp, zero_division=0)
        f1 = f1_score(y_true_resamp, y_pred_resamp, zero_division=0)
        
        stats.append([auc, brier, recall, f1])
        
    stats = np.array(stats)
    results = {}
    metrics_names = ['AUC', 'Brier', 'Recall', 'F1']
    
    for i, name in enumerate(metrics_names):
        point_est = np.mean(stats[:, i])
        lower = np.percentile(stats[:, i], 2.5)
        upper = np.percentile(stats[:, i], 97.5)
        results[name] = f"{point_est:.4f} [{lower:.4f}-{upper:.4f}]"
        
    return results

def main():
    print("Loading data from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    target = 'ever_readmitted'
    df = df.dropna(subset=[target])
    y = (df[target] > 0).astype(int).values
    
    # Feature exclusion matching HIR-M3 run
    exclude = [target, 'Patient_ID', 'episode_id', 'has_heart_failure', 'has_diabetes', 'ever_readmitted']
    X_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    print(f"Total features identified: {len(X_cols)}")
    X = df[X_cols].fillna(0).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    results_table = []
    
    # 1. Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)
    rf_probs = rf.predict_proba(X_test_sc)[:, 1]
    results_table.append({'Model': 'Random Forest', **bootstrap_metrics(y_test, rf_probs, model_name="RF")})
    
    # 2. LightGBM
    print("\nTraining LightGBM...")
    lgb_train = lgb.Dataset(X_train, label=y_train) # LGBM handles raw values well
    lgb_params = {'objective': 'binary', 'metric': 'auc', 'is_unbalance': True, 'verbose': -1}
    lgb_core = lgb.train(lgb_params, lgb_train, num_boost_round=100)
    lgb_probs = lgb_core.predict(X_test)
    results_table.append({'Model': 'LightGBM', **bootstrap_metrics(y_test, lgb_probs, model_name="LGBM")})
    
    # 3. HIR-M3
    print("\nEvaluating HIR-M3...")
    hir_model = HIRModel(num_features=len(X_cols)).to(DEVICE)
    hir_model.load_state_dict(torch.load(HIR_MODEL_PATH, map_location=DEVICE))
    hir_model.eval()
    with torch.no_grad():
        xt_t = torch.tensor(X_test_sc, dtype=torch.float32).to(DEVICE)
        hir_logits, _ = hir_model(xt_t)
        hir_probs = torch.sigmoid(hir_logits).squeeze().cpu().numpy()
    results_table.append({'Model': 'HIR-M3', **bootstrap_metrics(y_test, hir_probs, model_name="HIR-M3")})
    
    # 4. M3HKAN (Retraining or logic)
    print("\nTraining M3HKAN...")
    mic_idx, mes_idx, mac_idx = split_features_by_level(X_cols)
    print(f"M3HKAN Tiers -> Micro: {len(mic_idx)}, Meso: {len(mes_idx)}, Macro: {len(mac_idx)}")
    
    # Debug: Print features in each tier if empty
    if len(mac_idx) == 0:
        mac_idx = [len(X_cols)-1]
    if len(mes_idx) == 0:
        mes_idx = [len(X_cols)-2]
    
    m3_model = M3HKAN(len(mic_idx), len(mes_idx), len(mac_idx)).to(DEVICE)
    optimizer = torch.optim.Adam(m3_model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    def get_p(X, idxs): return X[:, idxs]
    mic_train_t = torch.tensor(get_p(X_train_sc, mic_idx), dtype=torch.float32)
    mes_train_t = torch.tensor(get_p(X_train_sc, mes_idx), dtype=torch.float32)
    mac_train_t = torch.tensor(get_p(X_train_sc, mac_idx), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(10):
        m3_model.train()
        optimizer.zero_grad()
        out = m3_model(mic_train_t, mes_train_t, mac_train_t).squeeze()
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
    m3_model.eval()
    with torch.no_grad():
        mic_t, mes_t, mac_t = torch.tensor(get_p(X_test_sc, mic_idx), dtype=torch.float32), \
                             torch.tensor(get_p(X_test_sc, mes_idx), dtype=torch.float32), \
                             torch.tensor(get_p(X_test_sc, mac_idx), dtype=torch.float32)
        m3_probs = torch.sigmoid(m3_model(mic_t, mes_t, mac_t).squeeze()).cpu().numpy()
    results_table.append({'Model': 'M3HKAN', **bootstrap_metrics(y_test, m3_probs, model_name="M3HKAN")})
    
    # 5. Hybrid (90-10)
    print("\nEvaluating Hybrid 90-10...")
    hybrid_probs = 0.9 * lgb_probs + 0.1 * hir_probs
    results_table.append({'Model': 'Hybrid 90-10', **bootstrap_metrics(y_test, hybrid_probs, model_name="Hybrid")})
    
    # Compile and Display
    final_df = pd.DataFrame(results_table)
    print("\n--- Model Performance Comparison (95% CI) ---")
    print(final_df.to_markdown(index=False))
    
    final_df.to_csv('final_bootstrap_comparison.csv', index=False)

if __name__ == "__main__":
    main()
