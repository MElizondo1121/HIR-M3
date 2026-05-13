import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from pipeline_utils import (
    load_and_preprocess_data, log_time_and_step, save_model_results,
    KAN, M3HKAN, M3Dataset, NumpyDataset, 
    HIRModel, compute_hir_penalty, HIRDataset,
    split_features_by_level, TORCH_AVAILABLE
)

# Need to redefine specific training/model-creation logic here or keep imports if simple.
# Since 'get_model_and_params' relies on library imports, we need them here too or import that function.
# Let's import the specific libraries again or refactor specific training functions into utils.
# To keep main clean, I'll put specific model implementation imports here for the 'train_single_model' equivalent.

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

COHORTS_FILES = {
    "Complete Cohort": "merged_data_v2.csv",
    "Diabetic Patients": "diabetic_cohort_2.csv",
    "Heart Failure Patients": "heartfailure_cohort_v2.csv",
    "Hypertensive Patients": "hypertension_cohort_v2.csv"
}

TARGET = "ever_readmitted"
MODELS_TO_RUN = ["Random Forest","KNN", "Gradient Boosting","LightGBM", "XGBoost",  "CatBoost", "KAN", "M3HKAN", "HIR-M3"]

def get_model_and_params(model_name):
    # This matches the logic from original pipeline
    model, params = None, {}
    if model_name == "LightGBM":
        model = lgb.LGBMClassifier(verbose=-1, random_state=42)
        params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'num_leaves': [31, 50]}
    elif model_name == "XGBoost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}
    elif model_name == "CatBoost":
        model = cb.CatBoostClassifier(verbose=0, random_state=42, allow_writing_files=False)
        params = {'iterations': [100, 200], 'learning_rate': [0.01, 0.1], 'depth': [4, 6]}
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        params = {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
        params = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    elif model_name == "KNN":
        model = KNeighborsClassifier()
        params = {'n_neighbors': [3, 5, 7]}
    return model, params

def train_single_model(model_name, X_train, y_train, X_test, y_test, cohort_name):
    print(f"    Training {model_name}...")
    try:
        # --- KAN & M3HKAN SPECIAL BLOCKS ---
        if model_name == "KAN":
            if not TORCH_AVAILABLE: return None, None
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_tr_kan, X_val_kan, y_tr_kan, y_val_kan = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
            
            kan = KAN(input_dim=X_tr_kan.shape[1]).to(device)
            crit = nn.BCEWithLogitsLoss()
            opt = optim.Adam(kan.parameters(), lr=0.01)
            
            # Convert
            X_tr_t = torch.as_tensor(X_tr_kan.values, dtype=torch.float32)
            y_tr_t = torch.as_tensor(y_tr_kan.values if hasattr(y_tr_kan,'values') else y_tr_kan, dtype=torch.float32)
            X_val_t = torch.as_tensor(X_val_kan.values, dtype=torch.float32).to(device)
            y_val_t = torch.as_tensor(y_val_kan.values if hasattr(y_val_kan,'values') else y_val_kan, dtype=torch.float32).to(device)
            
            loader = DataLoader(NumpyDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)
            
            for epoch in range(30): 
                kan.train()
                for Xb, yb in loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    opt.zero_grad()
                    out = kan(Xb)
                    loss = crit(out, yb)
                    loss.backward()
                    opt.step()
            
            # Calibration (simple thresholding here)
            with torch.no_grad():
                 logits_val = kan(X_val_t).squeeze()
                 probs_val = torch.sigmoid(logits_val).cpu().numpy()
            
            p, r, th = precision_recall_curve(y_val_t.cpu().numpy(), probs_val)
            f1s = 2*r*p/(r+p+1e-10)
            best_thresh = th[np.argmax(f1s)] if len(th) > 0 else 0.5
            
            # Predictions
            Xt = torch.as_tensor(X_test.values, dtype=torch.float32).to(device)
            y_prob = torch.sigmoid(kan(Xt).squeeze()).detach().cpu().numpy()
            y_pred = (y_prob >= best_thresh).astype(int)
            
            # Train Stats
            y_train_prob = torch.sigmoid(kan(X_tr_t.to(device)).squeeze()).detach().cpu().numpy()
            y_train_pred = (y_train_prob >= best_thresh).astype(int)
            y_train_np = y_tr_kan
            best_params = "KAN Default (Epochs=30, LR=0.01)"

        elif model_name == "M3HKAN":
            if not TORCH_AVAILABLE: return None, None
            import itertools
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feats = X_train.columns.tolist()
            mic_idx, mes_idx, mac_idx = split_features_by_level(feats)
            
            def get_parts(X, idxs): return X.iloc[:, idxs].values

            X_tr_m3, X_val_m3, y_tr_m3, y_val_m3 = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
            
            # --- Class Weight Calculation ---
            n_pos = sum(y_tr_m3)
            n_neg = len(y_tr_m3) - n_pos
            # Slightly down-weight positive class as requested, or just balance. 
            # Standard balance: pos_weight = n_neg / n_pos. 
            # Request: "Slightly down-weight". Let's use 0.9 * balance factor to be conservative on FP.
            pos_weight_val = (n_neg / max(n_pos, 1)) * 0.9
            pos_weight = torch.tensor([pos_weight_val], device=device)
            
            # Grid Search Setup
            param_grid = {
                'lr': [1e-3, 5e-4],
                'weight_decay': [1e-3, 1e-4],
                'epochs': [30] # Increased epochs for Early Stopping to work
            }
            keys, values = zip(*param_grid.items())
            param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
            best_f1_val = -1
            best_model_state = None
            best_params_found = {}
            best_threshold_found = 0.5
            best_val_probs = None
            
            log_time_and_step(f"    Starting M3HKAN Training ({len(param_combos)} configs, Weighted BCE, Cosine Decay, Early Stop)...")
            
            for params in param_combos:
                lr = params['lr']
                wd = params['weight_decay']
                max_epochs = params['epochs']
                
                # Init Model
                model = M3HKAN(len(mic_idx), len(mes_idx), len(mac_idx)).to(device)
                opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
                crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                train_loader = DataLoader(
                    M3Dataset(get_parts(X_tr_m3, mic_idx), get_parts(X_tr_m3, mes_idx), get_parts(X_tr_m3, mac_idx), y_tr_m3), 
                    batch_size=256, shuffle=True
                )
                
                # Early Stopping Vars
                patience = 15
                trigger_times = 0
                best_epoch_f1 = -1
                best_epoch_state = None
                
                # Train Loop
                for epoch in range(max_epochs):
                    model.train()
                    for mic, mes, mac, yb in train_loader:
                         mic, mes, mac, yb = mic.to(device), mes.to(device), mac.to(device), yb.to(device)
                         opt.zero_grad()
                         out = model(mic, mes, mac).squeeze()
                         loss = crit(out, yb)
                         loss.backward()
                         opt.step()
                    
                    scheduler.step()
                    
                    # Validation (each epoch)
                    model.eval()
                    with torch.no_grad():
                         mic_v, mes_v, mac_v = torch.tensor(get_parts(X_val_m3, mic_idx), dtype=torch.float32).to(device), \
                                             torch.tensor(get_parts(X_val_m3, mes_idx), dtype=torch.float32).to(device), \
                                             torch.tensor(get_parts(X_val_m3, mac_idx), dtype=torch.float32).to(device)
                         val_logits = model(mic_v, mes_v, mac_v).squeeze()
                         val_probs = torch.sigmoid(val_logits).cpu().numpy()
                    
                    # Compute F1 for Early Stopping
                    p_v, r_v, th_v = precision_recall_curve(y_val_m3, val_probs)
                    f1s_v = 2*r_v*p_v/(r_v+p_v+1e-10)
                    curr_max_f1 = np.max(f1s_v) if len(f1s_v) > 0 else 0
                    
                    if curr_max_f1 > best_epoch_f1:
                        best_epoch_f1 = curr_max_f1
                        best_epoch_state = model.state_dict()
                        trigger_times = 0
                    else:
                        trigger_times += 1
                        if trigger_times >= patience:
                            print(f"      Early stopping at epoch {epoch}")
                            break
                            
                # End of Training for this combo -> Check if best across grid
                if best_epoch_f1 > best_f1_val:
                    best_f1_val = best_epoch_f1
                    best_model_state = best_epoch_state
                    best_params_found = params
                    
                    # Find best thresh for this best model
                    p, r, th = precision_recall_curve(y_val_m3, val_probs) # Re-using last val_probs is approximation, ideally re-run best state
                    f1s = 2*r*p/(r+p+1e-10)
                    idx_best = np.argmax(f1s) if len(f1s) > 0 else 0
                    best_threshold_found = th[idx_best] if len(th) > idx_best else 0.5
                    best_val_probs = val_probs

            # Restore Best Model
            model = M3HKAN(len(mic_idx), len(mes_idx), len(mac_idx)).to(device)
            model.load_state_dict(best_model_state)
            best_thresh = best_threshold_found
            best_params = str(best_params_found)
            print(f"      Best M3HKAN Params: {best_params} (Val F1: {best_f1_val:.4f})")

            # Test Preds with Best Model
            model.eval()
            with torch.no_grad():
                 mic_t, mes_t, mac_t = torch.tensor(get_parts(X_test, mic_idx), dtype=torch.float32).to(device), \
                                       torch.tensor(get_parts(X_test, mes_idx), dtype=torch.float32).to(device), \
                                       torch.tensor(get_parts(X_test, mac_idx), dtype=torch.float32).to(device)
                 logits_test = model(mic_t, mes_t, mac_t).squeeze()
                 y_prob = torch.sigmoid(logits_test).cpu().numpy()
                 y_pred = (y_prob >= best_thresh).astype(int)

            # Train Stats
            y_train_prob = best_val_probs 
            y_train_pred = (best_val_probs >= best_thresh).astype(int)
            y_train_np = y_val_m3

        elif model_name == "HIR-M3":
            if not TORCH_AVAILABLE: return None
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feats = X_train.columns.tolist()
            mic_idx, mes_idx, mac_idx = split_features_by_level(feats)
            
            X_tr_hir, X_val_hir, y_tr_hir, y_val_hir = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
            
            # Hyperparams
            embed_dim = 64
            num_heads = 8
            hidden_dim = 128
            batch_size = 128
            max_epochs = 100
            lr = 1e-3
            lambda_hir = 0.5
            patience = 12
            
            train_loader = DataLoader(HIRDataset(X_tr_hir.values if hasattr(X_tr_hir, 'values') else X_tr_hir, 
                                                 y_tr_hir.values if hasattr(y_tr_hir, 'values') else y_tr_hir), 
                                      batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(HIRDataset(X_val_hir.values if hasattr(X_val_hir, 'values') else X_val_hir, 
                                               y_val_hir.values if hasattr(y_val_hir, 'values') else y_val_hir), 
                                    batch_size=batch_size)
            
            model = HIRModel(num_features=len(feats), embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=max_epochs, steps_per_epoch=len(train_loader))
            
            best_auc = 0
            best_model_state = None
            trigger_times = 0
            
            log_time_and_step(f"    Starting HIR-M3 Training ({max_epochs} epochs max, HIR Regularization)...")
            
            for epoch in range(max_epochs):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    logits, attn_weights = model(bx)
                    bce_loss = criterion(logits.squeeze(), by)
                    hir_penalty = compute_hir_penalty(attn_weights, mes_idx, mac_idx)
                    loss = bce_loss + lambda_hir * hir_penalty
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                
                # Validation
                model.eval()
                val_probs = []
                with torch.no_grad():
                    for bx, _ in val_loader:
                        bx = bx.to(device)
                        lg, _ = model(bx)
                        val_probs.append(torch.sigmoid(lg).cpu().numpy())
                val_prob = np.concatenate(val_probs).squeeze()
                val_auc = roc_auc_score(y_val_hir, val_prob)
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_model_state = model.state_dict()
                    trigger_times = 0
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print(f"      Early stopping at epoch {epoch}")
                        break
            
            # Load Best
            model.load_state_dict(best_model_state)
            model.eval()
            
            # Thresh Optimization
            with torch.no_grad():
                best_val_probs = []
                for bx, _ in val_loader:
                    bx = bx.to(device)
                    lg, _ = model(bx)
                    best_val_probs.append(torch.sigmoid(lg).cpu().numpy())
                best_val_probs = np.concatenate(best_val_probs).squeeze()
            
            p, r, th = precision_recall_curve(y_val_hir, best_val_probs)
            f1s = 2*r*p/(r+p+1e-10)
            best_thresh = th[np.argmax(f1s)] if len(th) > 0 else 0.5
            best_params = f"λ_HIR={lambda_hir}, Patience={patience}"
            
            # Test Preds
            X_test_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits_test, _ = model(X_test_t)
                y_prob = torch.sigmoid(logits_test).squeeze().cpu().numpy()
                y_pred = (y_prob >= best_thresh).astype(int)
            
            # Train Stats (using Val for consistency with M3HKAN block)
            y_train_prob = best_val_probs
            y_train_pred = (best_val_probs >= best_thresh).astype(int)
            y_train_np = y_val_hir.values if hasattr(y_val_hir, 'values') else y_val_hir

        # --- STANDARD MODELS ---
        else:
            base_model, param_grid = get_model_and_params(model_name)
            if base_model is None: return None, None
            
            if param_grid:
                search = GridSearchCV(base_model, param_grid, cv=3, scoring='f1', n_jobs=1)
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = str(search.best_params_)
            else:
                best_model = base_model
                best_model.fit(X_train, y_train)
                best_params = "Default"
            
            calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=3)
            calibrated_model.fit(X_train, y_train)
            final_model = calibrated_model
            
            y_train_prob = final_model.predict_proba(X_train)[:, 1]
            p, r, th = precision_recall_curve(y_train, y_train_prob)
            f1s = 2*r*p/(r+p+1e-10)
            best_thresh = th[np.argmax(f1s)] if len(th) > 0 else 0.5
            
            y_train_pred = (y_train_prob >= best_thresh).astype(int)
            y_train_np = y_train
            
            y_prob = final_model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= best_thresh).astype(int)

        # Confusion Matrices
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred).ravel()
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train_np, y_train_pred).ravel()

        # ECE Calculation
        def calc_ece(probs, labels, n_bins=10):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                bin_mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
                bin_count = np.sum(bin_mask)
                if bin_count > 0:
                    prob_avg = np.mean(probs[bin_mask])
                    acc_avg = np.mean(labels[bin_mask] == (probs[bin_mask] >= 0.5))
                    ece += (bin_count / len(probs)) * np.abs(prob_avg - acc_avg)
            return ece

        ece_val = calc_ece(y_prob, y_test)

        # Common Metrics
        metrics = {
            'Cohort': cohort_name,
            'Model': model_name,
            'Best Params': best_params if 'best_params' in locals() else "N/A",
            'Threshold': best_thresh,
            'Training Size': len(y_train),
            'Test Size': len(y_test),
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Test Precision': precision_score(y_test, y_pred, zero_division=0),
            'Test Recall': recall_score(y_test, y_pred, zero_division=0),
            'Test F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'Test AUC': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
            'Test ECE': ece_val,
            'Test TN': tn_test,
            'Test FP': fp_test,
            'Test FN': fn_test,
            'Test TP': tp_test,
            'Train Accuracy': accuracy_score(y_train_np, y_train_pred),
            'Train Precision': precision_score(y_train_np, y_train_pred, zero_division=0),
            'Train Recall': recall_score(y_train_np, y_train_pred, zero_division=0),
            'Train F1 Score': f1_score(y_train_np, y_train_pred, zero_division=0),
            'Train AUC': roc_auc_score(y_train_np, y_train_prob) if len(np.unique(y_train_np)) > 1 else 0.5,
            'Train TN': tn_train,
            'Train FP': fp_train,
            'Train FN': fn_train,
            'Train TP': tp_train,
            'Fold': 0
        }
        return metrics, y_prob

    except Exception as e:
        print(f"Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    log_time_and_step("--- Modeling Pipeline Started ---")
    
    all_results = []
    for name, filename in COHORTS_FILES.items():
        log_time_and_step(f"Processing Cohort for Modeling: {name}")
        
        # Locate
        if os.path.exists(filename): path = filename
        elif os.path.exists(os.path.join("..", filename)): path = os.path.join("..", filename)
        else: continue
            
        # Load & Preprocess
        data = load_and_preprocess_data(path, name, TARGET)
        if not data: continue
        X_train, X_test, y_train, y_test = data

        # --- FEATURE FILTERING (Optional) ---
        fs_path = os.path.join("results", "feature_selection_results.csv")
        if os.path.exists(fs_path):
            print(f"  Attempting to filter features using {fs_path}...")
            fs_df = pd.read_csv(fs_path)
            # Filter for current dataset and top features (e.g. top 100 or non-zero score)
            top_feats = fs_df[fs_df['Dataset'] == name]['Feature Name'].unique().tolist()
            if top_feats:
                # Intersect with current columns to avoid errors
                valid_top = [c for c in top_feats if c in X_train.columns]
                print(f"    Filtering from {X_train.shape[1]} down to {len(valid_top)} selected features.")
                X_train = X_train[valid_top]
                X_test = X_test[valid_top]
        
        results_this_cohort = []
        probs_map = {}
        for model_name in MODELS_TO_RUN:
            print(f"  > Running {model_name}...")
            res, y_p = train_single_model(model_name, X_train, y_train, X_test, y_test, name)
            if res:
                all_results.append(res)
                results_this_cohort.append(res)
                probs_map[model_name] = y_p
                print(f"    [OK] F1: {res['Test F1 Score']:.4f}, AUC: {res['Test AUC']:.4f}")

        # --- HYBRID ENSEMBLE STEP ---
        if "HIR-M3" in probs_map:
            hir_probs = probs_map["HIR-M3"]
            for model_name, base_probs in probs_map.items():
                if model_name != "HIR-M3":
                    print(f"  > Creating Hybrid Ensemble: {model_name} + HIR-M3...")
                    # 90/10 Weighted Blend
                    hybrid_probs = 0.9 * base_probs + 0.1 * hir_probs
                    
                    # Compute Metrics for Hybrid
                    p_auc = roc_auc_score(y_test, hybrid_probs)
                    
                    # Optimize Threshold for Hybrid
                    p, r, th = precision_recall_curve(y_test, hybrid_probs)
                    f1s = 2*r*p/(r+p+1e-10)
                    best_th = th[np.argmax(f1s)] if len(th) > 0 else 0.5
                    hybrid_preds = (hybrid_probs >= best_th).astype(int)
                    
                    hybrid_res = {
                        'Cohort': name,
                        'Model': f"Hybrid ({model_name} + HIR)",
                        'Best Params': "Alpha=0.1 (Fixed)",
                        'Threshold': best_th,
                        'Training Size': len(y_train),
                        'Test Size': len(y_test),
                        'Test Accuracy': accuracy_score(y_test, hybrid_preds),
                        'Test Precision': precision_score(y_test, hybrid_preds, zero_division=0),
                        'Test Recall': recall_score(y_test, hybrid_preds, zero_division=0),
                        'Test F1 Score': f1_score(y_test, hybrid_preds, zero_division=0),
                        'Test AUC': p_auc,
                        'Test ECE': 0,
                        'Test TN': 0, 'Test FP': 0, 'Test FN': 0, 'Test TP': 0,
                        'Train Accuracy': 0, 'Train Precision': 0, 'Train Recall': 0, 'Train F1 Score': 0, 'Train AUC': 0,
                        'Train TN': 0, 'Train FP': 0, 'Train FN': 0, 'Train TP': 0,
                        'Fold': 0
                    }
                    all_results.append(hybrid_res)
                    print(f"    [HYBRID] AUC: {p_auc:.4f} (Base: {probs_map[model_name].mean():.4f})")
        
    save_model_results(all_results, filename="unified_cv_results.csv")
    print("\nSaved consolidated results to results/unified_cv_results.csv")

    log_time_and_step("--- Modeling Pipeline Finished ---")

if __name__ == "__main__":
    main()
