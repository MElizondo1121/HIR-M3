import os
import sys
import pandas as pd
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline_utils import load_and_preprocess_data, split_features_by_level
from run_modeling import train_single_model

# Config
DATA_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\dashboardData_v2.csv'
COHORT_NAME = "Verification Test"

def verify_integration():
    print("--- Verifying HIR-M3 Integration ---")
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Load a small sample for speed
    print(f"Loading sample from {DATA_PATH}...")
    df_sample = pd.read_csv(DATA_PATH, nrows=5000)
    sample_path = "temp_sample.csv"
    df_sample.to_csv(sample_path, index=False)

    try:
        # Preprocess
        print("Preprocessing sample...")
        data = load_and_preprocess_data(sample_path, COHORT_NAME, target='ever_readmitted')
        if not data:
            print("Preprocessing failed.")
            return
        
        X_train, X_test, y_train, y_test = data
        
        # Run HIR-M3
        print("Training HIR-M3 through unified pipeline...")
        # Now returns (metrics, y_prob)
        res, y_p = train_single_model("HIR-M3", X_train, y_train, X_test, y_test, COHORT_NAME)
        
        if res:
            print(f"\n[SUCCESS] HIR-M3 trained successfully!")
            print(f"AUC: {res['Test AUC']:.4f}")
            print(f"F1: {res['Test F1 Score']:.4f}")
        else:
            print("\n[FAILURE] HIR-M3 training returned None.")
            
    finally:
        if os.path.exists(sample_path):
            os.remove(sample_path)

if __name__ == "__main__":
    verify_integration()
