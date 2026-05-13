import os
import sys
import pandas as pd
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_modeling import main, COHORTS_FILES, MODELS_TO_RUN

# Config
DATA_PATH = r'c:\Users\mirna\OneDrive\Desktop\oasis_data\version2\dashboardData_v2.csv'
TEMP_COHORT = "Ensemble_Test.csv"

def verify_ensemble_pipeline():
    print("--- Verifying Full Ensemble Pipeline ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Create a small sample
    print(f"Loading sample from {DATA_PATH}...")
    df_sample = pd.read_csv(DATA_PATH, nrows=2000)
    df_sample.to_csv(TEMP_COHORT, index=False)

    # Modify COHORTS_FILES for testing
    global COHORTS_FILES
    original_cohorts = COHORTS_FILES.copy()
    COHORTS_FILES.clear()
    COHORTS_FILES["Test Cohort"] = TEMP_COHORT
    
    # Run only a few models for speed
    MODELS_TO_RUN.clear()
    MODELS_TO_RUN.extend(["Random Forest", "LightGBM", "HIR-M3"])

    try:
        # Run main()
        print("Executing Pipeline Main Loop...")
        main()
        
        # Check results
        results_path = os.path.join("results", "unified_cv_results.csv")
        if os.path.exists(results_path):
            try:
                res_df = pd.read_csv(results_path)
                print("\nPipeline Execution Results:")
                cols = [c for c in ['Model', 'Test AUC'] if c in res_df.columns]
                print(res_df[cols].to_string())
                
                # Check if Hybrid models exist
                hybrids = res_df[res_df['Model'].str.contains("Hybrid", na=False)]
                if not hybrids.empty:
                    print(f"\n[SUCCESS] Found {len(hybrids)} hybrid ensembles in the results!")
                else:
                    print("\n[FAILURE] No hybrid ensembles found in results.")
            except Exception as e:
                print(f"\n[PARTIAL SUCCESS] Pipeline finished, but CSV reading failed: {e}")
                print("Checking the file manually...")
                with open(results_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-3:]: # Check last 3 lines
                        print(f"  Row: {line.strip()}")
                if any("Hybrid" in line for line in lines):
                    print("[SUCCESS] Hybrid results detected manually in CSV!")
        else:
            print("\n[FAILURE] results/unified_cv_results.csv was not created.")
            
    finally:
        # Cleanup and Restore
        if os.path.exists(TEMP_COHORT):
            os.remove(TEMP_COHORT)
        COHORTS_FILES.clear()
        COHORTS_FILES.update(original_cohorts)

if __name__ == "__main__":
    verify_ensemble_pipeline()
