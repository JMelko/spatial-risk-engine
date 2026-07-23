import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def run_null_models():
    print("1. Loading Feature Matrix...")
    df = pd.read_parquet("data/processed/model_ready_features_v3.parquet")
    
    # Standard clean up
    df.dropna(subset=['ndvi'], inplace=True)
    numerical_cols = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    for col in numerical_cols:
        df = df[df[col] != -9999.0]
        
    print("2. Isolating Target (Mexican Jay)...")
    y = (df['scientific_name'] == 'Aphelocoma wollweberi').astype(int)
    
    # We strip out all environmental variables and only keep spatial coordinates
    X_spatial = df[['x_meters', 'y_meters']]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_spatial, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n--- NULL MODEL 1: Pure Random Guess ---")
    # Generate completely random probabilities between 0 and 1
    np.random.seed(42)
    random_probs = np.random.rand(len(y_test))
    random_auc = roc_auc_score(y_test, random_probs)
    print(f"   -> Random Guess ROC-AUC: {random_auc:.4f}")
    
    print("\n--- NULL MODEL 2: Pure Spatial Memorization ---")
    print("   -> Training XGBoost using ONLY X/Y coordinates (no environmental data)...")
    
    pos_cases = y_train.sum()
    imbalance_ratio = (len(y_train) - pos_cases) / pos_cases
    
    spatial_model = xgb.XGBClassifier(
        tree_method='hist', 
        device='cuda',             
        scale_pos_weight=imbalance_ratio, 
        random_state=42
    )
    
    spatial_model.fit(X_train, y_train)
    spatial_probs = spatial_model.predict_proba(X_test)[:, 1]
    spatial_auc = roc_auc_score(y_test, spatial_probs)
    
    print(f"   -> Spatial Only ROC-AUC: {spatial_auc:.4f}")
    
    print("\n==================================================")
    print("FINAL EVALUATION COMPARISON")
    print("==================================================")
    print(f"Baseline 1 (Random Guess):          {random_auc:.4f}")
    print(f"Baseline 2 (Pure Geography):        {spatial_auc:.4f}")
    print(f"Our Environmental Model:            0.9575")
    print("==================================================")

if __name__ == "__main__":
    run_null_models()