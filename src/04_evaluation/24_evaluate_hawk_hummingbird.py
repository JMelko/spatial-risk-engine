import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time

def evaluate_continuous_models():
    print("1. Loading Feature Matrix (This takes a moment)...")
    df = pd.read_parquet("data/processed/model_ready_features_v3.parquet")
    
    df.dropna(subset=['ndvi'], inplace=True)
    numerical_cols = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    for col in numerical_cols:
        df = df[df[col] != -9999.0]
        
    features = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    
    # Ensure spatial split is possible
    split_line = df['x_meters'].median()
    
    targets = {
        "Harriss_Hawk": {
            "sci_name": "Parabuteo unicinctus",
            "model_path": "data/processed/harriss_hawk_continuous_model.json",
            "prefix": "harriss_hawk"
        },
        "Annas_Hummingbird": {
            "sci_name": "Calypte anna",
            "model_path": "data/processed/annas_hummingbird_continuous_model.json",
            "prefix": "annas_hummingbird"
        }
    }

    for name, config in targets.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING: {name.replace('_', ' ')}")
        print(f"{'='*60}")
        
        y = (df['scientific_name'] == config['sci_name']).astype(int)
        
        # Load Model
        model = xgb.XGBClassifier(tree_method='hist', device='cuda')
        model.load_model(config['model_path'])
        
        # --- TEST 1 & 2: PDP and SHAP ---
        print("\n[Tests 1 & 2] Generating Interpretability Plots...")
        X_sample, _, y_sample, _ = train_test_split(df[features], y, train_size=20000, random_state=42, stratify=y)
        
        # PDP
        fig, ax = plt.subplots(figsize=(15, 8))
        PartialDependenceDisplay.from_estimator(
            model, X_sample, features=['elevation_m', 'slope_deg', 'distance_to_water_m'], ax=ax, grid_resolution=50
        )
        plt.suptitle(f"Partial Dependence Plots: {name.replace('_', ' ')}")
        plt.tight_layout()
        plt.savefig(f"data/processed/{config['prefix']}_pdp.png")
        plt.close()
        
        # SHAP (Using Native XGBoost Backend to bypass parser bug)
        model.set_params(device='cpu') 
        dmatrix = xgb.DMatrix(X_sample)
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
        shap_values = contribs[:, :-1]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(f"data/processed/{config['prefix']}_shap.png")
        plt.close()
        print(f"   -> Saved {config['prefix']}_pdp.png and {config['prefix']}_shap.png")

        # --- TEST 3: SPATIAL CROSS-VALIDATION ---
        print("\n[Test 3] Running Spatial Block Cross-Validation...")
        west_mask = df['x_meters'] < split_line
        
        X_west, y_west = df.loc[west_mask, features], y[west_mask]
        X_east, y_east = df.loc[~west_mask, features], y[~west_mask]
        
        # Train West -> Test East
        model_west = xgb.XGBClassifier(tree_method='hist', device='cuda', scale_pos_weight=(len(y_west)-y_west.sum())/y_west.sum())
        model_west.fit(X_west, y_west)
        auc_w_e = roc_auc_score(y_east, model_west.predict_proba(X_east)[:, 1])
        
        # Train East -> Test West
        model_east = xgb.XGBClassifier(tree_method='hist', device='cuda', scale_pos_weight=(len(y_east)-y_east.sum())/y_east.sum())
        model_east.fit(X_east, y_east)
        auc_e_w = roc_auc_score(y_west, model_east.predict_proba(X_west)[:, 1])
        
        print(f"   -> Fold 1 (Train West, Test East) AUC: {auc_w_e:.4f}")
        print(f"   -> Fold 2 (Train East, Test West) AUC: {auc_e_w:.4f}")

        # --- TEST 4: NULL MODEL BASELINE ---
        print("\n[Test 4] Running Null Spatial Baseline...")
        X_spatial = df[['x_meters', 'y_meters']]
        X_sp_train, X_sp_test, y_sp_train, y_sp_test = train_test_split(X_spatial, y, test_size=0.2, random_state=42, stratify=y)
        
        null_model = xgb.XGBClassifier(tree_method='hist', device='cuda', scale_pos_weight=(len(y_sp_train)-y_sp_train.sum())/y_sp_train.sum())
        null_model.fit(X_sp_train, y_sp_train)
        auc_null = roc_auc_score(y_sp_test, null_model.predict_proba(X_sp_test)[:, 1])
        
        # Standard Environmental Score for Comparison
        X_env_train, X_env_test, y_env_train, y_env_test = train_test_split(df[features], y, test_size=0.2, random_state=42, stratify=y)
        model.set_params(device='cuda')
        model.fit(X_env_train, y_env_train)
        auc_env = roc_auc_score(y_env_test, model.predict_proba(X_env_test)[:, 1])
        
        print(f"   -> Pure Spatial (Coordinates Only) AUC: {auc_null:.4f}")
        print(f"   -> Full Environmental Model AUC:        {auc_env:.4f}")

if __name__ == "__main__":
    evaluate_continuous_models()