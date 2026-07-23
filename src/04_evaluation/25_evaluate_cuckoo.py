import pandas as pd
import geopandas as gpd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def evaluate_cuckoo_model():
    print("1. Loading Grid Features and Spatial Geometries...")
    features_df = pd.read_parquet("data/processed/cuckoo_grid_features.parquet")
    grid_gdf = gpd.read_file("data/processed/cuckoo_grid_geometries.gpkg")
    
    print("2. Calculating Grid Centroids for Spatial Testing...")
    # Calculate the exact center coordinate (X and Y) directly on the GeoDataFrame
    grid_gdf['x_meters'] = grid_gdf.geometry.centroid.x
    grid_gdf['y_meters'] = grid_gdf.geometry.centroid.y
    
    # Merge only the raw coordinates back onto the tabular feature matrix
    df = features_df.merge(grid_gdf[['grid_id', 'x_meters', 'y_meters']], on='grid_id')
    
    features = ['elevation_mean', 'slope_mean', 'aspect_mean', 'ndvi_max', 'water_dist_min']
    X = df[features]
    y = df['target_presence']
    
    print("3. Loading Pre-Trained Cuckoo Model...")
    model = xgb.XGBClassifier(tree_method='hist', device='cuda')
    model.load_model("data/processed/yellow_billed_cuckoo_grid_model.json")
    
    # --- TEST 1 & 2: PDP and SHAP ---
    print("\n[Tests 1 & 2] Generating Interpretability Plots...")
    # We can use an 80% split for our sample since 31,866 grids easily fits in memory
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
    
    # PDP (Focusing on Elevation, NDVI Max, and Water Distance)
    fig, ax = plt.subplots(figsize=(15, 8))
    PartialDependenceDisplay.from_estimator(
        model, X_sample, features=['elevation_mean', 'ndvi_max', 'water_dist_min'], ax=ax, grid_resolution=50
    )
    plt.suptitle("Partial Dependence Plots: Yellow-billed Cuckoo (Macro-Grid)")
    plt.tight_layout()
    plt.savefig("data/processed/yellow_billed_cuckoo_pdp.png")
    plt.close()
    
    # SHAP
    model.set_params(device='cpu')
    dmatrix = xgb.DMatrix(X_sample)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_values = contribs[:, :-1]
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("data/processed/yellow_billed_cuckoo_shap.png")
    plt.close()
    print("   -> Saved yellow_billed_cuckoo_pdp.png and yellow_billed_cuckoo_shap.png")
    
    # --- TEST 3: SPATIAL CROSS-VALIDATION ---
    print("\n[Test 3] Running Spatial Block Cross-Validation (East vs. West)...")
    split_line = df['x_meters'].median()
    west_mask = df['x_meters'] < split_line
    
    X_west, y_west = X[west_mask], y[west_mask]
    X_east, y_east = X[~west_mask], y[~west_mask]
    
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
    
    # Standard Grid Environmental Score for Comparison
    model.set_params(device='cuda')
    X_env_train, X_env_test, y_env_train, y_env_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_env_train, y_env_train)
    auc_env = roc_auc_score(y_env_test, model.predict_proba(X_env_test)[:, 1])
    
    print(f"   -> Pure Spatial (Grid Centroids Only) AUC: {auc_null:.4f}")
    print(f"   -> Full Grid Environmental Model AUC:      {auc_env:.4f}")

if __name__ == "__main__":
    evaluate_cuckoo_model()