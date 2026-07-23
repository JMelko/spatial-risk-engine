import pandas as pd
import geopandas as gpd
import xgboost as xgb

def generate_cuckoo_hsi():
    print("1. Loading Grid Features and Spatial Geometries...")
    features_df = pd.read_parquet("data/processed/cuckoo_grid_features.parquet")
    grid_gdf = gpd.read_file("data/processed/cuckoo_grid_geometries.gpkg")

    print("2. Loading Trained Cuckoo Grid Model...")
    model = xgb.XGBClassifier(tree_method='hist', device='cuda')
    model.load_model("data/processed/yellow_billed_cuckoo_grid_model.json")

    print("3. Predicting HSI Probabilities on the GPU...")
    # Ensure the features are in the exact same order as training
    features = ['elevation_mean', 'slope_mean', 'aspect_mean', 'ndvi_max', 'water_dist_min']
    X = features_df[features]
    
    # Predict continuous probability of presence (Class 1)
    probs = model.predict_proba(X)[:, 1]
    
    # Attach the probabilities back to our tabular data
    features_df['hsi_probability'] = probs
    
    print("4. Joining HSI Scores to the Spatial Grid...")
    # Merge the probabilities back to the spatial polygons using the shared grid_id
    hsi_gdf = grid_gdf.merge(
        features_df[['grid_id', 'hsi_probability']], 
        on='grid_id', 
        how='inner'
    )
    
    print("5. Exporting Final HSI GeoPackage...")
    output_path = "data/processed/yellow_billed_cuckoo_hsi_5070.gpkg"
    hsi_gdf.to_file(output_path, driver="GPKG")
    
    print(f"-> Success! Cuckoo HSI Grid saved to {output_path}")

if __name__ == "__main__":
    generate_cuckoo_hsi()