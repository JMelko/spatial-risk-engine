import os
import rasterio
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_raster_stack_sampled(sample_fraction=0.1):
    print("1. Loading aligned raster stack into memory...")
    stack_dir = "data/processed/aligned_stack"
    
    files = {
        "dem": "aligned_az_nm_dem_5070.tif",
        "slope": "aligned_slope_5070.tif",
        "aspect": "aligned_aspect_5070.tif",
        "distance_to_water": "aligned_distance_to_water_5070.tif",
        "ndvi": "aligned_ndvi_5070_resampled.tif"
    }
    
    arrays = {}
    mask = None
    
    for key, filename in files.items():
        filepath = os.path.join(stack_dir, filename)
        with rasterio.open(filepath) as src:
            data = src.read(1)
            arrays[key] = data
            
            valid_pixel = (~np.isnan(data)) & (data != -9999.0)
            if mask is None:
                mask = valid_pixel
            else:
                mask = mask & valid_pixel

    print("2. Extracting and randomly sampling valid pixels to protect RAM...")
    valid_indices = np.where(mask)
    total_valid = len(valid_indices[0])
    print(f"   Total valid 30m ground pixels: {total_valid:,}")
    
    # Take a controlled random sample (e.g., 10%) to fit comfortably in RAM and train instantly on GPU
    sample_size = int(total_valid * sample_fraction)
    print(f"   Downsampling to {sample_fraction*100}% ({sample_size:,} pixels) for robust modeling...")
    
    rng = np.random.default_rng(seed=42)
    sampled_idx = rng.choice(total_valid, size=sample_size, replace=False)
    
    df_data = {}
    for key, data in arrays.items():
        # Index directly using the sampled valid coordinates
        flat_data = data[valid_indices]
        df_data[key] = flat_data[sampled_idx]
        
    df = pd.DataFrame(df_data)
    return df

def train_model_gpu(df):
    print("3. Preparing features and target proxy...")
    X = df[['dem', 'slope', 'aspect', 'distance_to_water']]
    y = df['ndvi']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("4. Training XGBoost Regressor on NVIDIA RTX 4070 (CUDA)...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        tree_method='hist',
        device='cuda',  # Harnesses your RTX 4070 GPU
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("5. Evaluating model performance...")
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"   Model R² Score: {r2:.4f}")
    print(f"   Root Mean Squared Error: {rmse:.4f}")
    
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(importance.to_string(index=False))

if __name__ == "__main__":
    dataframe = load_raster_stack_sampled(sample_fraction=0.1) # Safe 10% sample
    train_model_gpu(dataframe)