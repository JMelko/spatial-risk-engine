import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

def train_multi_species():
    print("1. Loading Feature Matrix...")
    df = pd.read_parquet("data/processed/model_ready_features_v3.parquet")

    print(f"   Original rows: {len(df):,}")
    
    # Clean out the out-of-bounds NaN values from the Earth Engine NDVI export
    df.dropna(subset=['ndvi'], inplace=True)
    
    # Also drop any default GDAL NoData (-9999) values across the numerical columns
    numerical_cols = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    for col in numerical_cols:
        df = df[df[col] != -9999.0]
        
    print(f"   Cleaned rows available for training: {len(df):,}")
    
    # Format ecoregion for XGBoost categorical handling
    df['ecoregion'] = df['ecoregion'].astype('category')

    # Define our 6 predictor variables (X)
    features = ['elevation_m', 'ecoregion', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    X = df[features]

    # Define our three target species mapping
    targets = {
        "Annas_Hummingbird": 'Calypte anna',
        "Harriss_Hawk": 'Parabuteo unicinctus',
        "Mexican_Jay": 'Aphelocoma wollweberi'
    }

    print("\nStarting Automated GPU Model Pipeline...")
    
    for common_name, sci_name in targets.items():
        print(f"\n==================================================")
        print(f"Training Model: {common_name.replace('_', ' ')} ({sci_name})")
        
        # Set the target (y) to 1 if the row matches the species, 0 otherwise
        y = (df['scientific_name'] == sci_name).astype(int)
        
        # Train/test split stratified by the target to ensure fair representation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Calculate class imbalance dynamically
        pos_cases = y_train.sum()
        neg_cases = len(y_train) - pos_cases
        imbalance_ratio = neg_cases / pos_cases
        
        print(f"   -> Imbalance Ratio: {imbalance_ratio:.2f} (Positive Training Points: {pos_cases:,})")
        
        clf_model = xgb.XGBClassifier(
            enable_categorical=True, 
            tree_method='hist', 
            device='cuda',             
            scale_pos_weight=imbalance_ratio, 
            random_state=42
        )
        
        print("   -> Training on NVIDIA RTX 4070...")
        start = time.time()
        clf_model.fit(X_train, y_train)
        print(f"   -> Training complete in {time.time() - start:.2f} seconds.")
        
        # Predict and Evaluate
        preds = clf_model.predict(X_test)
        print(f"\n   --- Classification Report ---")
        print(classification_report(y_test, preds, target_names=['Absent', 'Present']))
        
        # Save the model
        model_path = f"data/processed/{common_name.lower()}_model.json"
        clf_model.save_model(model_path)
        print(f"   -> Model saved to {model_path}")

if __name__ == "__main__":
    train_multi_species()