import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import time

def train_continuous_models():
    print("1. Loading Feature Matrix...")
    df = pd.read_parquet("data/processed/model_ready_features_v3.parquet")
    
    # Clean out the out-of-bounds NaN values from the NDVI export and standard GDAL NoData
    df.dropna(subset=['ndvi'], inplace=True)
    numerical_cols = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    for col in numerical_cols:
        df = df[df[col] != -9999.0]
        
    print(f"   Cleaned rows available for training: {len(df):,}")
    
    # Define our 5 CONTINUOUS predictor variables (dropping ecoregion entirely)
    features = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    X = df[features]

    targets = {
        "Annas_Hummingbird": 'Calypte anna',
        "Harriss_Hawk": 'Parabuteo unicinctus',
        "Mexican_Jay": 'Aphelocoma wollweberi'
    }

    print("\nStarting Continuous-Only GPU Model Pipeline...")
    
    for common_name, sci_name in targets.items():
        print(f"\n{'='*50}")
        print(f"Training & Analyzing: {common_name.replace('_', ' ')} ({sci_name})")
        
        y = (df['scientific_name'] == sci_name).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        pos_cases = y_train.sum()
        neg_cases = len(y_train) - pos_cases
        imbalance_ratio = neg_cases / pos_cases
        
        # Notice we removed enable_categorical=True since we are only passing numeric floats now
        clf_model = xgb.XGBClassifier(
            tree_method='hist', 
            device='cuda',             
            scale_pos_weight=imbalance_ratio, 
            random_state=42
        )
        
        print("   -> Training on NVIDIA RTX 4070...")
        start = time.time()
        clf_model.fit(X_train, y_train)
        print(f"   -> Training complete in {time.time() - start:.2f} seconds.")
        
        # 1. Extract and display Feature Importances
        importance = clf_model.feature_importances_
        imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        imp_df = imp_df.sort_values(by='Importance', ascending=False)
        print("\n   --- Feature Importances ---")
        print(imp_df.to_string(index=False))
        
        # 2. Calculate Continuous SDM Metrics (ROC-AUC)
        probs = clf_model.predict_proba(X_test)[:, 1] 
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        
        print("\n   --- Model Performance ---")
        print(f"   ROC-AUC Score: {roc_auc:.4f}")
        print(f"   PR-AUC Score:  {pr_auc:.4f}")
        
        # Save the new continuous-only model distinct from the previous versions
        model_path = f"data/processed/{common_name.lower()}_continuous_model.json"
        clf_model.save_model(model_path)
        print(f"   -> Model saved to {model_path}")

if __name__ == "__main__":
    train_continuous_models()