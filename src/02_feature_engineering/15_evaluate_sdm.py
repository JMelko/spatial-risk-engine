import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_sdm_models():
    print("1. Loading Feature Matrix and Re-establishing Test Set...")
    df = pd.read_parquet("data/processed/model_ready_features_v3.parquet")
    
    # Apply the same exact cleaning steps so our test split perfectly matches training
    df.dropna(subset=['ndvi'], inplace=True)
    numerical_cols = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    for col in numerical_cols:
        df = df[df[col] != -9999.0]
        
    df['ecoregion'] = df['ecoregion'].astype('category')
    features = ['elevation_m', 'ecoregion', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    X = df[features]
    
    targets = {
        "Annas_Hummingbird": 'Calypte anna',
        "Harriss_Hawk": 'Parabuteo unicinctus',
        "Mexican_Jay": 'Aphelocoma wollweberi'
    }
    
    for common_name, sci_name in targets.items():
        print(f"\n{'='*50}")
        print(f"Analyzing: {common_name.replace('_', ' ')} ({sci_name})")
        
        # Recreate the exact test set split used during training
        y = (df['scientific_name'] == sci_name).astype(int)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Load the pre-trained model back onto the GPU
        model = xgb.XGBClassifier(enable_categorical=True, tree_method='hist', device='cuda')
        model.load_model(f"data/processed/{common_name.lower()}_model.json")
        
        # 1. Extract and display Feature Importances
        importance = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        imp_df = imp_df.sort_values(by='Importance', ascending=False)
        print("\n--- Feature Importances (What drives the habitat?) ---")
        print(imp_df.to_string(index=False))
        
        # 2. Calculate Continuous SDM Metrics (ROC-AUC)
        # Using predict_proba extracts the continuous suitability index (0.0 to 1.0)
        print("\n--- Model Performance (Continuous Probabilities) ---")
        probs = model.predict_proba(X_test)[:, 1] 
        
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        
        print(f"ROC-AUC Score: {roc_auc:.4f} (Target > 0.80)")
        print(f"PR-AUC Score:  {pr_auc:.4f}")

if __name__ == "__main__":
    evaluate_sdm_models()