import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import time

def train_cuckoo_model():
    print("1. Loading Grid Feature Matrix...")
    df = pd.read_parquet("data/processed/cuckoo_grid_features.parquet")
    
    # Define our 5 Zonal Statistic predictor variables
    features = ['elevation_mean', 'slope_mean', 'aspect_mean', 'ndvi_max', 'water_dist_min']
    X = df[features]
    y = df['target_presence']

    print(f"\n==================================================")
    print("Training & Analyzing: Yellow-billed Cuckoo (Coccyzus americanus) - Grid Model")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pos_cases = y_train.sum()
    neg_cases = len(y_train) - pos_cases
    imbalance_ratio = neg_cases / pos_cases
    
    print(f"   -> Imbalance Ratio: {imbalance_ratio:.2f} (Positive Training Cells: {pos_cases:,})")
    
    clf_model = xgb.XGBClassifier(
        tree_method='hist', 
        device='cuda',             
        scale_pos_weight=imbalance_ratio, 
        random_state=42
    )
    
    print("   -> Training on GPU...")
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
    
    model_path = "data/processed/yellow_billed_cuckoo_grid_model.json"
    clf_model.save_model(model_path)
    print(f"   -> Model saved to {model_path}")

if __name__ == "__main__":
    train_cuckoo_model()