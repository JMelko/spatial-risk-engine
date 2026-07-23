import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import time

def spatial_cross_validation():
    print("1. Loading Feature Matrix...")
    df = pd.read_parquet("data/processed/model_ready_features_v3.parquet")
    
    # Standard clean up
    df.dropna(subset=['ndvi'], inplace=True)
    numerical_cols = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    for col in numerical_cols:
        df = df[df[col] != -9999.0]
        
    print("2. Isolating Target (Mexican Jay) and Identifying Spatial Blocks...")
    df['target'] = (df['scientific_name'] == 'Aphelocoma wollweberi').astype(int)
    features = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    
    # We will use the median X coordinate (Longitude) to split the Southwest in half
    # This roughly creates a West Block (Arizona) and an East Block (New Mexico)
    if 'x_meters' not in df.columns:
        print("Error: 'x_meters' column missing. Cannot perform spatial split.")
        return
        
    split_line = df['x_meters'].median()
    
    west_block = df[df['x_meters'] < split_line]
    east_block = df[df['x_meters'] >= split_line]
    
    print(f"   -> West Block: {len(west_block):,} rows")
    print(f"   -> East Block: {len(east_block):,} rows")
    
    # Helper function to train and test
    def train_and_test(train_df, test_df, fold_name):
        X_train = train_df[features]
        y_train = train_df['target']
        X_test = test_df[features]
        y_test = test_df['target']
        
        pos_cases = y_train.sum()
        imbalance_ratio = (len(y_train) - pos_cases) / pos_cases
        
        model = xgb.XGBClassifier(
            tree_method='hist', 
            device='cuda',             
            scale_pos_weight=imbalance_ratio, 
            random_state=42
        )
        
        start = time.time()
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1] 
        roc_auc = roc_auc_score(y_test, probs)
        
        print(f"   -> {fold_name} ROC-AUC Score: {roc_auc:.4f} (Time: {time.time() - start:.2f}s)")
        return roc_auc

    print("\n3. Running Spatial Fold 1: Train on West, Test on East")
    train_and_test(west_block, east_block, "Fold 1 (West->East)")
    
    print("\n4. Running Spatial Fold 2: Train on East, Test on West")
    train_and_test(east_block, west_block, "Fold 2 (East->West)")

if __name__ == "__main__":
    spatial_cross_validation()