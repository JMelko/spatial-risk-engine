import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

def interpret_models():
    print("1. Loading Feature Matrix...")
    df = pd.read_parquet("data/processed/model_ready_features_v3.parquet")
    
    # Standard clean up
    df.dropna(subset=['ndvi'], inplace=True)
    numerical_cols = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    for col in numerical_cols:
        df = df[df[col] != -9999.0]
        
    features = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    X = df[features]
    
    # We will use the Mexican Jay to test this, as it has a very distinct habitat signature
    print("\n2. Isolating the Mexican Jay for Interpretation...")
    y = (df['scientific_name'] == 'Aphelocoma wollweberi').astype(int)
    
    # We take a very small, stratified random sample for SHAP and PDPs. 
    # Calculating game-theory permutations across 55 million rows would crash the system.
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=20000, random_state=42, stratify=y)
    
    print("3. Loading Pre-Trained Mexican Jay Model...")
    model = xgb.XGBClassifier(tree_method='hist', device='cuda')
    model.load_model("data/processed/mexican_jay_continuous_model.json")
    
    # --- PART 1: PARTIAL DEPENDENCE PLOTS (PDP) ---
    print("4. Generating Partial Dependence Plots (PDP)...")
    fig, ax = plt.subplots(figsize=(15, 8))
    # Plotting the marginal effect of Elevation, Slope, and Distance to Water
    display = PartialDependenceDisplay.from_estimator(
        model, 
        X_sample, 
        features=['elevation_m', 'slope_deg', 'distance_to_water_m'],
        ax=ax,
        grid_resolution=50 # Smoothness of the curve
    )
    plt.suptitle("Partial Dependence Plots: Mexican Jay Habitat Drivers")
    plt.tight_layout()
    plt.savefig("data/processed/mexican_jay_pdp.png")
    print("   -> Saved PDP visualization to data/processed/mexican_jay_pdp.png")
    plt.close()

   # --- PART 2: SHAP VALUES ---
    print("5. Calculating SHAP Values (Using XGBoost Native C++ Backend)...")
    # Move model to CPU to avoid device conflicts
    model.set_params(device='cpu') 
    
    # Bypass the shap library parser bug by using XGBoost's native SHAP calculator
    dmatrix = xgb.DMatrix(X_sample)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    
    # The last column in contribs is the bias (base score), we only want the feature columns
    shap_values = contribs[:, :-1]
    
    print("6. Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("data/processed/mexican_jay_shap.png")
    print("   -> Saved SHAP visualization to data/processed/mexican_jay_shap.png")
    plt.close()

if __name__ == "__main__":
    interpret_models()