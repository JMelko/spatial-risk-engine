import pandas as pd

def verify_features():
    print("Loading model_ready_features_v3.parquet...")
    df = pd.read_parquet("data/processed/model_ready_features_v3.parquet")
    
    print("\n--- Data Overview ---")
    print(f"Total Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n--- Missing Values Check ---")
    # It is normal to have some nulls if points fell outside the state borders
    env_cols = ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
    print(df[env_cols].isnull().sum())
    
    print("\n--- Sample of the New Environmental Variables ---")
    print(df[['scientific_name'] + env_cols].head())
    
    print("\n--- Target Species Occurrence Counts ---")
    targets = ['Calypte anna', 'Parabuteo unicinctus', 'Aphelocoma wollweberi']
    counts = df[df['scientific_name'].isin(targets)]['scientific_name'].value_counts()
    print(counts.to_string())

if __name__ == "__main__":
    verify_features()