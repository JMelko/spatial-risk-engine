import os
import glob
import pandas as pd

def validate_processed_sample(processed_dir):
    print("Finding parquet partition files...")
    search_path = os.path.join(processed_dir, "part-*.parquet")
    part_files = glob.glob(search_path)
    
    if not part_files:
        print(f"No partition files found in {processed_dir}")
        return
        
    sample_file = part_files[0]
    print(f"Loading sample file: {os.path.basename(sample_file)}")
    
    df = pd.read_parquet(sample_file)
    
    print("\n--- Data Structure & Types ---")
    print(df.info())
    
    print("\n--- Sample Records ---")
    print(df.head())
    
    print("\n--- Data Sources in this Partition ---")
    print(df['source'].value_counts())
    
    print("\n--- Observation Count Summary ---")
    print(df['observation_count'].describe())

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed/combined_species_data.parquet"
    validate_processed_sample(PROCESSED_DIR)