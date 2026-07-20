import os
import glob
import pandas as pd

def validate_inat_sample(interim_dir):
    print("Finding parquet partition files...")
    search_path = os.path.join(interim_dir, "part-*.parquet")
    part_files = glob.glob(search_path)
    
    if not part_files:
        print(f"No partition files found in {interim_dir}")
        return
        
    sample_file = part_files[0]
    print(f"Loading sample file: {os.path.basename(sample_file)}")
    
    df = pd.read_parquet(sample_file)
    
    print("\n--- Data Structure & Types ---")
    print(df.info())
    
    print("\n--- Sample Records ---")
    print(df.head())
    
    print("\n--- Missing Value Check ---")
    print(df.isnull().sum())

if __name__ == "__main__":
    INAT_INTERIM_DIR = "data/interim/inaturalist_filtered.parquet"
    validate_inat_sample(INAT_INTERIM_DIR)