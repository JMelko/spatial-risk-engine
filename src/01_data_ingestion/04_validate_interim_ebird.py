import os
import glob
import pandas as pd

def validate_ebird_sample(interim_dir):
    print("Finding parquet partition files...")
    # Find all part files inside the directory
    search_path = os.path.join(interim_dir, "part-*.parquet")
    part_files = glob.glob(search_path)
    
    if not part_files:
        print(f"No partition files found in {interim_dir}")
        return
        
    # Grab just the first partition file to sample the data safely
    sample_file = part_files[0]
    print(f"Loading sample file: {os.path.basename(sample_file)}")
    
    # Read the single partition into Pandas
    df = pd.read_parquet(sample_file)
    
    print("\n--- Data Structure & Types ---")
    print(df.info())
    
    print("\n--- Sample Records ---")
    print(df.head())
    
    print("\n--- Missing Value Check ---")
    print(df.isnull().sum())
    
    print("\n--- Zero-Fill Verification (Observation Counts) ---")
    print(df['observation_count'].value_counts().head(10))

if __name__ == "__main__":
    EBIRD_INTERIM_DIR = "data/interim/ebird_zero_filled.parquet"
    validate_ebird_sample(EBIRD_INTERIM_DIR)