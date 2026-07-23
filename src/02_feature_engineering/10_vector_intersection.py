import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

def extract_vector_features_optimized(points_path, vector_path, output_path, test_mode=False):
    print("1. Loading exact points from Parquet...")
    df = pd.read_parquet(points_path)
    
    # POINT 1: Test Mode Implementation
    if test_mode:
        print(">>> TEST MODE ACTIVE: Only running first 50,000 rows <<<")
        df = df.head(50000)
    
    print(f"2. Loading and Optimizing Ecoregion polygons from {vector_path}...")
    polygons_gdf = gpd.read_file(vector_path)[['US_L3NAME', 'geometry']]
    
    # POINT 2 (Optimization): Simplify polygon geometries by 50 meters to drastically reduce math overhead
    polygons_gdf['geometry'] = polygons_gdf['geometry'].simplify(50)
    
    print("3. Executing Chunked Spatial Join...")
    # POINT 2 (Optimization): Chunking the data to prevent RAM bottlenecks
    chunk_size = 100000
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    
    results = []
    
    # POINT 4 (Progress Tracker): Wrapping our chunks in tqdm for a visual loading bar
    for chunk in tqdm(chunks, desc="Processing Spatial Join", unit="chunk"):
        # Build geometries only for the current chunk
        chunk_gdf = gpd.GeoDataFrame(
            chunk, 
            geometry=gpd.points_from_xy(chunk.x_meters, chunk.y_meters),
            crs="EPSG:5070"
        )
        
        # Execute the join on the optimized, simplified polygons
        joined = gpd.sjoin(
            chunk_gdf, 
            polygons_gdf, 
            how="left", 
            predicate="intersects"
        )
        
        # Strip spatial data and append to our results list
        results.append(pd.DataFrame(joined.drop(columns=['geometry', 'index_right'])))
    
    print("4. Consolidating chunks and saving to Parquet...")
    final_df = pd.concat(results)
    final_df.rename(columns={'US_L3NAME': 'ecoregion'}, inplace=True)
    
    final_df.to_parquet(output_path, index=False)
    print(f"Success! Vector extraction complete. Saved to {output_path}")

    print("\n--- Sample of Feature Engineered Data ---")
    print(final_df[final_df['ecoregion'].notnull()].head().to_string(index=False))

if __name__ == "__main__":
    POINTS_DIR = "data/processed/exact_points_5070.parquet"
    VECTOR_FILE = "data/raw/reg9_reg3_eco_l3_Merged_Reprojected.shp" 
    OUTPUT_FILE = "data/processed/points_with_ecoregion.parquet"
    
    # Run the optimized function
    # NOTE: Set test_mode=False to run the full dataset once the test succeeds
    extract_vector_features_optimized(POINTS_DIR, VECTOR_FILE, OUTPUT_FILE, test_mode=False)