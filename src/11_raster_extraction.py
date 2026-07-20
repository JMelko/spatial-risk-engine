import pandas as pd
import rasterio
from tqdm import tqdm

def extract_multiple_rasters(points_path, raster_paths, output_path, test_mode=False):
    print("1. Loading points with ecoregion data from Parquet...")
    df = pd.read_parquet(points_path)
    
    if test_mode:
        print(">>> TEST MODE ACTIVE: Only running first 50,000 rows <<<")
        df = df.head(50000)
        
    # We chunk the dataframe to prevent Python memory bottlenecks
    chunk_size = 100000
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    
    # Initialize lists to hold the extracted values
    all_elevations = []
    all_slopes = []
    all_aspects = []
    
    print("2. Connecting to Raster datasets...")
    # Open all three raster connections simultaneously
    with rasterio.open(raster_paths['elevation']) as src_elev, \
         rasterio.open(raster_paths['slope']) as src_slope, \
         rasterio.open(raster_paths['aspect']) as src_aspect:
        
        print("3. Executing Chunked Multi-Raster Extraction...")
        for chunk in tqdm(chunks, desc="Extracting Terrain Chunks", unit="chunk"):
            
            # Create a list of (X, Y) tuples for this specific chunk
            coords = list(zip(chunk.x_meters, chunk.y_meters))
            
            # Extract values from all three rasters for this chunk
            chunk_elevations = [val[0] for val in src_elev.sample(coords)]
            chunk_slopes = [val[0] for val in src_slope.sample(coords)]
            chunk_aspects = [val[0] for val in src_aspect.sample(coords)]
            
            # Append the chunk results to the master lists
            all_elevations.extend(chunk_elevations)
            all_slopes.extend(chunk_slopes)
            all_aspects.extend(chunk_aspects)
            
    # Add the extracted lists as new columns to the dataframe
    df['elevation_m'] = all_elevations
    df['slope_deg'] = all_slopes
    df['aspect_deg'] = all_aspects
    
    print("4. Saving fully engineered dataset to Parquet...")
    df.to_parquet(output_path, index=False)
    
    print(f"Success! Multi-raster extraction complete. Saved to {output_path}")
    print("\n--- Sample of Enhanced Terrain Data ---")
    print(df[['scientific_name', 'elevation_m', 'slope_deg', 'aspect_deg']].head().to_string(index=False))

if __name__ == "__main__":
    POINTS_DIR = "data/processed/points_with_ecoregion.parquet"
    OUTPUT_FILE = "data/processed/final_engineered_points.parquet"
    
    # Define a dictionary of our raster paths
    RASTER_FILES = {
        'elevation': "data/raw/az_nm_dem_5070.tif",
        'slope': "data/raw/slope_5070.tif",
        'aspect': "data/raw/aspect_5070.tif"
    }
    
    # Running in test mode first to verify the new layers
    extract_multiple_rasters(POINTS_DIR, RASTER_FILES, OUTPUT_FILE, test_mode=False)