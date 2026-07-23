import pandas as pd
import rasterio
from tqdm import tqdm

def extract_multiple_rasters(points_path, raster_paths, output_path, test_mode=False):
    print("1. Loading points with ecoregion data from Parquet...")
    df = pd.read_parquet(points_path)
    
    if test_mode:
        print(">>> TEST MODE ACTIVE: Only running first 50,000 rows <<<")
        df = df.head(50000)
        
    chunk_size = 100000
    chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    
    all_elevations = []
    all_slopes = []
    all_aspects = []
    all_ndvis = []
    all_water_dists = []
    
    print("2. Connecting to Raster datasets in aligned_stack...")
    with rasterio.open(raster_paths['elevation']) as src_elev, \
         rasterio.open(raster_paths['slope']) as src_slope, \
         rasterio.open(raster_paths['aspect']) as src_aspect, \
         rasterio.open(raster_paths['ndvi']) as src_ndvi, \
         rasterio.open(raster_paths['water']) as src_water:
        
        print("3. Executing Chunked Multi-Raster Extraction...")
        for chunk in tqdm(chunks, desc="Extracting Environmental Chunks", unit="chunk"):
            coords = list(zip(chunk.x_meters, chunk.y_meters))
            
            chunk_elevations = [val[0] for val in src_elev.sample(coords)]
            chunk_slopes = [val[0] for val in src_slope.sample(coords)]
            chunk_aspects = [val[0] for val in src_aspect.sample(coords)]
            chunk_ndvis = [val[0] for val in src_ndvi.sample(coords)]
            chunk_water = [val[0] for val in src_water.sample(coords)]
            
            all_elevations.extend(chunk_elevations)
            all_slopes.extend(chunk_slopes)
            all_aspects.extend(chunk_aspects)
            all_ndvis.extend(chunk_ndvis)
            all_water_dists.extend(chunk_water)
            
    df['elevation_m'] = all_elevations
    df['slope_deg'] = all_slopes
    df['aspect_deg'] = all_aspects
    df['ndvi'] = all_ndvis
    df['distance_to_water_m'] = all_water_dists
    
    print("4. Saving fully engineered dataset to Parquet...")
    df.to_parquet(output_path, index=False)
    
    print(f"Success! Multi-raster extraction complete. Saved to {output_path}")

if __name__ == "__main__":
    POINTS_DIR = "data/processed/points_with_ecoregion.parquet"
    OUTPUT_FILE = "data/processed/model_ready_features_v3.parquet"
    
    RASTER_FILES = {
        'elevation': "data/processed/aligned_stack/aligned_az_nm_dem_5070.tif",
        'slope': "data/processed/aligned_stack/aligned_slope_5070.tif",
        'aspect': "data/processed/aligned_stack/aligned_aspect_5070.tif",
        'ndvi': "data/processed/aligned_stack/aligned_ndvi_5070.tif",
        'water': "data/processed/aligned_stack/aligned_distance_to_water_5070.tif"
    }
    
    extract_multiple_rasters(POINTS_DIR, RASTER_FILES, OUTPUT_FILE, test_mode=False)