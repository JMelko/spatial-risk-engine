import geopandas as gpd
from osgeo import gdal
import os
import glob
import time

def standardize_raster_stack():
    print("1. Fetching Official US Census Boundaries...")
    start_time = time.time()
    
    # Download the lightweight 20m resolution state boundaries directly from the Census Bureau
    census_url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_20m.zip"
    states = gpd.read_file(census_url)
    
    # Filter for Arizona (State FP 04) and New Mexico (State FP 35)
    az_nm = states[states['STATEFP'].isin(['04', '35'])]
    
    # Reproject to our project CRS and dissolve into a single continuous polygon
    az_nm_5070 = az_nm.to_crs("EPSG:5070").dissolve()
    
    # Save the master cutline to disk for GDAL to use
    cutline_path = "data/processed/master_boundary_5070.gpkg"
    az_nm_5070.to_file(cutline_path, driver="GPKG")
    print("   Master boundary saved.")

    print("\n2. Locating Rasters for Alignment...")
    # Find all rasters in both the raw and processed folders
    raw_rasters = glob.glob("data/raw/*.tif")
    processed_rasters = glob.glob("data/processed/*.tif")
    
    # Exclude temporary or already-aligned rasters if you run this multiple times
    all_rasters = [r for r in raw_rasters + processed_rasters if "aligned_" not in os.path.basename(r)]
    
    # Create an output directory for the pristine stack
    output_dir = "data/processed/aligned_stack"
    os.makedirs(output_dir, exist_ok=True)

    print(f"   Found {len(all_rasters)} rasters to process.")

    print("\n3. Executing Out-of-Core GDAL Warp...")
    # GDAL Warp Options: Crop to the polygon, enforce NoData, and perfectly align pixels
    warp_options = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=cutline_path,
        cropToCutline=True,
        dstNodata=-9999.0,
        multithread=True,
        creationOptions=["COMPRESS=LZW", "TILED=YES"]
    )

    for raster_path in all_rasters:
        filename = os.path.basename(raster_path)
        out_path = os.path.join(output_dir, f"aligned_{filename}")
        
        print(f"   Clipping: {filename}...")
        gdal.Warp(out_path, raster_path, options=warp_options)

    end_time = time.time()
    print(f"\nSuccess! Entire spatial stack aligned in {round(end_time - start_time, 2)} seconds.")
    print(f"Your pristine, ML-ready rasters are located in: {output_dir}")

if __name__ == "__main__":
    # Enable GDAL exceptions to keep the terminal clean
    gdal.UseExceptions()
    standardize_raster_stack()