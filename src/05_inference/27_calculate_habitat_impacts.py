import geopandas as gpd
from rasterstats import zonal_stats
import pandas as pd
import numpy as np

def calculate_impacts():
    print("1. Loading Simulated Transmission Corridor (100m ROW)...")
    corridor = gpd.read_file("data/processed/simulated_transmission_corridor_5070.gpkg")
    
    # Define our threshold for "High Risk" habitat
    RISK_THRESHOLD = 0.70
    
    # Conversion factor: 1 square meter = 0.000247105 acres
    # 1 pixel (30m x 30m) = 900 square meters
    SQM_TO_ACRES = 0.000247105
    PIXEL_AREA_SQM = 900

    results = []

    print("\n2. Analyzing Continuous Raster Impacts (30m Resolution)...")
    rasters = {
        "Harris's Hawk": "data/processed/harriss_hawk_hsi_5070.tif",
        "Mexican Jay": "data/processed/mexican_jay_hsi_5070.tif",
        "Anna's Hummingbird": "data/processed/annas_hummingbird_hsi_5070.tif"
    }

    for species, path in rasters.items():
        print(f"   -> Extracting pixels for {species}...")
        
        # We use raster_out=True to pull the actual numpy array of pixels inside the polygon
        stats = zonal_stats(
            corridor, 
            path, 
            nodata=-9999.0, 
            raster_out=True
        )
        
        # Extract the masked array for the corridor
        pixel_array = stats[0]['mini_raster_array']
        
        # Count how many pixels exceed our 0.70 threshold
        high_risk_pixels = (pixel_array > RISK_THRESHOLD).sum()
        
        # Convert pixel count to acres
        impact_acres = high_risk_pixels * PIXEL_AREA_SQM * SQM_TO_ACRES
        results.append({"Species": species, "High Risk Acres Impacted": round(impact_acres, 2)})

    print("\n3. Analyzing Vector Grid Impacts (Yellow-billed Cuckoo)...")
    print("   -> Running spatial intersection...")
    cuckoo_grid = gpd.read_file("data/processed/yellow_billed_cuckoo_hsi_5070.gpkg")
    
    # Filter the grid down to only the High Risk cells
    high_risk_cuckoo = cuckoo_grid[cuckoo_grid['hsi_probability'] > RISK_THRESHOLD]
    
    # Intersect the high-risk grids with the transmission corridor footprint
    cuckoo_impact = gpd.overlay(corridor, high_risk_cuckoo, how='intersection')
    
    # Calculate the area of the intersected polygons in square meters, then convert to acres
    cuckoo_acres = cuckoo_impact.geometry.area.sum() * SQM_TO_ACRES
    results.append({"Species": "Yellow-billed Cuckoo", "High Risk Acres Impacted": round(cuckoo_acres, 2)})

    print("\n==================================================")
    print("FINAL ENVIRONMENTAL IMPACT REPORT")
    print("Project: Southwest Intertie Project (500kV)")
    print(f"Threshold: HSI > {RISK_THRESHOLD}")
    print("==================================================")
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print("==================================================")

if __name__ == "__main__":
    calculate_impacts()