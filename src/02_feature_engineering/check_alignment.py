import rasterio
import os

def check_raster_alignment():
    # Define the core predictor stack
    stack_dir = "data/processed/aligned_stack"
    core_files = [
        "aligned_az_nm_dem_5070.tif",
        "aligned_slope_5070.tif",
        "aligned_aspect_5070.tif",
        "aligned_distance_to_water_5070.tif",
        "aligned_ndvi_5070_resampled.tif"
    ]
    
    print(f"{'Raster Name':<35} | {'CRS':<10} | {'Dimensions (W x H)':<22} | {'Resolution'}")
    print("-" * 95)
    
    for filename in core_files:
        filepath = os.path.join(stack_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"MISSING: {filename}")
            continue
            
        with rasterio.open(filepath) as src:
            name = filename[:33]
            crs = src.crs.to_string() if src.crs else "None"
            dims = f"{src.width} x {src.height}"
            res = f"{src.res[0]} x {src.res[1]}"
            
            print(f"{name:<35} | {crs:<10} | {dims:<22} | {res}")

if __name__ == "__main__":
    check_raster_alignment()