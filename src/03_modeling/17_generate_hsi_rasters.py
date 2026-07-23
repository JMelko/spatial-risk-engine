import rasterio
import numpy as np
import xgboost as xgb
from tqdm import tqdm

def generate_hsi():
    raster_paths = {
        'elevation': "data/processed/aligned_stack/aligned_az_nm_dem_5070.tif",
        'slope': "data/processed/aligned_stack/aligned_slope_5070.tif",
        'aspect': "data/processed/aligned_stack/aligned_aspect_5070.tif",
        'ndvi': "data/processed/aligned_stack/aligned_ndvi_5070_resampled.tif",
        'water': "data/processed/aligned_stack/aligned_distance_to_water_5070.tif"
    }

    targets = [
        "annas_hummingbird",
        "harriss_hawk",
        "mexican_jay"
    ]

    # Grab metadata from the DEM to structure our output rasters
    with rasterio.open(raster_paths['elevation']) as src:
        meta = src.meta.copy()
        # Update metadata to hold float32 probabilities (0.0 to 1.0) and set a standard NoData value
        meta.update(
            dtype=rasterio.float32,
            nodata=-9999.0
        )
        # Extract the internal block geometries to chunk the processing
        windows = [window for ij, window in src.block_windows(1)]

    for target in targets:
        print(f"\n{'='*50}")
        print(f"Generating HSI Raster for: {target.replace('_', ' ').title()}")
        
        # Load the continuous-only model onto the GPU
        model_path = f"data/processed/{target}_continuous_model.json"
        model = xgb.XGBClassifier(tree_method='hist', device='cuda')
        model.load_model(model_path)
        
        out_path = f"data/processed/{target}_hsi_5070.tif"
        
        # Open the 5 input rasters and the 1 output raster simultaneously
        with rasterio.open(raster_paths['elevation']) as src_elev, \
             rasterio.open(raster_paths['slope']) as src_slope, \
             rasterio.open(raster_paths['aspect']) as src_aspect, \
             rasterio.open(raster_paths['ndvi']) as src_ndvi, \
             rasterio.open(raster_paths['water']) as src_water, \
             rasterio.open(out_path, 'w', **meta) as dst:
             
             for window in tqdm(windows, desc="Predicting Habitat Blocks"):
                 
                 # 1. Read the exact spatial block for all 5 layers
                 elev = src_elev.read(1, window=window)
                 slope = src_slope.read(1, window=window)
                 aspect = src_aspect.read(1, window=window)
                 ndvi = src_ndvi.read(1, window=window)
                 water = src_water.read(1, window=window)
                 
                 # 2. Identify valid pixels (ignoring -9999 boundaries and Earth Engine NaNs)
                 valid_mask = (elev != -9999.0) & (slope != -9999.0) & (aspect != -9999.0) & \
                              (ndvi != -9999.0) & (~np.isnan(ndvi)) & (water != -9999.0)
                 
                 # 3. Create a blank output block filled with the NoData value
                 out_probs = np.full(elev.shape, -9999.0, dtype=np.float32)
                 
                 # 4. If the block contains valid land, run the inference
                 if valid_mask.any():
                     # Stack the valid pixels in the exact order the model expects:
                     # ['elevation_m', 'slope_deg', 'aspect_deg', 'ndvi', 'distance_to_water_m']
                     X = np.column_stack((
                         elev[valid_mask],
                         slope[valid_mask],
                         aspect[valid_mask],
                         ndvi[valid_mask],
                         water[valid_mask]
                     ))
                     
                     # Predict the probability of presence (Class 1) and map back to the 2D block
                     probs = model.predict_proba(X)[:, 1]
                     out_probs[valid_mask] = probs
                     
                 # 5. Write the completed probability block directly to the TIFF
                 dst.write(out_probs, 1, window=window)
                 
        print(f"-> Successfully saved HSI raster to {out_path}")

if __name__ == "__main__":
    generate_hsi()