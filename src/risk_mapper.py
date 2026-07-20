import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import xgboost as xgb
from pyproj import Transformer

def create_risk_map(target_lat, target_lon, radius_km, output_filename):
    print("1. Loading Trained Model...")
    model = xgb.XGBClassifier()
    model.load_model("data/processed/hummingbird_habitat_model.json")
    
    print(f"2. Translating Coordinates ({target_lat}, {target_lon})...")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    center_x, center_y = transformer.transform(target_lon, target_lat)
    
    print(f"3. Generating Spatial Grid ({radius_km}km radius)...")
    pixel_size = 30
    total_meters = radius_km * 2 * 1000 
    width = int(total_meters / pixel_size)
    height = width
    
    xmin = center_x - (total_meters / 2)
    ymax = center_y + (total_meters / 2)
    
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs = xmin + (cols * pixel_size)
    ys = ymax - (rows * pixel_size)
    
    df_map = pd.DataFrame({
        'x': xs.flatten(),
        'y': ys.flatten()
    })
    
    print("4. Extracting Topography from Rasters...")
    raster_paths = {
        'elevation': "data/raw/az_nm_dem_5070.tif",
        'slope': "data/raw/slope_5070.tif",
        'aspect': "data/raw/aspect_5070.tif"
    }
    
    coords = list(zip(df_map.x, df_map.y))
    
    with rasterio.open(raster_paths['elevation']) as src_elev, \
         rasterio.open(raster_paths['slope']) as src_slope, \
         rasterio.open(raster_paths['aspect']) as src_aspect:
             
        df_map['elevation_m'] = [val[0] for val in src_elev.sample(coords)]
        df_map['slope_deg'] = [val[0] for val in src_slope.sample(coords)]
        df_map['aspect_deg'] = [val[0] for val in src_aspect.sample(coords)]
        
    df_map.loc[df_map['aspect_deg'] < 0, 'aspect_deg'] = -1
    
    print("5. Assigning Ecoregion...")
    df_map['ecoregion'] = 'Sonoran Basin and Range'
    df_map['ecoregion'] = df_map['ecoregion'].astype('category')
    
    print("6. AI Prediction: Calculating Habitat Probabilities...")
    X_predict = df_map[['elevation_m', 'ecoregion', 'slope_deg', 'aspect_deg']]
    probabilities = model.predict_proba(X_predict)[:, 1]
    
    print("7. Applying Radial Mask...")
    # Calculate distance of each pixel from the center point using Pythagoras
    distances = np.sqrt((df_map['x'] - center_x)**2 + (df_map['y'] - center_y)**2)
    # Create a boolean mask where True means it falls inside our requested radius
    valid_mask = distances <= (radius_km * 1000)
    
    # Apply the mask: set pixels outside the radius to NaN (Not a Number)
    probabilities = np.where(valid_mask, probabilities, np.nan)
    
    print("8. Rendering GeoTIFF...")
    risk_array = probabilities.reshape(height, width)
    transform = from_origin(xmin, ymax, pixel_size, pixel_size)
    
    out_path = f"data/processed/{output_filename}.tif"
    with rasterio.open(
        out_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=risk_array.dtype,
        crs='EPSG:5070',
        transform=transform,
        nodata=np.nan # Explicitly tell GIS software that NaNs are empty space
    ) as dst:
        dst.write(risk_array, 1)
        
    print(f"\nSuccess! Risk map saved to {out_path}")
    
    # 9. CALCULATE IMPACT ACREAGE
    acres_per_pixel = 0.222394
    # Only count pixels that fall within the circle mask
    total_pixels = np.sum(valid_mask)
    
    # NumPy automatically ignores NaNs when executing > or < comparisons
    high_risk_pixels = np.sum(probabilities >= 0.70)
    low_risk_pixels = np.sum(probabilities < 0.30)
    moderate_risk_pixels = total_pixels - high_risk_pixels - low_risk_pixels
    
    stats = {
        "total": round(total_pixels * acres_per_pixel),
        "high": round(high_risk_pixels * acres_per_pixel),
        "moderate": round(moderate_risk_pixels * acres_per_pixel),
        "low": round(low_risk_pixels * acres_per_pixel)
    }
    
    return stats