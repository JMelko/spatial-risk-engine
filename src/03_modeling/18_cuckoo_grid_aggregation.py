import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import box

def create_aggregation_grid():
    print("1. Reading raster bounds for the Southwest...")
    raster_path = "data/processed/aligned_stack/aligned_az_nm_dem_5070.tif"
    
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    print("2. Generating 5km x 5km spatial fishnet grid...")
    grid_size = 5000  # 5,000 meters = 5km
    
    xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top
    
    cols = list(np.arange(xmin, xmax, grid_size))
    rows = list(np.arange(ymin, ymax, grid_size))
    
    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(box(x, y, x + grid_size, y + grid_size))
            
    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
    grid['grid_id'] = grid.index
    print(f"   Created {len(grid):,} distinct grid cells.")
    
    return grid

def aggregate_cuckoo_points(grid):
    print("3. Loading and filtering Yellow-billed Cuckoo records...")
    points_df = pd.read_parquet("data/processed/exact_points_5070.parquet")
    
    cuckoo_df = points_df[points_df['scientific_name'] == 'Coccyzus americanus'].copy()
    cuckoo_gdf = gpd.GeoDataFrame(
        cuckoo_df,
        geometry=gpd.points_from_xy(cuckoo_df.x_meters, cuckoo_df.y_meters),
        crs="EPSG:5070"
    )
    
    print(f"   Found {len(cuckoo_gdf):,} obfuscated occurrence points.")
    
    print("4. Executing spatial join against the grid...")
    joined = gpd.sjoin(grid, cuckoo_gdf, how='left', predicate='intersects')
    
    # If 'index_right' is not null, a bird was found in that cell
    joined['has_bird'] = joined['index_right'].notnull()
    
    # Group by the grid cell ID and check if any birds were present
    presence = joined.groupby('grid_id')['has_bird'].max().astype(int)
    grid['target_presence'] = grid['grid_id'].map(presence)
    
    pos_cells = grid['target_presence'].sum()
    print(f"   Presence assigned to {pos_cells:,} out of {len(grid):,} landscape cells.")
    
    return grid

def extract_zonal_statistics(grid):
    print("5. Extracting Zonal Statistics from Rasters (This may take a few minutes)...")
    
    # Define the raster paths and the specific summary statistic we want to extract
    raster_configs = {
        'elevation_mean': ("data/processed/aligned_stack/aligned_az_nm_dem_5070.tif", 'mean'),
        'slope_mean': ("data/processed/aligned_stack/aligned_slope_5070.tif", 'mean'),
        'aspect_mean': ("data/processed/aligned_stack/aligned_aspect_5070.tif", 'mean'),
        'ndvi_max': ("data/processed/aligned_stack/aligned_ndvi_5070.tif", 'max'),
        'water_dist_min': ("data/processed/aligned_stack/aligned_distance_to_water_5070.tif", 'min')
    }
    
    for col_name, (path, stat) in raster_configs.items():
        print(f"   Processing {col_name}...")
        stats = zonal_stats(
            grid, 
            path, 
            stats=stat, 
            nodata=-9999.0, 
            geojson_out=False
        )
        grid[col_name] = [s[stat] for s in stats]
        
    print("6. Cleaning and Exporting Feature Matrix...")
    # Drop grid cells that fell entirely outside our state boundaries (NaN values)
    grid = grid.dropna(subset=['elevation_mean', 'ndvi_max', 'water_dist_min'])
    
    # Save the tabular feature matrix for XGBoost training
    tabular_output = "data/processed/cuckoo_grid_features.parquet"
    pd.DataFrame(grid.drop(columns=['geometry'])).to_parquet(tabular_output, index=False)
    
    # Save the vector geometries so we can map the results later in QGIS
    spatial_output = "data/processed/cuckoo_grid_geometries.gpkg"
    grid[['grid_id', 'target_presence', 'geometry']].to_file(spatial_output, driver="GPKG")
    
    print(f"   -> Saved tabular model features to {tabular_output}")
    print(f"   -> Saved grid geometries to {spatial_output}")

if __name__ == "__main__":
    grid_gdf = create_aggregation_grid()
    grid_gdf = aggregate_cuckoo_points(grid_gdf)
    extract_zonal_statistics(grid_gdf)