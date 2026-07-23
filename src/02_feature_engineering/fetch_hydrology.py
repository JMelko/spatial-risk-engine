import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from pynhd import NHD
from osgeo import gdal
import time
import os

def generate_water_distance_raster():
    print("1. Reading Spatial Bounds from Base DEM...")
    base_raster_path = "data/raw/az_nm_dem_5070.tif"
    
    with rasterio.open(base_raster_path) as src:
        meta = src.meta.copy()
        bounds = src.bounds
        transform = src.transform
        shape = src.shape
        crs = src.crs

    print("2. Fetching NHD Flowlines via USGS API...")
    start_time = time.time()
    
    bbox = gpd.GeoSeries(
        gpd.points_from_xy([bounds.left, bounds.right], [bounds.bottom, bounds.top]), 
        crs=crs
    ).to_crs("EPSG:4326").total_bounds
    
    nhd = NHD("flowline_mr")
    flowlines = nhd.bygeom(tuple(bbox))
    flowlines_5070 = flowlines.to_crs(crs)
    print(f"   Successfully extracted {len(flowlines_5070)} water features.")

    print("3. Rasterizing Hydrology (RAM Check: ~1.6GB integer array)...")
    water_mask = rasterize(
        [(geom, 1) for geom in flowlines_5070.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    print("4. Saving Temporary Mask to Disk for Out-of-Core Processing...")
    temp_mask_path = "data/processed/temp_water_mask.tif"
    meta.update(dtype=rasterio.uint8, count=1, nodata=0)
    
    with rasterio.open(temp_mask_path, 'w', **meta) as dst:
        dst.write(water_mask, 1)
        
    # Flush the 1.66 billion pixel array from RAM
    del water_mask

    print("5. Calculating Proximity on Hard Drive via GDAL (Bypassing RAM)...")
    out_path = "data/processed/distance_to_water_5070.tif"
    
    # Open the temporary mask directly from the hard drive
    src_ds = gdal.Open(temp_mask_path)
    src_band = src_ds.GetRasterBand(1)
    
    # Create the output dataset for the float32 distance values
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(out_path, src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.SetNoDataValue(-9999.0)
    
    # Compute distance. DISTUNITS=GEO calculates exact meters based on EPSG:5070
    options = ["VALUES=1", "DISTUNITS=GEO", "NODATA=-9999.0"]
    gdal.ComputeProximity(src_band, dst_band, options)
    
    # Safely close datasets to flush math to disk
    dst_band = None
    dst_ds = None
    src_band = None
    src_ds = None

    print("6. Cleaning up temporary files...")
    if os.path.exists(temp_mask_path):
        os.remove(temp_mask_path)

    end_time = time.time()
    print(f"\nSuccess! Hydrology raster generated in {round(end_time - start_time, 2)} seconds.")
    print(f"Output saved to: {out_path}")

if __name__ == "__main__":
    generate_water_distance_raster()