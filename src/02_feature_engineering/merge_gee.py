import os
import glob
from osgeo import gdal

def merge_gee_export():
    print("1. Finding Earth Engine export tiles...")
    tiles = glob.glob("data/processed/aligned_ndvi_5070-*.tif")
    
    if len(tiles) == 0:
        print("Error: Could not find the Earth Engine tiles in data/processed/")
        return

    print(f"   Found {len(tiles)} tiles. Building Virtual Raster (VRT)...")
    vrt_path = "data/processed/temp_gee_merge.vrt"
    gdal.BuildVRT(vrt_path, tiles)

    out_path = "data/processed/aligned_ndvi_5070.tif"
    
    print("2. Stitching into a single, final GeoTIFF...")
    # Translate the VRT into a single compressed TIFF, allowing BigTIFF for massive files
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        noData=-9999.0,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    )
    
    gdal.Translate(out_path, vrt_path, options=translate_options)
    
    print("3. Cleaning up temporary files...")
    os.remove(vrt_path)
    for tile in tiles:
        os.remove(tile)
        
    print(f"\nSuccess! Final single matrix is ready: {out_path}")

if __name__ == "__main__":
    gdal.UseExceptions()
    merge_gee_export()