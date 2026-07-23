from osgeo import gdal

def align_ndvi_to_dem_exact():
    print("1. Opening reference DEM to extract exact bounds and dimensions...")
    ref_raster = "data/processed/aligned_stack/aligned_az_nm_dem_5070.tif"
    src_ndvi = "data/processed/aligned_stack/aligned_ndvi_5070.tif"
    out_ndvi = "data/processed/aligned_stack/aligned_ndvi_5070_resampled.tif"
    
    with gdal.Open(ref_raster) as ref:
        geotransform = ref.GetGeoTransform()
        x_size = ref.RasterXSize
        y_size = ref.RasterYSize
        
        # Extract exact bounding coordinates (minx, miny, maxx, maxy)
        minx = geotransform[0]
        maxy = geotransform[3]
        maxx = minx + (x_size * geotransform[1])
        miny = maxy + (y_size * geotransform[5]) # geotransform[5] is negative
        
        target_bounds = [minx, miny, maxx, maxy]
        x_res = geotransform[1]
        y_res = abs(geotransform[5])

    print("2. Warping NDVI to match exact DEM grid bounds and size...")
    gdal.Warp(
        out_ndvi,
        src_ndvi,
        format="GTiff",
        outputBounds=target_bounds,
        xRes=x_res,
        yRes=y_res,
        resampleAlg=gdal.GRA_Bilinear,
        dstNodata=-9999.0,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    )
    
    print(f"Success! Exactly matched NDVI saved to: {out_ndvi}")

if __name__ == "__main__":
    gdal.UseExceptions()
    align_ndvi_to_dem_exact()