import ee
import geopandas as gpd

def generate_gee_ndvi():
    print("1. Initializing Google Earth Engine...")
    ee.Initialize(project='az-nm-conflict-risk') # Ensure your ID is here

    print("2. Loading Master Boundary...")
    boundary = gpd.read_file("data/processed/master_boundary_5070.gpkg")
    minx, miny, maxx, maxy = boundary.to_crs("EPSG:4326").total_bounds
    region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

    print("3. Building Cloud-Masked Landsat 8/9 Pipeline...")
    def mask_clouds(image):
        qa = image.select('QA_PIXEL')
        cloud_shadow_bit = 1 << 4
        clouds_bit = 1 << 3
        mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0).And(qa.bitwiseAnd(clouds_bit).eq(0))
        return image.updateMask(mask)

    def calculate_ndvi(image):
        optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
        scaled_image = image.addBands(optical_bands, overwrite=True)
        ndvi = scaled_image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        return ndvi

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    
    landsat = l8.merge(l9) \
        .filterBounds(region) \
        .filterDate('2025-07-15', '2025-09-30') \
        .map(mask_clouds) \
        .map(calculate_ndvi)

    print("4. Computing Temporal Median & Setting NoData...")
    median_ndvi = landsat.median().unmask(-9999.0).clip(region)

    print("5. Submitting 30m Export Task to Google Drive...")
    task = ee.batch.Export.image.toDrive(
        image=median_ndvi,
        description='az_nm_monsoon_ndvi_30m_expanded',
        folder='EarthEngine_Exports',
        fileNamePrefix='aligned_ndvi_5070',
        region=region.getInfo()['coordinates'],
        scale=30,
        crs='EPSG:5070',
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task.start()
    print("\nSuccess! Expanded task submitted to Google.")

if __name__ == "__main__":
    generate_gee_ndvi()