import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd

def create_transmission_corridor():
    print("1. Defining Transmission Line Route Coordinates...")
    # Coordinates in Longitude, Latitude (EPSG:4326)
    # Routing from NM across the AZ border, over the San Pedro River, towards Tucson
    route_points = [
        (-107.75, 32.26),  # Point A: Deming, NM (Desert scrub)
        (-108.70, 32.35),  # Point B: Lordsburg, NM
        (-109.22, 32.26),  # Point C: San Simon, AZ (Border crossing)
        (-110.29, 31.96),  # Point D: Benson, AZ (San Pedro River crossing)
        (-111.00, 32.10)   # Point E: Tucson, AZ (Substation terminus)
    ]
    
    # Create the geometric line
    line = LineString(route_points)
    
    # Load into a GeoDataFrame with WGS84 CRS
    gdf = gpd.GeoDataFrame(
        {'project_name': ['Southwest Intertie Project'], 'voltage_kv': [500]}, 
        geometry=[line], 
        crs="EPSG:4326"
    )
    
    print("2. Reprojecting to CONUS Albers (EPSG:5070)...")
    gdf_5070 = gdf.to_crs("EPSG:5070")
    
    print("3. Generating a 100-meter Right-of-Way (ROW) Buffer...")
    # Buffer the line by 100 meters to simulate the physical footprint and analysis area
    gdf_5070['geometry'] = gdf_5070.geometry.buffer(100)
    
    print("4. Exporting Corridor GeoPackage...")
    output_path = "data/processed/simulated_transmission_corridor_5070.gpkg"
    gdf_5070.to_file(output_path, driver="GPKG")
    
    print(f"   -> Success! Simulated ROW saved to {output_path}")

if __name__ == "__main__":
    create_transmission_corridor()