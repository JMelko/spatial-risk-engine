import rasterio
import numpy as np

def test_ndvi_stats():
    file_path = "data/processed/aligned_ndvi_5070.tif"
    
    with rasterio.open(file_path) as src:
        data = src.read(1)
        
        # Filter out both NaN values and our -9999.0 NoData baseline
        valid_data = data[(~np.isnan(data)) & (data != -9999.0)]
        
        if valid_data.size == 0:
            print("Error: The array contains no valid data.")
            return

        print(f"Minimum NDVI: {np.min(valid_data):.2f}")
        print(f"Maximum NDVI: {np.max(valid_data):.2f}")
        print(f"Mean NDVI:    {np.mean(valid_data):.2f}")

if __name__ == "__main__":
    test_ndvi_stats()