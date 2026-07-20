# Spatial Habitat Risk Engine

An end-to-end spatial data pipeline and interactive web dashboard that utilizes machine learning to predict highly defensible ecological conflict zones in the American Southwest.

## Architecture & Stack
* **Frontend:** Streamlit, Folium, Leaflet
* **Backend:** Python, XGBoost, NumPy, Rasterio, PyProj
* **Geospatial Processing:** Automated transformation between EPSG:5070 (CONUS Albers) for equal-area matrix math and EPSG:4326 (Web Mercator) for interactive mapping.

## Current Model: Anna's Hummingbird (*Calypte anna*)
This tool predicts the habitat probability of the Anna's Hummingbird using a custom-trained XGBoost classification model. 

### Ecological Data Engineering
The model was trained by extracting localized topographies across millions of data points, utilizing:
1. **Digital Elevation Models (DEM):** USGS 30m resolution matrices.
2. **Terrain Derivatives:** Algorithmically calculated Slope and Aspect contours.
3. **Ecoregion Masking:** Categorical sorting utilizing EPA Level III boundaries (Sonoran Basin and Range).

### Features
* **Dynamic Radial Buffers:** Users click the interactive basemap to drop a pin and define a custom survey radius (in kilometers).
* **Matrix Masking:** The backend utilizes the Pythagorean theorem to mathematically mask the square NumPy array into a true biological survey circle.
* **Impact Analysis:** Automatically calculates affected acreage based on a 30m pixel resolution (0.222394 acres/pixel), categorizing the landscape into High, Moderate, and Low risk.
* **GIS Export:** Generates an on-the-fly, localized GeoTIFF with explicitly defined `NoData` boundaries for seamless integration into desktop GIS software (QGIS/ArcGIS) for NEPA/ESA reporting.

## Usage
1. Clone the repository.
2. Build the environment: `conda env create -f environment.yml`
3. Activate: `conda activate conflict-risk-env`
4. Launch the dashboard: `streamlit run app.py`