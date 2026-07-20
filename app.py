import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import folium
import copy
from streamlit_folium import st_folium
from pyproj import Transformer
from src.risk_mapper import create_risk_map

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Habitat Risk Engine", layout="wide")

st.title("Anna's Hummingbird Spatial Risk Engine")
st.markdown("Click anywhere on the map to define a project center point, then generate a predictive habitat risk overlay.")

# 2. SESSION STATE
if "target_lat" not in st.session_state:
    st.session_state["target_lat"] = 32.3358
if "target_lon" not in st.session_state:
    st.session_state["target_lon"] = -110.8812
if "radius_km" not in st.session_state:
    st.session_state["radius_km"] = 2.5
if "overlay_data" not in st.session_state:
    st.session_state["overlay_data"] = None
if "impact_stats" not in st.session_state:
    st.session_state["impact_stats"] = None

# 3. SIDEBAR CONTROLS
st.sidebar.header("Project Parameters")
st.sidebar.info("👆 Click on the map to drop a pin, or enter coordinates below.")

input_lat = st.sidebar.number_input("Latitude", value=st.session_state["target_lat"], format="%.4f")
input_lon = st.sidebar.number_input("Longitude", value=st.session_state["target_lon"], format="%.4f")
input_radius = st.sidebar.slider("Buffer Radius (km)", min_value=1.0, max_value=10.0, value=st.session_state["radius_km"], step=0.5)

st.session_state["target_lat"] = input_lat
st.session_state["target_lon"] = input_lon
st.session_state["radius_km"] = input_radius

run_button = st.sidebar.button("Generate Risk Map", type="primary")

# 4. MAP INTERFACE
col1, col2 = st.columns([2, 1])

with col1:
    m = folium.Map(
        location=[st.session_state["target_lat"], st.session_state["target_lon"]], 
        zoom_start=12, 
        tiles="CartoDB positron"
    )
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite View',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.Marker(
        [st.session_state["target_lat"], st.session_state["target_lon"]], 
        popup="Project Center",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # 5. AI EXECUTION & OVERLAY LOGIC
    if run_button:
        filename = "streamlit_temp_risk"
        out_path = f"data/processed/{filename}.tif"
        
        with st.spinner(f'Analyzing {st.session_state["radius_km"]}km buffer...'):
            try:
                stats = create_risk_map(
                    st.session_state["target_lat"], 
                    st.session_state["target_lon"], 
                    st.session_state["radius_km"], 
                    filename
                )
                st.session_state["impact_stats"] = stats
                
                with rasterio.open(out_path) as src:
                    risk_array = src.read(1)
                    bounds = src.bounds
                
                transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
                lon_min, lat_min = transformer.transform(bounds.left, bounds.bottom)
                lon_max, lat_max = transformer.transform(bounds.right, bounds.top)
                
                # Fetch colormap, copy it, and configure NaNs to be fully transparent (alpha=0.0)
                cm = copy.copy(plt.get_cmap('RdYlGn_r'))
                cm.set_bad(alpha=0.0)
                
                # Apply the colormap (this converts our array into a colorized visual image map)
                colored_array = (cm(risk_array) * 255).astype(np.uint8)
                
                st.session_state["overlay_data"] = {
                    "image": colored_array,
                    "bounds": [[lat_min, lon_min], [lat_max, lon_max]]
                }
                
            except Exception as e:
                st.error(f"Execution Error: {e}")

    if st.session_state["overlay_data"] is not None:
        folium.raster_layers.ImageOverlay(
            image=st.session_state["overlay_data"]["image"],
            bounds=st.session_state["overlay_data"]["bounds"],
            opacity=0.6,
            name="Habitat Risk Gradient"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    
    map_data = st_folium(m, height=700, use_container_width=True)
    
    if map_data and map_data.get("last_clicked"):
        new_lat = map_data["last_clicked"]["lat"]
        new_lon = map_data["last_clicked"]["lng"]
        
        if new_lat != st.session_state["target_lat"] or new_lon != st.session_state["target_lon"]:
            st.session_state["target_lat"] = new_lat
            st.session_state["target_lon"] = new_lon
            st.rerun()

# 6. IMPACT STATISTICS & EXPORT
with col2:
    st.markdown("### Impact Analysis")
    
    if st.session_state["impact_stats"] is not None:
        stats = st.session_state["impact_stats"]
        st.metric(label="Total Survey Area", value=f"{stats['total']:,} Acres")
        
        st.markdown(f"**🔴 High Risk (>70%):** {stats['high']:,} acres")
        st.markdown(f"**🟡 Moderate Risk:** {stats['moderate']:,} acres")
        st.markdown(f"**🟢 Low Risk (<30%):** {stats['low']:,} acres")
    else:
        st.info("Run the model to calculate affected acreage.")
    
    st.markdown("---")
    st.markdown("### Export Tools")
    st.markdown("Download the spatial raster for NEPA/ESA desktop reporting.")
    
    try:
        with open("data/processed/streamlit_temp_risk.tif", "rb") as file:
            st.download_button(
                label="📥 Download GeoTIFF",
                data=file,
                file_name="habitat_risk_map.tif",
                mime="image/tiff",
                use_container_width=True
            )
            
    except FileNotFoundError:
        st.warning("GeoTIFF not yet generated.")