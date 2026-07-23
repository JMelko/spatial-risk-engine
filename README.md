Southwest Avian Habitat Conflict Engine
Overview
The Southwest Avian Habitat Conflict Engine is an automated spatial machine learning pipeline designed to streamline National Environmental Policy Act (NEPA) compliance and infrastructure routing. By integrating high-resolution topographical and biological raster data with advanced XGBoost classification models, this tool predicts critical avian habitat suitability across Arizona and New Mexico.

Crucially, the pipeline extends beyond ecological modeling by intersecting predictive habitat surfaces with proposed infrastructure footprints (e.g., a 500kV transmission line Right-of-Way), automatically quantifying direct habitat impacts in actionable acreage metrics for Environmental Impact Statements (EIS) and mitigation cost estimates.

Target Species & Ecological Profiles
The engine models four distinct avian profiles to ensure the pipeline accurately captures varying ecological niches and data limitations:

Harris's Hawk (Parabuteo unicinctus): Sonoran Desert specialist (Continuous 30m Raster).

Mexican Jay (Aphelocoma wollweberi): Madrean Sky Island mid-elevation specialist (Continuous 30m Raster).

Anna's Hummingbird (Calypte anna): Widespread habitat generalist (Continuous 30m Raster).

Yellow-billed Cuckoo (Coccyzus americanus): Riparian specialist modeled using a 5km macro-grid to bypass federal coordinate obfuscation for sensitive species.

Technical Architecture
The repository is structured into five modular stages to ensure reproducibility from raw data ingestion to final regulatory reporting.

src/01_data_ingestion/
Handles the programmatic retrieval, filtering, and cleaning of raw spatial data, including U.S. Forest Service datasets, eBird observational records, and Digital Elevation Models (DEMs).

src/02_feature_engineering/
Engineers the predictive spatial feature matrix.

Continuous Models: Processes 30m resolution rasters for elevation, slope, aspect, NDVI (vegetation density), and distance to water.

Macro-Grid Models: Solves the Modifiable Areal Unit Problem (MAUP) for obfuscated data by converting localized coordinates into 5km x 5km analytical grids, applying zonal statistics (e.g., ndvi_max, water_dist_min) to extract core landscape features without diluting riparian corridors.

src/03_modeling/
Trains the predictive models using XGBoost. The pipeline utilizes native CUDA/GPU hardware acceleration (tree_method='hist', device='cuda') to process tens of millions of spatial pixels efficiently, applying scale_pos_weight algorithms to account for severe presence/absence class imbalances inherent to spatial ecology.

src/04_evaluation/
Executes strict regulatory validation protocols to open the machine learning "black box" and prove biological accuracy:

SHAP & Partial Dependence Plots (PDPs): Validates that the model learned true ecological thresholds (e.g., penalizing flat, low-elevation terrain for Mexican Jays) rather than memorizing statistical noise.

Spatial Block Cross-Validation: Defeats spatial autocorrelation by geographically splitting the training and testing sets (East vs. West). Proves the physical rules transfer across state lines.

Null Model Baselining: Tests the environmental models against pure coordinate-memorization models to ensure true predictive value over geographic clustering (the "Zip Code Trap").

src/05_inference/
The applied consulting module. Scripts simulated infrastructure footprints (e.g., Southwest Intertie Project), applies Right-of-Way (ROW) buffers, and utilizes the rasterstats library to calculate the exact acreage of "High Risk" habitat (HSI > 0.70) impacted by the proposed development.


Installation & Environment Setup
This pipeline requires a robust spatial and machine learning stack. An exact environment state is provided in environment.yml.

1. Clone the repository:

Bash
git clone https://github.com/yourusername/az-nm-conflict-risk.git
cd az-nm-conflict-risk
2. Build the Conda Environment:
This will install all necessary dependencies, including geopandas, rasterstats, shap, and CUDA-enabled xgboost.

Bash
conda env create -f environment.yml
conda activate conflict-risk-env
(Note: A local containerized deployment via Docker is also supported for complete OS and dependency isolation).

Usage: Running a Risk Overlay
To run a NEPA impact extraction on a new proposed infrastructure route:

Place your proposed infrastructure shapefile or GPKG into the data/raw/ directory.

Ensure the geometry is reprojected to CONUS Albers (EPSG:5070).

Update the file path in src/05_inference/27_calculate_habitat_impacts.py.

Execute the extraction script:

Bash
python src/05_inference/27_calculate_habitat_impacts.py
Expected Output:
The script will output an executive summary table detailing the impacted acreage for each species, ready for direct inclusion in regulatory documentation.

Plaintext
==================================================
FINAL ENVIRONMENTAL IMPACT REPORT
Project: Southwest Intertie Project (500kV)
Threshold: HSI > 0.7
==================================================
             Species  High Risk Acres Impacted
       Harris's Hawk                   3271.65
         Mexican Jay                    279.55
  Anna's Hummingbird                    381.63
Yellow-billed Cuckoo                   1429.16*
==================================================
*Note: Macro-grid species represent Consultation Zones requiring targeted pedestrian surveys to delineate precise gallery boundaries.
