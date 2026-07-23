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
