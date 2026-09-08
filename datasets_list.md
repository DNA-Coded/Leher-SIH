# Leher-SIH: Datasets and Sources

The Leher project is built on a foundation of rigorous, scientifically validated data. Below is the complete list of datasets used across the visualization platform, the Samudra Data Fabric, and the Rakshak Intelligence ML pipeline.

## 1. Oceanographic & Environmental Data

| Dataset | Source | Purpose / Variables |
|---------|--------|---------------------|
| **GLORYS12V1 (Global Ocean Physics Analysis and Forecast)** | [Copernicus Marine Service (CMEMS)](https://marine.copernicus.eu/) | The primary source of physical ocean parameters. Provides 3D variables: `thetao` (Temp), `so` (Salinity), `uo`/`vo` (Currents). Provides 2D variables: `zos` (Sea Surface Height), `mlotst` (Mixed Layer), `bottomT` (Sea-floor Temp), `siconc` / `sithick` (Sea Ice). |
| **Global Ocean Biogeochemistry Analysis and Forecast** | [Copernicus Marine Service (CMEMS)](https://marine.copernicus.eu/) | Source of biogeochemical data, specifically `chl` (Chlorophyll-a mass concentration) used in the Rakshak Marine Ecosystem Health Engine (HAB risk). |
| **GEBCO (General Bathymetric Chart of the Oceans)** | [GEBCO](https://www.gebco.net/) | Authoritative 15 arc-second resolution global ocean depth dataset. Used for high-resolution 3D bathymetry rendering on the Cesium globe and computing coastal proximity for ML features. |
| **Argo GDAC (Global Data Assembly Centre)** | [Argo](https://argo.ucsd.edu/) | In-situ physical ocean profile data from autonomous profiling floats. Used in the "Model vs. Observation Comparison" feature to validate GLORYS model data against real-world observations. |

## 2. Machine Learning Training Data (Rakshak Intelligence)

| Dataset | Source | Purpose / Variables |
|---------|--------|---------------------|
| **IMD Best Track Dataset** | [India Meteorological Department (IMD)](https://mausam.imd.gov.in/) | Primary historical record of cyclonic disturbances over the North Indian Ocean. Used to label historical GLORYS ocean states for training the XGBoost Cyclone and Storm Surge detection models. |
| **IBTrACS (International Best Track Archive for Climate Stewardship)** | [NOAA NCEI](https://www.ncdc.noaa.gov/ibtracs/) | Supplementary historical cyclone track data (`ibtracs_indian_ocean.csv`) used alongside IMD data to ensure comprehensive ML training coverage. |

## 3. Operational & Maritime Data

| Dataset | Source | Purpose / Variables |
|---------|--------|---------------------|
| **AIS (Automatic Identification System) Vessel Positions** | [INCOIS](https://incois.gov.in/) | Real-time vessel position data. Used by the Fisherman Safe Zone algorithm and Nearby Support system to locate vessels inside or near active hazard polygons. |
| **Tide Gauge Data (Predicted & Observed)** | [INCOIS / IMD](https://incois.gov.in/) | High-frequency coastal tide measurements. Used by Engine 3 (Extreme Tide Detection) to detect anomalies and storm surges near the coast. |

## 4. Scientific Methodologies (Thresholds & Logic)

| Source | Application in Leher |
|--------|----------------------|
| **NOAA Coral Reef Watch** | Methodology and thermal stress thresholds used in the Ecosystem Health Engine to calculate Coral Bleaching Risk (e.g., triggering Watch/Warning/Alert levels as SST climbs past 29°C). |
