# Decoupling the Direct and Indirect Effects of Local Climate Zones on NO₂ Dynamics across the Beijing–Tianjin–Hebei Region

This repository contains the data and code accompanying the research: **Decoupling the Direct and Indirect Effects of Local Climate Zones on NO₂ Dynamics across the Beijing–Tianjin–Hebei Region**.

## Methodology Overview

The framework integrates:

1. **Multi-Source Atmospheric and Urban Feature Fusion**: A unified 100 m feature space is constructed by integrating **Sentinel-5P TROPOMI NO₂**, **GEOS-CF atmospheric reanalysis**, **Local Climate Zones (LCZs)**, **AlphaEarth Embedding Features (AAEF)**, meteorological variables, and anthropogenic activity indicators. GEOS-CF is incorporated to reconstruct temporally continuous NO₂ fields and alleviate cloud-induced missing observations.

2. **High-Resolution NO₂ Mapping Framework**: An XGBoost-based downscaling framework is developed to reconstruct annual and seasonal NO₂ concentrations at 100 m spatial resolution. The framework integrates atmospheric chemistry constraints, urban morphology descriptors, meteorological interactions, and semantic embedding representations. Model robustness is evaluated using random sample-based validation, spatial cross-validation, temporal cross-validation, and Leave-One-City-Out (LOCO) validation.

3. **Explainable Attribution and Causal Decoupling**: SHAP (SHapley Additive exPlanations) is first employed to identify nonlinear statistical relationships between LCZs and atmospheric pollution. Subsequently, **Double Machine Learning (DML)** and **Causal Mediation Analysis (CMA)** are introduced to disentangle:
   - **Natural Direct Effects (NDE)** caused by intrinsic source–sink and aerodynamic effects of urban morphology.
   - **Natural Indirect Effects (NIE)** mediated through urban heat island (UHI), turbulence enhancement, and boundary-layer modification.

4. **Counterfactual Urban Renewal Simulation**: A counterfactual simulation framework is constructed to evaluate NO₂ responses under hypothetical LCZ conversion scenarios (e.g., Compact High-rise → Open High-rise, Heavy Industry → Urban Vegetation). The framework quantifies seasonal source–sink transitions, spatial heterogeneity of mitigation benefits, and morphology-dependent nonlinear responses.

![Overall workflow of the proposed LCZ–NO₂ causal inference framework.](images/framework.png)

## Data Source

The framework integrates atmospheric observations, urban morphology descriptors, meteorological reanalysis, and auxiliary geographic datasets.

### 1. Atmospheric Observation Data

Used for NO₂ reconstruction and atmospheric constraint.

* **Sentinel-5P TROPOMI NO₂**: Tropospheric NO₂ column density observations for atmospheric pollution reconstruction, sourced from [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2).

* **NASA GEOS-CF**: Simulated atmospheric composition and meteorological fields used for NO₂ gap filling and atmospheric correction, sourced from [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/NASA_GEOS-CF_v1_rpl_tavg1hr).

* **CHAP Dataset**: 1 km ground-level NO₂ dataset used as auxiliary atmospheric reference, available at [Zenodo](https://doi.org/10.5281/zenodo.15218546).

* **Ground Monitoring Stations**: Surface NO₂ observations for calibration and validation, sourced from [China National Environmental Monitoring Center (CNEMC)](https://www.cnemc.cn/).

---

### 2. Urban Morphology and Semantic Features

Used for urban form characterization and semantic representation.

* **Seasonal Local Climate Zones (LCZ)**: 10 m seasonally consistent LCZ maps derived from our previous study, representing urban surface structure and land-cover composition across different seasons. The dataset was generated using the AAEF-HMM coupled framework proposed in:
  *Seasonally Consistent Local Climate Zone Mapping via Annual AlphaEarth Embedding Features and Hidden Markov Modeling*.
  Dataset available at [Zenodo](https://doi.org/10.5281/zenodo.19393486).

  The seasonal LCZ product enables explicit characterization of intra-annual urban morphology dynamics and seasonal source–sink transitions in atmospheric pollution processes.

* **Google Satellite Embedding (V1)**: 10 m resolution. **64-dimensional Annual AlphaEarth Embedding Features (AAEF)** derived from self-supervised learning, serving as stable semantic urban representations, sourced from [GEE Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL).

---

### 3. Meteorological Data

Used to characterize thermodynamic and boundary-layer processes.

* **ERA5-Land**: Provides meteorological variables including 2 m air temperature ($t_{2m}$), 10 m zonal wind ($u_{10}$), 10 m meridional wind ($v_{10}$), surface pressure (sp), and dew-point temperature (td), sourced from [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY).

* **ERA5**: Provides boundary-layer height (BLH) and surface solar radiation (SSR) for atmospheric stability characterization, sourced from [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_HOURLY).

---

### 4. Auxiliary Data

Used to characterize anthropogenic intensity and geographic constraints.

* **SRTM DEM**: 30 m elevation data, sourced from [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003).

* **ALOS DSM**: 30 m surface elevation data used to characterize urban vertical structure, sourced from [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V4_1).

* **Landsat 8 Surface Reflectance**: Multi-spectral bands (B3, B4, B5, B6) used for surface characterization, sourced from [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2).

* **OpenStreetMap (OSM)**: Vector road-network data used to derive road density and traffic intensity indicators, sourced from [OpenStreetMap](https://www.openstreetmap.org/).

* **WorldPop**: 100 m population density dataset, sourced from [WorldPop](https://hub.worldpop.org/geodata/listing?id=135).

* **VIIRS Nighttime Lights (NTL)**: 500 m nighttime light data representing anthropogenic activity intensity, sourced from [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG).

## Project Structure

```text
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── environment.yml
│
├─ codes/
│  ├─ Data_Preprocessing/
│  │  ├─ build_feature_matrix.py
│  │  ├─ geoscf_gap_filling.py
│  │  └─ extract_lcz_aef_features.py
│  │
│  ├─ NO2_Mapping/
│  │  ├─ train_xgboost_no2.py
│  │  ├─ predict_100m_no2.py
│  │  ├─ spatial_cv_validation.py
│  │  └─ loco_validation.py
│  │
│  ├─ Explainable_Attribution/
│  │  ├─ shap_analysis.py
│  │  ├─ shap_dependence_plot.py
│  │  └─ feature_interaction_analysis.py
│  │
│  ├─ Causal_Inference/
│  │  ├─ double_machine_learning.py
│  │  ├─ causal_mediation_analysis.py
│  │  └─ nde_nie_decomposition.py
│  │
│  ├─ Counterfactual_Simulation/
│  │  ├─ lcz_conversion_simulation.py
│  │  └─ source_sink_transition_analysis.py
│  │
│  └─ Visualization/
│     ├─ plot_no2_maps.py
│     ├─ plot_shap_summary.py
│     ├─ plot_causal_effects.py
│     └─ plot_counterfactual_maps.py
│
├─ datas/
│  ├─ sample_station_data.csv
│  ├─ sample_lcz_grid.tif
│  ├─ sample_aef_features.npy
│  └─ demo_no2_prediction.csv
│
├─ figures/
│
├─ results/
│  ├─ model_performance/
│  ├─ shap_outputs/
│  ├─ causal_effects/
│  └─ counterfactual_simulations/
│
└─ images/
   ├─ framework.png
   ├─ result1.png
   ├─ result2.png
   └─ result3.png
