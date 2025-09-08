# LakeBot-STAN-Net: Deep Learning for Water Quality Estimation from Sentinel-2 Imagery

## Overview
This project implements a deep learning model (STAN-Net) to estimate 10 water quality parameters (e.g., Chlorophyll-a, TSS, Turbidity) from Sentinel-2 satellite images. The model is designed to work on any water body, producing pixel-wise prediction maps. It incorporates spectral indices as additional input channels for improved accuracy.

## Key Features
- Pixel-wise predictions for spatial maps.
- Integrates 13 Sentinel-2 bands + 5 spectral indices (NDCI, NDTI, NDWI, FAI, CDOM approx).
- Trained on matchup datasets like GLORIA or LIMNADES.
- Supports any location with proper Sentinel-2 input TIFFs.

## Requirements
- Python 3.8+
- Libraries: `torch`, `rasterio`, `numpy`, `pandas`
- Install: `pip install torch rasterio numpy pandas`
- For data fetching: `pip install planetary-computer pystac-client` (optional for STAC API)

## Directory Structure
- `data/images/`: Store Sentinel-2 TIFFs (13-band stacked, Level-2A preferred).
- `data/labels.csv`: CSV with ground truth (columns: image_file, temp, chla, tss, dom, turbidity, depth, secchi, clarity, color, sediments, avg_depth).
- Model weights: `lakebot_stan_net.pth` (generated after training).

## Usage
1. **Prepare Data**:
   - Download Sentinel-2 images (e.g., via Copernicus Hub or STAC API).
   - Create `labels.csv` from datasets like GLORIA (match in-situ to images).
   - Ensure TIFFs are stacked (bands 1-13) and preprocessed (atmospheric correction, cloud masking).

2. **Train the Model**: