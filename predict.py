from lakebot_stan_net import LakeBot_STAN_NET
import torch
import rasterio
import numpy as np
import os  # Added missing import

# CONFIG
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD MODEL
model = LakeBot_STAN_NET(in_bands=18)
model.load_state_dict(torch.load("lakebot_stan_net.pth", map_location=device))
model.eval()

# HELPER TO LOAD AND COMPUTE INDICES, PAD TO PATCH_SIZE (from train.py)
def load_sentinel2_image(path):
    if not os.path.exists(path):
        print(f"Image not found: {path}")
        return None, 0, 0
    with rasterio.open(path) as src:
        bands = src.read().astype(np.float32)  # (num_bands, H, W)
        h, w = bands.shape[1], bands.shape[2]
        bands = np.nan_to_num(bands, nan=0.0, posinf=0.0, neginf=0.0) / 10000.0
        transform = src.transform
        crs = src.crs
    patch_size = 64
    # Pad bands to at least patch_size if smaller
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h > 0 or pad_w > 0:
        bands = np.pad(bands, ((0,0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        h, w = bands.shape[1], bands.shape[2]
    # Compute 5 example indices (assume first 12 bands: 0=B1, 3=B4, etc.; adjust if order differs)
    if bands.shape[0] >= 12:
        ndci = (bands[4] - bands[3]) / (bands[4] + bands[3] + 1e-10)  # B5 - B4
        ndti = (bands[3] - bands[2]) / (bands[3] + bands[2] + 1e-10)  # B4 - B3
        ndwi = (bands[2] - bands[7]) / (bands[2] + bands[7] + 1e-10)  # B3 - B8
        fai = bands[7] - (bands[3] + (bands[11] - bands[3]) * (842-665)/(2190-665))
        cdom = bands[1] / (bands[0] + 1e-10)  # B2 / B1
        indices = np.stack([ndci, ndti, ndwi, fai, cdom], axis=0)  # (5, H, W)
        stacked = np.concatenate([bands[:12], indices], axis=0)  # 12 bands + 5 = 17
        if stacked.shape[0] < 18:
            stacked = np.pad(stacked, ((0, 18 - stacked.shape[0]), (0,0), (0,0)), mode='constant', constant_values=0)
    else:
        stacked = np.zeros((18, patch_size, patch_size), dtype=np.float32)
        print(f"Warning: {path} has only {bands.shape[0]} bands, padded with zeros.")
    return stacked, transform, crs, patch_size, patch_size  # Return padded size

# LOAD IMAGE
image_path = r"C:\Users\Dell\lakebot_project\S2_2025_06_07_Aurangabad.tif"  # Full path to your image
stacked, transform, crs, h, w = load_sentinel2_image(image_path)
if stacked is None:
    print("Failed to load image. Check path.")
else:
    images = torch.from_numpy(stacked).unsqueeze(0).to(device)  # (1,18,h,w)

    # DEPTH INPUT (user-provided or estimated)
    depth_value = 1.0  # Replace as needed
    depths = torch.tensor([depth_value], dtype=torch.float32).to(device)

    # PREDICTION
    with torch.no_grad():
        outputs = model(images, depths)  # (1,10,h,w)

    # SAVE PARAMETER MAPS (multi-band TIFF)
    output_path = r"C:\Users\Dell\lakebot_project\water_quality_maps.tif"
    with rasterio.open(output_path, 'w', driver='GTiff', height=h, width=w, count=10, dtype='float32', crs=crs, transform=transform) as dst:
        for i in range(10):
            dst.write(outputs[0, i].cpu().numpy(), i+1)
    print(f"✅ Maps saved to {output_path}")

    # PRINT AVERAGE VALUES
    water_params = [
        "Temperature (°C)",
        "Chlorophyll-a (mg/m³)",
        "Total Suspended Solids (mg/L)",
        "Dissolved Organic Matter (mg/L)",
        "Turbidity (NTU)",
        "Water Depth (m)",
        "Secchi Depth (m)",
        "Water Clarity (index)",
        "Water Color (index)",
        "Suspended Sediments (mg/L)"
    ]
    print("Average Water Parameters:")
    for i, param in enumerate(water_params):
        value = outputs[0, i].mean().item()
        print(f"{param}: {value:.4f}")