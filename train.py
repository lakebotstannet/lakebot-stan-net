from lakebot_stan_net import LakeBot_STAN_NET
import torch
import torch.nn as nn
import torch.optim as optim
import rasterio
import numpy as np
import pandas as pd
import os

# CONFIG
patch_size = 64  # Smaller to fit clipped images
learning_rate = 0.001
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = "data/"  # Directory with Geppert_2022.tab and 'images/'

# Robust way to read tab file: Skip lines until we find the header with 11 fields starting with 'Event'
def skip_header_to_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data_lines = []
    header_found = False
    for line in lines:
        fields = line.strip().split('\t')
        if len(fields) == 11 and fields[0] == 'Event':
            header_found = True
            continue  # Skip the header line itself
        if header_found and len(fields) == 11:
            data_lines.append(line)
        elif header_found and len(fields) < 11:
            break  # Stop if incomplete line after header
    
    # Write temp data file without header
    temp_file = file_path.replace('.tab', '_data_only.tab')
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write('\t'.join(['Event', 'Sample ID', 'Latitude', 'Longitude', 'Date/Time', 'Samp type', 'Sample comment', 'δ18O H2O [‰ SMOW]', 'δD H2O [‰ SMOW]', 'δ18O H2O std dev [±]', 'δD H2O std dev [±]']) + '\n')  # Add explicit header
        f.writelines(data_lines)
    
    return temp_file

# HELPER TO LOAD AND COMPUTE INDICES, PAD TO PATCH_SIZE
def load_sentinel2_image(path):
    if not os.path.exists(path):
        print(f"Image not found: {path}")
        return None, 0, 0
    with rasterio.open(path) as src:
        bands = src.read().astype(np.float32)  # (num_bands, H, W)
        h, w = bands.shape[1], bands.shape[2]
        bands = np.nan_to_num(bands, nan=0.0, posinf=0.0, neginf=0.0) / 10000.0
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
    return stacked, patch_size, patch_size  # Now always patch_size

# LOAD DATA FROM TAB
gloria_tab = os.path.join(data_dir, "Geppert_2022.tab")
temp_tab = skip_header_to_data(gloria_tab)
labels_df = pd.read_csv(temp_tab, sep='\t').head(10)  # First 10 for training
os.remove(temp_tab)  # Cleanup

images = []
labels = []
depths_list = []
for _, row in labels_df.iterrows():
    sample_id = row['Sample ID']
    img_path = os.path.join(data_dir, 'images', f"S2_{sample_id}.tif")
    stacked, h, w = load_sentinel2_image(img_path)
    if stacked is None or h == 0:
        continue  # Skip if no image
    images.append(torch.from_numpy(stacked).unsqueeze(0).to(device))  # (1,18,p,p)
    
    # Generate dummy labels (use δ18O proxy for temp, others dummy)
    delta18o = row['δ18O H2O [‰ SMOW]'] if pd.notna(row['δ18O H2O [‰ SMOW]']) else 0.0
    temp_proxy = 20.0 + delta18o * 2  # Dummy proxy
    label_values = np.array([temp_proxy, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0, 0.5, 0.0, 0.0], dtype=np.float32)
    label_map = torch.tensor(label_values).unsqueeze(1).unsqueeze(1).repeat(1, patch_size, patch_size).unsqueeze(0).to(device)  # (1,10,p,p)
    labels.append(label_map)
    depths_list.append(5.0)  # Dummy avg_depth

if len(images) == 0:
    print("No images found! Check data/images/ folder.")
else:
    images = torch.cat(images)  # (N,18,p,p)
    labels = torch.cat(labels)  # (N,10,p,p)
    depths = torch.tensor(depths_list, dtype=torch.float32).to(device)  # (N,)

    print(f"Training with {len(images)} images, shape {images.shape}")

    # TRAINING SETUP
    model = LakeBot_STAN_NET(in_bands=18).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # TRAIN LOOP
    for epoch in range(epochs):
        outputs = model(images, depths)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss = {loss.item():.6f}")

    # SAVE MODEL
    torch.save(model.state_dict(), "lakebot_stan_net.pth")
    print("✅ Model saved as lakebot_stan_net.pth")