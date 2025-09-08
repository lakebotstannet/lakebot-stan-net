from planetary_computer import sign_inplace
import pystac_client
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Uncomment if authentication is needed for signed URLs
# sign_inplace()

data_dir = "data/"
gloria_tab = os.path.join(data_dir, "Geppert_2022.tab")  # Tab-delimited file

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

# Create temp file with clean data
temp_tab = skip_header_to_data(gloria_tab)
df = pd.read_csv(temp_tab, sep='\t').head(10)  # First 10 rows for more chances
print("Cleaned data preview:")
print(df.head())

# Clean up temp file if needed
os.remove(temp_tab)

catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=sign_inplace)

# Function to download band with retries
def download_band_with_retry(href, retries=3, timeout=30):
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=1, status_forcelist=[500, 502, 503, 504, 10054])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    for attempt in range(retries):
        try:
            resp = session.get(href, timeout=timeout)
            if resp.status_code == 200:
                return resp.content
            else:
                print(f"Attempt {attempt+1} failed for {href} (status {resp.status_code})")
        except Exception as e:
            print(f"Attempt {attempt+1} error for {href}: {e}")
    return None

for idx, row in df.iterrows():
    # Parse date from 'Date/Time' column (e.g., "2016-08-29T00:00:00" -> "2016-08-29")
    date_str = str(row['Date/Time']).split('T')[0]  # Extract YYYY-MM-DD part
    date = datetime.strptime(date_str, '%Y-%m-%d')
    time_start = (date - timedelta(days=5)).strftime('%Y-%m-%d')  # Wider window: ±5 days
    time_end = (date + timedelta(days=5)).strftime('%Y-%m-%d')
    
    # Use 'Latitude' and 'Longitude' columns, larger bbox for search
    lon, lat = row['Longitude'], row['Latitude']
    search_bbox = [lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1]  # Larger for search
    
    # Search with cloud filter
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=search_bbox,
        datetime=f"{time_start}/{time_end}",
        query={"eo:cloud_cover": {"lt": 50}}  # Filter <50% cloud cover
    )
    items = list(search.items())
    
    if not items:
        # Fallback without cloud filter
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=search_bbox,
            datetime=f"{time_start}/{time_end}"
        )
        items = list(search.items())
    
    if items:
        item = items[0]  # Take first matching
        print(f"Found image for {date_str}: {item.id} (cloud cover: {item.properties.get('eo:cloud_cover', 'N/A')})")
        
        # Define bands and their target resolution (20m for all)
        band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        stacked_data = []
        successful_bands = []
        reference_profile = None  # Will use first successful band as reference
        
        for band in band_names:
            if band in item.assets:
                href = item.assets[band].href
                content = download_band_with_retry(href)
                if content:
                    try:
                        with rasterio.MemoryFile(content) as memfile:
                            with memfile.open() as src:
                                # Define target CRS and transform for 20m res around point (small clip ~1km)
                                dst_crs = src.crs
                                dst_height, dst_width = 50, 50  # ~1km at 20m
                                dst_transform, dst_width, dst_height = calculate_default_transform(
                                    src.crs, dst_crs, src.width, src.height, *src.bounds,
                                    dst_width=dst_width, dst_height=dst_height
                                )
                                
                                # Clip to small window around lat/lon (approximate)
                                data = np.zeros((dst_height, dst_width), dtype=src.dtypes[0])
                                reproject(
                                    source=rasterio.band(src, 1),
                                    destination=data,
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=dst_transform,
                                    dst_crs=dst_crs,
                                    resampling=Resampling.bilinear  # Or nearest for categorical
                                )
                                
                                stacked_data.append(data)
                                successful_bands.append(band)
                                
                                if reference_profile is None:
                                    reference_profile = src.profile.copy()
                                    reference_profile.update(
                                        height=dst_height, width=dst_width, transform=dst_transform,
                                        count=0  # Will update later
                                    )
                                
                    except Exception as e:
                        print(f"Error processing {band}: {e}")
                else:
                    print(f"Failed to download {band} after retries")
        
        if stacked_data and reference_profile:
            stacked = np.stack(stacked_data, axis=0)
            reference_profile.update(count=len(stacked_data), dtype=stacked.dtype)
            sample_id = row['Sample ID']  # Use Sample ID for filename
            filename = f"S2_{sample_id}.tif"
            output_path = os.path.join(data_dir, "images", filename)
            with rasterio.open(output_path, 'w', **reference_profile) as dst:
                dst.write(stacked)
            print(f"Saved: {output_path} ({len(successful_bands)} bands stacked)")
        else:
            print(f"No bands stacked for {sample_id}")
    else:
        print(f"No image found for {date_str} at lat {row['Latitude']}, lon {row['Longitude']}")

print("Done fetching images!")