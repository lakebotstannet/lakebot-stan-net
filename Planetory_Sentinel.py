from planetary_computer import sign_inplace
import pystac_client
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

# Example: Fetch image for a date/location from GLORIA CSV (assume csv has 'date', 'lat', 'lon', 'chl_a', etc.)
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=sign_inplace)
time_range = "2020-01-01/2020-01-02"  # ±1 day
bbox = [lon-0.01, lat-0.01, lon+0.01, lat+0.01]  # Small box
search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=time_range)
items = list(search.items())
if items:
    item = items[0]
    href = item.assets['B04'].href  # Example band; stack all 13
    # Download and stack bands (use rasterio to open and resample)
    # Extract at lat/lon