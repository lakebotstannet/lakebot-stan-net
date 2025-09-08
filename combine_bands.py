import rasterio
from rasterio.merge import merge
import glob

# Find all 13 band files (change to your folder)
band_files = glob.glob('C:\Users\Dell\WaterProject\S2_12Bands_50km.tif')  # B01 to B12, B8A

# Open all band files
src_files = [rasterio.open(file) for file in band_files]

# Merge them into one
merged, transform = merge(src_files)

# Save as one file with 13 bands
with rasterio.open('full_13bands.tif', 'w', driver='GTiff', height=merged.shape[1], width=merged.shape[2], count=13, dtype=str(merged.dtype), crs=src_files[0].crs, transform=transform) as dst:
    dst.write(merged)
print("Yay! Your 13-band picture is saved as full_13bands.tif")