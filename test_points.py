import torch
from lakebot_stan_net import LakeBot_STAN_NET

# In test_points.py, replace the band_values part with:
band_values = [
    [610.0, 982.0, 1416.0, 1214.0, 1195.0, 906.0, 878.0, 863.0, 898.0, 890.0, 1014.0, 869.0],  # A
    [580.0, 766.0, 1134.0, 1036.0, 1235.0, 1199.0, 1194.0, 1104.0, 1252.0, 1373.0, 1157.0, 1001.0],  # B
    [490.0, 650.0, 1094.0, 1054.0, 1140.0, 1125.0, 1119.0, 1076.0, 1166.0, 1233.0, 1095.0, 946.0],  # C
    [774.0, 990.0, 1378.0, 1266.0, 1343.0, 1138.0, 1133.0, 1073.0, 1163.0, 1288.0, 1194.0, 1032.0],  # D
    [458.0, 642.0, 988.0, 1050.0, 1169.0, 1136.0, 1129.0, 1023.0, 1172.0, 1276.0, 1102.0, 947.0],  # E
    [691.0, 902.0, 1176.0, 1166.0, 1256.0, 701.0, 718.0, 668.0, 621.0, 838.0, 846.0, 673.0],  # F
    [779.0, 1078.0, 1598.0, 1578.0, 1686.0, 1916.0, 1975.0, 1824.0, 2016.0, 1932.0, 1937.0, 1466.0]  # G
]
# Make the brain
model = LakeBot_STAN_NET(in_bands=12)
model.load_state_dict(torch.load('lakebot_stan_net.pth', weights_only=True))
model.eval()

# Turn the bands into brain food (tensor)
images = torch.tensor(band_values).float().unsqueeze(1)  # Shape: (7, 1, 12)
depths = torch.rand(7)  # Random depths for each point

# Ask the brain for each point
for i, output in enumerate(model(images, depths)):
    print(f"Point {chr(65 + i)} Water Secrets:")
    print(f"1. Temperature (°C): {output[0].item():.2f}")
    print(f"2. Chlorophyll-a (mg/m³): {output[1].item():.2f}")
    print(f"3. Total Suspended Solids (mg/L): {output[2].item():.2f}")
    print(f"4. Dissolved Organic Matter (mg/L): {output[3].item():.2f}")
    print(f"5. Turbidity (NTU): {output[4].item():.2f}")
    print(f"6. pH: {output[5].item():.2f}")
    print(f"7. Water depth (m): {output[6].item():.2f}")
    print(f"8. Secchi depth (m): {output[7].item():.2f}")
    print(f"9. Water colour (spectral index): {output[8].item():.2f}")
    print(f"10. Suspended sediments (mg/L): {output[9].item():.2f}")
    print("----------")