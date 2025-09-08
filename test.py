from lakebot_stan_net import LakeBot_STAN_NET
import torch

model = LakeBot_STAN_NET(in_bands=18)
model.load_state_dict(torch.load('lakebot_stan_net.pth'))
model.eval()
test_image = torch.rand(1, 18, 64, 64)  # Fake input with 18 channels, 64x64 to match training
depths = torch.tensor([1.0])  # Tensor for batch of 1
prediction = model(test_image, depths)  # Pass the tensor
print("Water Parameter Maps Shape:", prediction.shape)  # (1,10,64,64)
print("Sample Average Predictions:", prediction.mean(dim=[2,3]).tolist())