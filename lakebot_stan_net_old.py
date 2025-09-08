import torch
import torch.nn as nn

class BandAdapter(nn.Module):  # Picks best colors for 12 bands
    def __init__(self, in_bands=12, out_bands=4):  # 12 input, reduce to 4 key bands
        super().__init__()
        self.in_bands = in_bands  # Store in_bands as instance variable
        self.linear = nn.Linear(in_bands, out_bands)

    def forward(self, x):
        # Handle 3D input (batch_size, 1, channels) or 4D (batch_size, channels, height, width)
        if x.dim() == 3:  # Shape: (batch_size, 1, channels)
            batch_size, _, channels = x.size()
            if channels != self.in_bands:
                raise ValueError(f"Expected {self.in_bands} channels, got {channels}")
            x = x.squeeze(1)  # Shape: (batch_size, channels)
            x = self.linear(x)  # Shape: (batch_size, out_bands)
            x = x.unsqueeze(2).unsqueeze(3)  # Shape: (batch_size, out_bands, 1, 1)
        else:  # Shape: (batch_size, channels, height, width)
            batch_size, channels, height, width = x.size()
            if channels != self.in_bands:
                raise ValueError(f"Expected {self.in_bands} channels, got {channels}")
            x = x.view(batch_size, channels, -1)  # Shape: (batch_size, 12, height * width)
            x = x.transpose(1, 2)  # Shape: (batch_size, height * width, 12)
            x = self.linear(x)  # Shape: (batch_size, height * width, 4)
            x = x.transpose(1, 2)  # Shape: (batch_size, 4, height * width)
            x = x.view(batch_size, -1, height, width)  # Shape: (batch_size, 4, height, width)
        return x

class SAM(nn.Module):  # Looks at important picture spots
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size=1)  # Works with 4 input bands
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn

class TAU(nn.Module):  # Remembers water changes
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(4, 4, batch_first=True)

    def forward(self, x):
        batch, channels, height, width = x.size()
        if height * width == 1:
            x = x.squeeze(2).squeeze(2)  # Shape: (batch_size, 4)
            x = x.unsqueeze(1)  # Shape: (batch_size, 1, 4) for GRU
            out, _ = self.gru(x)
            out = out.squeeze(1)  # Shape: (batch_size, 4)
            out = out.unsqueeze(2).unsqueeze(3)  # Shape: (batch_size, 4, 1, 1)
        else:
            x = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)  # Flatten
            out, _ = self.gru(x)
            out = out.reshape(batch, height, width, channels).permute(0, 3, 1, 2)  # Back to image
        return out

class DAL(nn.Module):  # Thinks about deep water
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(4, 10)  # 10 water secrets

    def forward(self, x, depth):
        features = x.mean([2, 3])  # Average over height and width (works for 1x1)
        decay = torch.exp(-0.05 * depth)  # Light fades deeper
        features = features * decay.unsqueeze(1)
        return self.dense(features)

class LakeBot_STAN_NET(nn.Module):  # The whole brain
    def __init__(self, in_bands=12):
        super().__init__()
        self.in_bands = in_bands  # Store in_bands for the model
        self.adapter = BandAdapter(in_bands)
        self.sam = SAM()
        self.tau = TAU()
        self.dal = DAL()
        self.output = nn.Linear(10, 10)  # 10 water secrets

    def forward(self, x, depth=0.0):
        x = self.adapter(x)
        x = self.sam(x)
        x = self.tau(x)
        out = self.dal(x, depth.clone().detach())
        return self.output(out)