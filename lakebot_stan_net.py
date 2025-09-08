import torch
import torch.nn as nn

class BandAdapter(nn.Module):
    def __init__(self, in_bands=18, out_bands=4):  # 13 bands + 5 indices
        super().__init__()
        self.linear = nn.Linear(in_bands, out_bands)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).transpose(1, 2)
        x = self.linear(x).transpose(1, 2).view(batch_size, -1, height, width)
        return x

class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn

class TAU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(4, 4, batch_first=True)

    def forward(self, x):
        batch, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        out, _ = self.gru(x)
        out = out.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return out

class DAL(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 10, kernel_size=1)  # Pixel-wise output

    def forward(self, x, depth):
        decay = torch.exp(-0.05 * depth).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (N,) -> (N,1,1,1) for broadcasting
        x = x * decay
        return self.conv(x)

class LakeBot_STAN_NET(nn.Module):
    def __init__(self, in_bands=18):
        super().__init__()
        self.adapter = BandAdapter(in_bands)
        self.sam = SAM()
        self.tau = TAU()
        self.dal = DAL()

    def forward(self, x, depth=0.0):
        x = self.adapter(x)
        x = self.sam(x)
        x = self.tau(x)
        out = self.dal(x, depth)  # depth is already tensor, no torch.tensor()
        return out  # (B, 10, H, W) - maps for 10 parameters