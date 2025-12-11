import torch.nn as nn

class MaxPool(nn.Module):
    def __init__(self, k=2, s=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=k, stride=s)

    def forward(self, x):
        return self.pool(x)

class AvgPool(nn.Module):
    def __init__(self, k=2, s=2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=k, stride=s)

    def forward(self, x):
        return self.pool(x)

class AdaptivePool(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(out_size)

    def forward(self, x):
        return self.pool(x)
