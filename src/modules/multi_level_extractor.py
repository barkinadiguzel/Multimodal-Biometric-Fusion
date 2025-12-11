import torch
import torch.nn as nn

class MultiLevelExtractor(nn.Module):
    def __init__(self, dim_fc3=256, dim_fc6=1024):
        super().__init__()

        self.fc3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LazyLinear(dim_fc3),
            nn.ReLU(True)
        )

        self.fc6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LazyLinear(dim_fc6),
            nn.ReLU(True)
        )

    def forward(self, pool3_feat, deep_feat):
        shallow = self.fc3(pool3_feat)
        deep    = self.fc6(deep_feat)
        return shallow, deep
