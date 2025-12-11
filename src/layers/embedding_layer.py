import torch.nn as nn

class ShallowEmbedding(nn.Module):  # FC3
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DeepEmbedding(nn.Module):  # FC6
    def __init__(self, in_dim, out_dim=1024):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, out_dim)
        )

    def forward(self, x):
        return self.fc(x)
