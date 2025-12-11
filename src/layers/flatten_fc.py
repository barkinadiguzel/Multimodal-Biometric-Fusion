import torch.nn as nn

class FlattenFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)
