import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFusion(nn.Module):
    def __init__(self, embed_dim, modality_count):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(modality_count))
        self.embed_dim = embed_dim

    def forward(self, embeddings):
        w = F.softmax(self.weights, dim=0)
        fused = sum(w[i] * embeddings[i] for i in range(len(embeddings)))
        return fused


class BilevelFusion(nn.Module):
    # shallow + deep → final
    def __init__(self, dim_shallow, dim_deep, out_dim=1024):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_shallow + dim_deep, out_dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )

    def forward(self, shallow, deep):
        x = torch.cat([shallow, deep], dim=1)
        return self.fc(x)


class MultiAbstractFusion(nn.Module):
    # multi-level fusion: stack [FC3_face, FC3_iris, FC6_face, FC6_iris, ...]
    def __init__(self, in_dim, out_dim=1024):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        return self.fc(x)
