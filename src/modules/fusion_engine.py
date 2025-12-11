import torch
import torch.nn as nn

class BilevelFusion(nn.Module):
    def __init__(self, in_shallow, in_deep, out_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_shallow + in_deep, out_dim),
            nn.ReLU(True)
        )

    def forward(self, shallow, deep):
        x = torch.cat([shallow, deep], dim=1)
        return self.fc(x)


class FinalFusionEngine(nn.Module):
    def __init__(self, in_dim, num_classes=1000):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

    def forward(self, fused_modalities):
        # fused_modalities = [face_vec, iris_vec, finger_vec]
        x = torch.cat(fused_modalities, dim=1)
        return self.classifier(x)
