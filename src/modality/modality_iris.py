import torch
import torch.nn as nn
from .modality_common import VGGBackbone, make_imagenet_transform

class ModalityIris(nn.Module):

    def __init__(self, pretrained_backbone=True, repr_dim=1024):
        super().__init__()
        self.backbone = VGGBackbone(pretrained=pretrained_backbone)
        self.pool3x = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc3 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((2,16)),  
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.bilevel_proj = nn.Sequential(
            nn.LazyLinear(repr_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.transform = make_imagenet_transform((224,224), to_three_channels=True)

    def forward(self, x):
        is_pil = not isinstance(x, torch.Tensor)
        if is_pil:
            x = self.transform(x).unsqueeze(0)

        pool3, conv_final = self.backbone(x)
        shallow = self.pool3x(pool3)
        fc3 = self.fc3(shallow)
        fc6 = self.fc6(conv_final)
        repr_ = self.bilevel_proj(torch.cat([fc3, fc6], dim=1))
        return {"fc3": fc3, "fc6": fc6, "repr": repr_}
