import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from typing import Tuple

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def make_imagenet_transform(target_size: Tuple[int,int]=(224,224), to_three_channels: bool=True):
    transform_list = []
    transform_list.append(T.Resize(target_size))
    if to_three_channels:
        transform_list.append(T.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img))
    transform_list.append(T.ToTensor())
    transform_list.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(transform_list)

class VGGBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg19(pretrained=pretrained)
        self.features = vgg.features  
        self.pool_indices = [i for i, m in enumerate(self.features) if isinstance(m, nn.MaxPool2d)]
        if len(self.pool_indices) < 3:
            raise RuntimeError("Unexpected VGG features layout.")
        self.third_pool_idx = self.pool_indices[2]  

    def forward(self, x):
        pool3_feat = None
        out = x
        for idx, layer in enumerate(self.features):
            out = layer(out)
            if idx == self.third_pool_idx:
                pool3_feat = out  
        return pool3_feat, out
