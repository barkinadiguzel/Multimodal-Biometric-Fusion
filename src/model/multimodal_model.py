import torch
import torch.nn as nn

from src.modality.modality_face import FaceModality
from src.modality.modality_iris import IrisModality
from src.modality.modality_fingerprint import FingerprintModality

from src.layers.fusion_ops import (
    WeightedFusion,
    BilevelFusion
)


class MultimodalBiometricModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # 3 MODALITY
        self.face = FaceModality()
        self.iris = IrisModality()
        self.fingerprint = FingerprintModality()

        # SHALLOW LEVEL FUSION
        self.shallow_fusion = WeightedFusion(
            in_dims=[256, 256, 256],
            out_dim=256
        )

        # DEEP LEVEL FUSION
        self.deep_fusion = WeightedFusion(
            in_dims=[1024, 1024, 1024],
            out_dim=1024
        )

        # BILEVEL FUSION
        self.bilevel = BilevelFusion(
            dim_shallow=256,
            dim_deep=1024,
            out_dim=1024
        )

        # CLASSIFIER
        self.classifier = nn.Linear(1024, num_classes)


    def forward(self, face_img, iris_img, fp_img):

        # MODALITY EXTRACTION
        face_shallow, face_deep = self.face(face_img)
        iris_shallow, iris_deep = self.iris(iris_img)
        fp_shallow,   fp_deep   = self.fingerprint(fp_img)

        # SHALLOW FUSION
        fused_shallow = self.shallow_fusion([
            face_shallow,
            iris_shallow,
            fp_shallow
        ])

        # DEEP FUSION
        fused_deep = self.deep_fusion([
            face_deep,
            iris_deep,
            fp_deep
        ])

        # BILEVEL FUSION
        final_embedding = self.bilevel(
            fused_shallow,
            fused_deep
        )

        # CLASSIFICATION
        logits = self.classifier(final_embedding)

        return logits
