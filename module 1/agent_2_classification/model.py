"""
ResNet Architecture for Breast Ultrasound Tumor Classification
Compatible with Agent 3 Grad-CAM Explainability hooks.
"""

import torch
import torch.nn as nn
from torchvision import models


class BreastCancerClassifier(nn.Module):
    """
    Fine-tuned ResNet model for 3-class breast ultrasound classification:
    [0: normal, 1: benign, 2: malignant]
    """

    def __init__(self, architecture: str = "resnet18", num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes

        if architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            in_features = base_model.fc.in_features
        elif architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            in_features = base_model.fc.in_features
        else:
            raise ValueError(f"Unsupported architecture: {architecture}. Choose 'resnet18' or 'resnet50'.")

        # Feature extractor layers (retained for Grad-CAM / Agent 3)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # Custom classification head with dropout for regularization
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map from the final convolutional block (layer4)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        features = self.extract_features(x)
        pooled = self.avgpool(features)
        flattened = torch.flatten(pooled, 1)
        logits = self.fc(flattened)
        return logits
