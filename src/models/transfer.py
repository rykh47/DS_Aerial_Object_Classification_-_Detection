from typing import Literal

import torch.nn as nn
from torchvision import models


ArchName = Literal["resnet50", "mobilenet_v3_small", "efficientnet_b0"]


def _replace_classifier_resnet(model: nn.Module, num_classes: int) -> nn.Module:
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


def _replace_classifier_mobilenet(model: nn.Module, num_classes: int) -> nn.Module:
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return model


def _replace_classifier_efficientnet(model: nn.Module, num_classes: int) -> nn.Module:
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return model


def build_model(arch: ArchName, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        return _replace_classifier_resnet(model, num_classes)
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        return _replace_classifier_mobilenet(model, num_classes)
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        return _replace_classifier_efficientnet(model, num_classes)
    raise ValueError(f"Unsupported arch: {arch}")


