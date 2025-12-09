from typing import Dict, List
import torch
import torch.nn as nn
from torchvision import models


def build_squeezenet(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    if pretrained:
        m = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    else:
        m = models.squeezenet1_1(weights=None)
    m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
    m.num_classes = num_classes
    return m


def build_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    if pretrained:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.num_classes = num_classes
    return m


def state_to_list(state: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    return [t.detach().cpu().clone() for _, t in state.items()]


def list_to_state(template: Dict[str, torch.Tensor], arrs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for (k, v), a in zip(template.items(), arrs):
        out[k] = a.to(v.device).type_as(v)
    return out
