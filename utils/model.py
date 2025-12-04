# Utilities for building models and converting parameters
from typing import Dict, List
import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Create a ResNet-18 tailored for CIFAR-size inputs."""
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()

    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    m.num_classes = num_classes
    return m


def state_to_list(state: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Flatten a state_dict to a list of tensors on CPU."""
    return [t.detach().cpu().clone() for _, t in state.items()]


def list_to_state(template: Dict[str, torch.Tensor], arrs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rebuild a state_dict from a list of tensors using a template for keys/dtypes/devices."""
    out: Dict[str, torch.Tensor] = {}
    for (k, v), a in zip(template.items(), arrs):
        out[k] = a.to(v.device).type_as(v)
    return out
