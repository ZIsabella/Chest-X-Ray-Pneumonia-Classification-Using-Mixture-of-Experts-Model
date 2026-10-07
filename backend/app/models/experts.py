import torch
import torch.nn as nn
from torchvision.models import densenet121, resnet50, DenseNet121_Weights, ResNet50_Weights
from .cbam import CBAM
from .capsnet import PrimaryCapsules, RoutingCapsules


class DenseNetExpert(nn.Module):
    """Expert 1: DenseNet-121 focusing on dense pathological patterns"""

    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base = densenet121(weights=weights)
        self.features = base.features
        in_features = base.classifier.in_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = torch.relu(feat)
        return self.classifier(feat)


class CapsNetExpert(nn.Module):
    """Expert 2: Capsule Network for spatial relationship recognition"""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # مپ فیچر را به اندازه استاندارد 14x14 می‌رساند
        self.pre_caps_pool = nn.AdaptiveAvgPool2d((14, 14))
        # با 8 کپسول و stride=2 در 14x14، ابعاد فیچر 7x7 می‌شود -> مجموعاً 8 * 49 = 392 کپسول ورودی
        self.primary = PrimaryCapsules(in_channels=128, out_caps=8, cap_dim=8, kernel_size=3, stride=2)
        self.routing = RoutingCapsules(num_in_caps=8 * 7 * 7, in_dim=8, num_out_caps=num_classes, out_dim=16,
                                       num_iterations=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv1(x)
        feat = self.pre_caps_pool(feat)
        u = self.primary(feat)
        v = self.routing(u)  # (B, num_classes, 16)
        # طول هر کپسول نشان‌دهنده احتمال حضور بیماری است
        lengths = torch.sqrt(torch.sum(v ** 2, dim=-1) + 1e-8)  # (B, num_classes)
        return lengths


class AttentionResNetExpert(nn.Module):
    """Expert 3: ResNet-50 integrated with CBAM attention"""

    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.cbam = CBAM(in_planes=2048, ratio=16)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
