"""Three-expert RGB chest X-ray mixture-of-experts classifier."""
from __future__ import annotations
import torch
from torch import nn
from .capsnet import CapsNetExpert


class ConvExpert(nn.Module):
    def __init__(self, channels: tuple[int, ...], num_classes: int):
        super().__init__()
        blocks=[]; incoming=3
        for outgoing in channels:
            blocks.extend([nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(outgoing), nn.GELU(),
                           nn.Conv2d(outgoing, outgoing, 3, padding=1, groups=outgoing, bias=False),
                           nn.Conv2d(outgoing, outgoing, 1, bias=False), nn.BatchNorm2d(outgoing), nn.GELU()])
            incoming=outgoing
        self.features=nn.Sequential(*blocks)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.head=nn.Linear(incoming, num_classes)
    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


class CXRMoESystem(nn.Module):
    """Return fused logits, expert gate weights, and expert logits."""
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        if num_classes != 4:
            raise ValueError("This project label contract requires exactly four classes")
        if in_channels != 3:
            raise ValueError("The project dataset transform supplies RGB (3-channel) images")
        self.experts=nn.ModuleList([
            ConvExpert((32, 64, 128), num_classes),
            ConvExpert((24, 48, 96, 160), num_classes),
            CapsNetExpert(num_classes=num_classes),
        ])
        self.gate=nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, len(self.experts)))
    def forward(self, x):
        expert_logits=torch.stack([model(x) for model in self.experts], dim=1)
        pooled=x.mean(dim=(1,2,3), keepdim=False).unsqueeze(1)
        gates=self.gate(pooled).softmax(dim=-1)
        fused=(expert_logits * gates.unsqueeze(-1)).sum(dim=1)
        return fused, gates, {"expert_logits": expert_logits}
