"""Compact capsule expert with a trainable unbounded class-logit head."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    norm_sq = (x * x).sum(dim=dim, keepdim=True)
    return (norm_sq / (1.0 + norm_sq)) * x / torch.sqrt(norm_sq + 1e-8)


class CapsNetExpert(nn.Module):
    """Convolutional primary capsules route to class capsules, then linear logits."""
    def __init__(self, num_classes: int = 4, capsule_dim: int = 8, routing_iters: int = 3):
        super().__init__()
        if num_classes <= 1 or capsule_dim <= 0 or routing_iters <= 0:
            raise ValueError("invalid capsule model dimensions")
        self.num_classes, self.capsule_dim, self.routing_iters = num_classes, capsule_dim, routing_iters
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.primary = nn.Conv2d(64, 16 * capsule_dim, 1)
        self.vote = nn.Linear(capsule_dim, num_classes * capsule_dim, bias=False)
        # A learned affine classification head produces real-valued logits rather than [0,1] lengths.
        self.classifier = nn.Linear(num_classes * capsule_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        u = self.primary(self.features(x)).permute(0, 2, 3, 1).contiguous()
        u = u.view(b, -1, 16, self.capsule_dim).mean(dim=2)
        u = squash(u)
        votes = self.vote(u).view(b, u.shape[1], self.num_classes, self.capsule_dim)
        routing = votes.new_zeros((b, u.shape[1], self.num_classes))
        for i in range(self.routing_iters):
            coupling = routing.softmax(dim=-1)
            out = squash((coupling.unsqueeze(-1) * votes).sum(dim=1))
            if i + 1 < self.routing_iters:
                routing = routing + (votes * out.unsqueeze(1)).sum(dim=-1)
        return self.classifier(out.reshape(b, -1))
