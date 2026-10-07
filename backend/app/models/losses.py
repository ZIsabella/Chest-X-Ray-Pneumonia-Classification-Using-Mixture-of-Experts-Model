"""Class-balanced focal loss using effective-number class weights."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ClassBalancedFocalLoss(nn.Module):
    """Focal loss with class weights computed from effective sample counts.

    ``class_counts`` is the keyword used by ``backend.training.train``. The
    ``samples_per_class`` argument remains available for backwards compatibility.
    Counts must follow the dataset's ``CLASS_NAMES`` order.
    """

    def __init__(
        self,
        samples_per_class=None,
        beta: float = 0.9999,
        gamma: float = 2.0,
        class_counts=None,
        reduction: str = "mean",
    ):
        super().__init__()
        counts = class_counts if class_counts is not None else samples_per_class
        if counts is None:
            counts = [1, 1, 1, 1]
        counts = torch.as_tensor(counts, dtype=torch.float32)
        if counts.ndim != 1 or counts.numel() != 4 or bool((counts <= 0).any()):
            raise ValueError("class counts must contain four positive training counts")
        if not 0 <= beta < 1 or gamma < 0 or reduction not in ("mean", "sum", "none"):
            raise ValueError("invalid beta, gamma, or reduction")

        # Effective-number weighting, normalized to mean weight 1.
        effective = (1.0 - beta ** counts) / (1.0 - beta) if beta else counts
        weights = 1.0 / effective
        weights = weights / weights.sum() * counts.numel()
        self.register_buffer("class_weights", weights)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        target_indices = targets.long()
        log_pt = log_probs.gather(1, target_indices.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        alpha = self.class_weights.to(logits.device)[target_indices]
        loss = -alpha * (1.0 - pt).pow(self.gamma) * log_pt
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
