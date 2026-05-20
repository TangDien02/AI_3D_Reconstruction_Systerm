from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.metrics.losses import chamfer_distance


@dataclass(frozen=True)
class ReconstructionLossOutput:
    total_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    domain_loss: torch.Tensor | None = None
    pseudo_loss: torch.Tensor | None = None
    consistency_loss: torch.Tensor | None = None


class DomainAdaptationObjective:
    """Combine source reconstruction loss with optional ADA and pseudo/consistency losses."""

    def __init__(
        self,
        domain_weight: float = 0.1,
        pseudo_weight: float = 0.25,
        consistency_weight: float = 0.1,
    ):
        self.domain_weight = domain_weight
        self.pseudo_weight = pseudo_weight
        self.consistency_weight = consistency_weight

    def __call__(
        self,
        source_points_pred: torch.Tensor,
        source_points_gt: torch.Tensor,
        domain_logits: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        pseudo_points_pred: torch.Tensor | None = None,
        pseudo_points_gt: torch.Tensor | None = None,
        consistency_loss: torch.Tensor | None = None,
    ) -> ReconstructionLossOutput:
        reconstruction_loss = chamfer_distance(source_points_pred, source_points_gt)
        total_loss = reconstruction_loss
        domain_loss = None
        pseudo_loss = None

        if domain_logits is not None and domain_labels is not None:
            domain_loss = F.cross_entropy(domain_logits, domain_labels.long())
            total_loss = total_loss + self.domain_weight * domain_loss

        if pseudo_points_pred is not None and pseudo_points_gt is not None:
            pseudo_loss = chamfer_distance(pseudo_points_pred, pseudo_points_gt.detach())
            total_loss = total_loss + self.pseudo_weight * pseudo_loss

        if consistency_loss is not None:
            total_loss = total_loss + self.consistency_weight * consistency_loss

        return ReconstructionLossOutput(
            total_loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            domain_loss=domain_loss,
            pseudo_loss=pseudo_loss,
            consistency_loss=consistency_loss,
        )


def make_domain_labels(source_batch_size: int, target_batch_size: int, device: torch.device) -> torch.Tensor:
    source = torch.zeros(source_batch_size, dtype=torch.long, device=device)
    target = torch.ones(target_batch_size, dtype=torch.long, device=device)
    return torch.cat([source, target], dim=0)
