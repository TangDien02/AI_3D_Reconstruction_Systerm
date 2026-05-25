from __future__ import annotations

import torch


def chamfer_distance(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric Chamfer Distance between predicted and ground-truth point clouds.

    pred_points: [B, N, 3]
    gt_points: [B, M, 3]
    """
    distances = torch.cdist(pred_points, gt_points, p=2)
    pred_to_gt = distances.min(dim=2).values
    gt_to_pred = distances.min(dim=1).values
    return (pred_to_gt.pow(2).mean(dim=1) + gt_to_pred.pow(2).mean(dim=1)).mean()


def weighted_chamfer_distance(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    gt_weight: float = 1.0,
    pred_weight: float = 1.0,
) -> torch.Tensor:
    """
    Chamfer Distance with separate weights for coverage and precision terms.

    Higher gt_weight encourages predicted points to cover more of the ground-truth surface.
    """
    distances = torch.cdist(pred_points, gt_points, p=2)
    pred_to_gt = distances.min(dim=2).values.pow(2).mean(dim=1)
    gt_to_pred = distances.min(dim=1).values.pow(2).mean(dim=1)
    return (pred_weight * pred_to_gt + gt_weight * gt_to_pred).mean()


def point_repulsion_loss(
    pred_points: torch.Tensor,
    k: int = 8,
    radius: float = 0.03,
    sample_size: int | None = 512,
) -> torch.Tensor:
    """
    Penalize predicted points that are too close to their nearest neighbors.

    The optional sample_size keeps this regularizer lightweight for 2048-point outputs.
    """
    _, num_points, _ = pred_points.shape
    if num_points < 2 or k <= 0 or radius <= 0:
        return pred_points.new_tensor(0.0)

    if sample_size is None or sample_size <= 0 or sample_size >= num_points:
        sample_indices = torch.arange(num_points, device=pred_points.device)
    else:
        sample_indices = torch.randperm(num_points, device=pred_points.device)[:sample_size]

    sampled_points = pred_points[:, sample_indices, :]
    distances = torch.cdist(sampled_points, pred_points, p=2)

    sample_positions = torch.arange(sample_indices.numel(), device=pred_points.device)
    self_mask = torch.zeros_like(distances, dtype=torch.bool)
    self_mask[:, sample_positions, sample_indices] = True
    distances = distances.masked_fill(self_mask, float("inf"))

    nearest_count = min(k, num_points - 1)
    nearest_distances = distances.topk(nearest_count, dim=2, largest=False).values
    penalty = torch.exp(-nearest_distances.pow(2) / (radius * radius))
    return penalty.mean()


def f_score(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    threshold: float = 0.05,
) -> tuple[float, float, float]:
    """
    Compute F-score, precision, and recall under a distance threshold.
    """
    distances = torch.cdist(pred_points, gt_points, p=2)
    pred_to_gt = distances.min(dim=2).values
    gt_to_pred = distances.min(dim=1).values

    precision = (pred_to_gt < threshold).float().mean().item()
    recall = (gt_to_pred < threshold).float().mean().item()
    if precision + recall == 0:
        return 0.0, float(precision), float(recall)

    score = 2 * precision * recall / (precision + recall)
    return float(score), float(precision), float(recall)
