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


def density_aware_chamfer_distance(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    alpha: float = 1.0,
    gt_weight: float = 1.0,
    pred_weight: float = 1.0,
) -> torch.Tensor:
    """
    Density-Aware Chamfer Distance (DCD).

    Standard Chamfer Distance allows many predicted points to map to the same
    GT point (many-to-one), which encourages point clustering. DCD fixes this
    by down-weighting each predicted point's contribution inversely proportional
    to how many other predicted points share its nearest GT neighbor.

    Symmetrically, on the GT side, each GT point's contribution is down-weighted
    by how many GT points share the same nearest predicted point.

    Args:
        pred_points: [B, N, 3] predicted point cloud
        gt_points: [B, M, 3] ground-truth point cloud
        alpha: Density penalty strength. Higher = stronger anti-clustering.
               alpha=0 reduces to standard Chamfer Distance.
               alpha=1 is the recommended default.
        gt_weight: Weight for the GT-to-pred term (coverage).
        pred_weight: Weight for the pred-to-GT term (precision).

    Reference: "Density-Aware Chamfer Distance as a Comprehensive Metric for
    Point Cloud Completion" (Wu et al., 2021)
    """
    # Pairwise L2 distances: [B, N, M]
    distances = torch.cdist(pred_points, gt_points, p=2)
    distances_sq = distances.pow(2)

    # --- Pred-to-GT direction (precision) ---
    # For each predicted point, find its nearest GT neighbor
    pred_to_gt_dist, pred_nn_indices = distances_sq.min(dim=2)  # [B, N]

    # Count how many predicted points share each GT neighbor
    batch_size, num_pred, num_gt = distances.shape
    # One-hot encode the nearest GT index for each pred point: [B, N, M]
    pred_nn_onehot = torch.zeros(
        batch_size, num_pred, num_gt,
        device=pred_points.device, dtype=pred_points.dtype,
    )
    pred_nn_onehot.scatter_(2, pred_nn_indices.unsqueeze(2), 1.0)
    # Count per GT point how many pred points chose it: [B, M]
    gt_counts = pred_nn_onehot.sum(dim=1).clamp(min=1.0)
    # Look up the count for each pred point's chosen GT: [B, N]
    pred_density = gt_counts.gather(1, pred_nn_indices)

    # Down-weight by density: point in a cluster of 10 gets weight 1/10^alpha
    pred_weights = (1.0 / pred_density).pow(alpha)
    # Normalize weights so they sum to N (preserves loss scale)
    pred_weights = pred_weights * (num_pred / pred_weights.sum(dim=1, keepdim=True).clamp(min=1e-8))
    pred_to_gt_term = (pred_weights * pred_to_gt_dist).mean(dim=1)

    # --- GT-to-Pred direction (coverage) ---
    # For each GT point, find its nearest predicted neighbor
    gt_to_pred_dist, gt_nn_indices = distances_sq.min(dim=1)  # [B, M]

    # Count how many GT points share each pred neighbor
    gt_nn_onehot = torch.zeros(
        batch_size, num_gt, num_pred,
        device=gt_points.device, dtype=gt_points.dtype,
    )
    gt_nn_onehot.scatter_(2, gt_nn_indices.unsqueeze(2), 1.0)
    pred_counts = gt_nn_onehot.sum(dim=1).clamp(min=1.0)  # [B, N]
    gt_density = pred_counts.gather(1, gt_nn_indices)  # [B, M]

    gt_weights = (1.0 / gt_density).pow(alpha)
    num_gt_actual = gt_points.shape[1]
    gt_weights = gt_weights * (num_gt_actual / gt_weights.sum(dim=1, keepdim=True).clamp(min=1e-8))
    gt_to_pred_term = (gt_weights * gt_to_pred_dist).mean(dim=1)

    return (pred_weight * pred_to_gt_term + gt_weight * gt_to_pred_term).mean()



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


def _sample_point_indices(num_points: int, sample_size: int | None, device: torch.device) -> torch.Tensor:
    if sample_size is None or sample_size <= 0 or sample_size >= num_points:
        return torch.arange(num_points, device=device)
    return torch.randperm(num_points, device=device)[:sample_size]


def detail_aware_coverage_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    k: int = 8,
    sample_size: int | None = 512,
    max_weight: float = 3.0,
    exponent: float = 1.0,
) -> torch.Tensor:
    """
    Emphasize coverage for sparse or thin ground-truth regions.

    This is a weighted ground-truth-to-prediction Chamfer term. GT points whose
    local kNN radius is larger get a higher weight, which helps thin structures
    such as chair legs and narrow edges survive the mean loss.
    """
    _, num_gt_points, _ = gt_points.shape
    if num_gt_points < 2:
        return gt_points.new_tensor(0.0)

    sample_indices = _sample_point_indices(num_gt_points, sample_size, gt_points.device)
    sampled_gt = gt_points[:, sample_indices, :]

    gt_to_pred = torch.cdist(sampled_gt, pred_points, p=2).min(dim=2).values.pow(2)

    neighbor_count = min(max(1, int(k)), num_gt_points - 1)
    gt_local_distances = torch.cdist(sampled_gt, gt_points, p=2)
    local_knn = gt_local_distances.topk(neighbor_count + 1, dim=2, largest=False).values[:, :, 1:]
    local_scale = local_knn.mean(dim=2)

    mean_scale = local_scale.mean(dim=1, keepdim=True).clamp_min(1e-8)
    weights = (local_scale / mean_scale).clamp_min(1e-8).pow(max(0.0, float(exponent)))
    max_weight = max(1.0, float(max_weight))
    weights = weights.clamp(min=1.0 / max_weight, max=max_weight)
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-8)
    return (weights * gt_to_pred).mean()


def point_uniformity_loss(
    pred_points: torch.Tensor,
    sample_size: int | None = 512,
) -> torch.Tensor:
    """
    Encourage predicted points to have a more even nearest-neighbor spacing.

    Repulsion only prevents extremely close points. This loss additionally
    discourages uneven clusters by pulling local spacing toward the batch's
    detached mean nearest-neighbor distance.
    """
    _, num_points, _ = pred_points.shape
    if num_points < 2:
        return pred_points.new_tensor(0.0)

    sample_indices = _sample_point_indices(num_points, sample_size, pred_points.device)
    sampled_points = pred_points[:, sample_indices, :]
    distances = torch.cdist(sampled_points, pred_points, p=2)

    self_mask = torch.zeros_like(distances, dtype=torch.bool)
    sample_positions = torch.arange(sample_indices.numel(), device=pred_points.device)
    self_mask[:, sample_positions, sample_indices] = True
    distances = distances.masked_fill(self_mask, float("inf"))

    nearest = distances.min(dim=2).values
    target_spacing = nearest.detach().mean(dim=1, keepdim=True).clamp_min(1e-6)
    normalized_error = (nearest - target_spacing).pow(2) / target_spacing.pow(2)
    return normalized_error.mean()


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
