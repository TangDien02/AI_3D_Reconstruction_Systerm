from __future__ import annotations

import torch
import torch.nn.functional as F

PRIMARY_POINTCLOUD_METRICS = [
    "chamfer_distance",
    "f_score",
    "precision",
    "recall",
]

VISUAL_DIAGNOSTIC_METRICS = [
    "surface_alignment_score",
    "detail_preservation_score",
    "structure_occupancy_score",
    "empty_space_score",
    "density_uniformity_score",
    "fine_f_score",
    "fine_precision",
    "fine_recall",
    "loose_f_score",
    "loose_precision",
    "loose_recall",
    "mean_pred_to_gt",
    "mean_gt_to_pred",
    "hausdorff_95",
    "coverage_gap",
    "outlier_ratio",
    "density_cv",
    "density_score",
    "clump_ratio",
    "occupancy_iou",
    "occupancy_precision",
    "occupancy_recall",
    "occupancy_f_score",
    "empty_space_violation",
    "visual_completeness_score",
    "visual_completeness_percent",
    "visual_quality_score",
]

ALL_POINTCLOUD_METRICS = PRIMARY_POINTCLOUD_METRICS + VISUAL_DIAGNOSTIC_METRICS

VISUAL_COMPLETENESS_WEIGHTS = {
    "surface_alignment_score": 0.30,
    "detail_preservation_score": 0.25,
    "structure_occupancy_score": 0.20,
    "empty_space_score": 0.15,
    "density_uniformity_score": 0.10,
}


def _as_batched_points(points: torch.Tensor) -> torch.Tensor:
    if points.ndim == 2:
        return points.unsqueeze(0)
    if points.ndim != 3:
        raise ValueError(f"Expected point cloud shape [B, N, 3] or [N, 3], got {tuple(points.shape)}")
    return points


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _threshold_metrics(
    pred_to_gt: torch.Tensor,
    gt_to_pred: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    precision = (pred_to_gt < threshold).float().mean()
    recall = (gt_to_pred < threshold).float().mean()
    denominator = precision + recall
    if float(denominator.detach().cpu()) == 0.0:
        fscore = precision.new_tensor(0.0)
    else:
        fscore = 2 * precision * recall / denominator
    return fscore, precision, recall


def _nearest_neighbor_density(
    pred_points: torch.Tensor,
    threshold: float,
    sample_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_points, _ = pred_points.shape
    if num_points < 2:
        zero = pred_points.new_tensor(0.0)
        one = pred_points.new_tensor(1.0)
        return zero, one, zero

    sample_count = min(max(2, int(sample_size)), num_points)
    sample_indices = torch.linspace(
        0,
        num_points - 1,
        steps=sample_count,
        device=pred_points.device,
    ).round().long().unique()
    sampled_points = pred_points[:, sample_indices, :]
    distances = torch.cdist(sampled_points, pred_points, p=2)

    self_mask = torch.zeros_like(distances, dtype=torch.bool)
    sample_positions = torch.arange(sample_indices.numel(), device=pred_points.device)
    self_mask[:, sample_positions, sample_indices] = True
    distances = distances.masked_fill(self_mask, float("inf"))

    nearest = distances.min(dim=2).values
    nearest_mean = nearest.mean(dim=1)
    nearest_std = nearest.std(dim=1, unbiased=False)
    density_cv_per_sample = nearest_std / nearest_mean.clamp_min(1e-8)
    density_score_per_sample = 1.0 / (1.0 + density_cv_per_sample)
    clump_ratio = (nearest < threshold * 0.35).float().mean()
    return density_cv_per_sample.mean(), density_score_per_sample.mean(), clump_ratio


def _point_occupancy_grid(
    points: torch.Tensor,
    min_xyz: torch.Tensor,
    extent: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    normalized = (points - min_xyz) / extent.clamp_min(1e-8)
    indices = torch.floor(normalized * (resolution - 1)).long().clamp(0, resolution - 1)
    linear = indices[:, 0] * resolution * resolution + indices[:, 1] * resolution + indices[:, 2]
    grid = torch.zeros(resolution**3, dtype=torch.bool, device=points.device)
    grid[linear] = True
    return grid.view(1, 1, resolution, resolution, resolution)


def _dilate_occupancy(grid: torch.Tensor, dilation: int) -> torch.Tensor:
    if dilation <= 0:
        return grid
    kernel_size = 2 * dilation + 1
    return F.max_pool3d(grid.float(), kernel_size=kernel_size, stride=1, padding=dilation) > 0


def _occupancy_metrics(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    resolution: int,
    dilation: int,
) -> dict[str, torch.Tensor]:
    resolution = max(8, int(resolution))
    dilation = max(0, int(dilation))

    ious = []
    precisions = []
    recalls = []
    fscores = []
    empty_violations = []

    for pred_sample, gt_sample in zip(pred_points, gt_points):
        all_points = torch.cat([pred_sample, gt_sample], dim=0)
        min_xyz = all_points.min(dim=0).values
        max_xyz = all_points.max(dim=0).values
        extent = (max_xyz - min_xyz).max().clamp_min(1e-6)
        padding = extent * 0.02
        min_xyz = min_xyz - padding
        extent = extent + padding * 2

        pred_grid = _point_occupancy_grid(pred_sample, min_xyz, extent, resolution)
        gt_grid = _point_occupancy_grid(gt_sample, min_xyz, extent, resolution)
        pred_flat = pred_grid.flatten()
        gt_flat = gt_grid.flatten()

        intersection = (pred_flat & gt_flat).float().sum()
        union = (pred_flat | gt_flat).float().sum().clamp_min(1.0)
        ious.append(intersection / union)

        pred_dilated = _dilate_occupancy(pred_grid, dilation).flatten()
        gt_dilated = _dilate_occupancy(gt_grid, dilation).flatten()
        pred_count = pred_flat.float().sum().clamp_min(1.0)
        gt_count = gt_flat.float().sum().clamp_min(1.0)

        occupancy_precision = (pred_flat & gt_dilated).float().sum() / pred_count
        occupancy_recall = (gt_flat & pred_dilated).float().sum() / gt_count
        denominator = occupancy_precision + occupancy_recall
        occupancy_fscore = (
            occupancy_precision.new_tensor(0.0)
            if float(denominator.detach().cpu()) == 0.0
            else 2 * occupancy_precision * occupancy_recall / denominator
        )

        precisions.append(occupancy_precision)
        recalls.append(occupancy_recall)
        fscores.append(occupancy_fscore)
        empty_violations.append(1.0 - occupancy_precision)

    return {
        "occupancy_iou": torch.stack(ious).mean(),
        "occupancy_precision": torch.stack(precisions).mean(),
        "occupancy_recall": torch.stack(recalls).mean(),
        "occupancy_f_score": torch.stack(fscores).mean(),
        "empty_space_violation": torch.stack(empty_violations).mean(),
    }


@torch.no_grad()
def compute_pointcloud_quality_metrics(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    threshold: float = 0.05,
    fine_threshold: float | None = None,
    loose_threshold: float | None = None,
    density_sample_size: int = 512,
    voxel_resolution: int = 32,
    occupancy_dilation: int = 1,
) -> dict[str, float]:
    """
    Compute official point-cloud metrics plus visual diagnostics.

    Official metrics stay compatible with the existing pipeline:
    Chamfer Distance, F-score, precision, and recall.

    Additional diagnostics are proxies for visual quality:
    fine F-score/detail recall, density/clump behavior, voxel occupancy overlap,
    and empty-space violations for cases where holes get filled by predictions.
    """
    pred_points = _as_batched_points(pred_points).float()
    gt_points = _as_batched_points(gt_points).float()
    if pred_points.shape[0] != gt_points.shape[0]:
        raise ValueError("pred_points and gt_points must have the same batch size.")

    threshold = float(threshold)
    fine_threshold = float(fine_threshold) if fine_threshold is not None else threshold * 0.5
    loose_threshold = float(loose_threshold) if loose_threshold is not None else threshold * 2.0

    distances = torch.cdist(pred_points, gt_points, p=2)
    pred_to_gt = distances.min(dim=2).values
    gt_to_pred = distances.min(dim=1).values

    chamfer = pred_to_gt.pow(2).mean(dim=1) + gt_to_pred.pow(2).mean(dim=1)
    fscore, precision, recall = _threshold_metrics(pred_to_gt, gt_to_pred, threshold)
    fine_fscore, fine_precision, fine_recall = _threshold_metrics(pred_to_gt, gt_to_pred, fine_threshold)
    loose_fscore, loose_precision, loose_recall = _threshold_metrics(pred_to_gt, gt_to_pred, loose_threshold)

    pred_p95 = torch.quantile(pred_to_gt, 0.95, dim=1)
    gt_p95 = torch.quantile(gt_to_pred, 0.95, dim=1)
    hausdorff_95 = torch.maximum(pred_p95, gt_p95).mean()
    density_cv, density_score, clump_ratio = _nearest_neighbor_density(
        pred_points,
        threshold=threshold,
        sample_size=density_sample_size,
    )
    occupancy = _occupancy_metrics(
        pred_points,
        gt_points,
        resolution=voxel_resolution,
        dilation=occupancy_dilation,
    )

    surface_alignment_score = fscore
    detail_preservation_score = fine_fscore
    structure_occupancy_score = occupancy["occupancy_f_score"]
    empty_space_score = 1.0 - occupancy["empty_space_violation"]
    density_uniformity_score = density_score
    visual_completeness_score = (
        VISUAL_COMPLETENESS_WEIGHTS["surface_alignment_score"] * surface_alignment_score
        + VISUAL_COMPLETENESS_WEIGHTS["detail_preservation_score"] * detail_preservation_score
        + VISUAL_COMPLETENESS_WEIGHTS["structure_occupancy_score"] * structure_occupancy_score
        + VISUAL_COMPLETENESS_WEIGHTS["empty_space_score"] * empty_space_score
        + VISUAL_COMPLETENESS_WEIGHTS["density_uniformity_score"] * density_uniformity_score
    )

    metrics = {
        "chamfer_distance": chamfer.mean(),
        "f_score": fscore,
        "precision": precision,
        "recall": recall,
        "surface_alignment_score": surface_alignment_score,
        "detail_preservation_score": detail_preservation_score,
        "structure_occupancy_score": structure_occupancy_score,
        "empty_space_score": empty_space_score,
        "density_uniformity_score": density_uniformity_score,
        "fine_f_score": fine_fscore,
        "fine_precision": fine_precision,
        "fine_recall": fine_recall,
        "loose_f_score": loose_fscore,
        "loose_precision": loose_precision,
        "loose_recall": loose_recall,
        "mean_pred_to_gt": pred_to_gt.mean(),
        "mean_gt_to_pred": gt_to_pred.mean(),
        "hausdorff_95": hausdorff_95,
        "coverage_gap": 1.0 - recall,
        "outlier_ratio": 1.0 - precision,
        "density_cv": density_cv,
        "density_score": density_score,
        "clump_ratio": clump_ratio,
        "visual_completeness_score": visual_completeness_score,
        "visual_completeness_percent": visual_completeness_score * 100.0,
        "visual_quality_score": visual_completeness_score,
        **occupancy,
    }
    return {name: _scalar(metrics[name]) for name in ALL_POINTCLOUD_METRICS}
