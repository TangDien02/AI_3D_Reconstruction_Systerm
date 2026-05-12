from __future__ import annotations

import torch


def chamfer_distance(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    """
    Tinh Chamfer Distance giua point cloud du doan va ground truth.

    pred_points: [B, N, 3]
    gt_points: [B, M, 3]
    Gia tri cang nho thi hai point cloud cang giong nhau.
    """
    distances = torch.cdist(pred_points, gt_points, p=2)
    pred_to_gt = distances.min(dim=2).values
    gt_to_pred = distances.min(dim=1).values
    return (pred_to_gt.pow(2).mean(dim=1) + gt_to_pred.pow(2).mean(dim=1)).mean()


def f_score(pred_points: torch.Tensor, gt_points: torch.Tensor, threshold: float = 0.05) -> tuple[float, float, float]:
    """
    Tinh F-score, precision va recall theo nguong khoang cach.

    F-score cang cao thi point cloud du doan vua dung vi tri,
    vua bao phu tot point cloud ground truth.
    """
    distances = torch.cdist(pred_points, gt_points, p=2)
    pred_to_gt = distances.min(dim=2).values
    gt_to_pred = distances.min(dim=1).values

    precision = (pred_to_gt < threshold).float().mean().item()
    recall = (gt_to_pred < threshold).float().mean().item()
    if precision + recall == 0:
        return 0.0, precision, recall
    return float(2 * precision * recall / (precision + recall)), float(precision), float(recall)
