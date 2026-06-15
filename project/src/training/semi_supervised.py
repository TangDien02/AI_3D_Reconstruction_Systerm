from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.metrics.losses import chamfer_distance
from src.utils.pointcloud_io import save_pointcloud_npy, save_pointcloud_ply


@dataclass(frozen=True)
class PseudoLabelRecord:
    object_id: str
    image_path: str
    pointcloud_path: str
    confidence: float
    uncertainty: float
    accepted: bool


class ReconstructionUncertaintyFilter:
    """Filter pseudo labels by prediction stability across augmented views."""

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold

    def score(self, predictions: list[torch.Tensor]) -> float:
        if len(predictions) < 2:
            return 0.0

        scores = []
        for index in range(len(predictions)):
            for other_index in range(index + 1, len(predictions)):
                pred_a = predictions[index].unsqueeze(0)
                pred_b = predictions[other_index].unsqueeze(0)
                scores.append(float(chamfer_distance(pred_a, pred_b).item()))
        return float(np.mean(scores)) if scores else 0.0

    def accept(self, predictions: list[torch.Tensor]) -> tuple[bool, float]:
        uncertainty = self.score(predictions)
        return uncertainty <= self.threshold, uncertainty


class PseudoLabelWriter:
    """Persist filtered pseudo point clouds with metadata for later retraining."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.pointcloud_dir = self.output_dir / "pointclouds"
        self.pointcloud_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        object_id: str,
        image_path: str | Path,
        points: torch.Tensor,
        confidence: float,
        uncertainty: float,
        accepted: bool,
        export_ply: bool = True,
    ) -> PseudoLabelRecord:
        points_np = points.detach().cpu().numpy().astype(np.float32)
        npy_path = save_pointcloud_npy(points_np, self.pointcloud_dir / f"{object_id}.npy")
        if export_ply:
            save_pointcloud_ply(points_np, self.pointcloud_dir / f"{object_id}.ply")
        return PseudoLabelRecord(
            object_id=object_id,
            image_path=str(image_path),
            pointcloud_path=str(npy_path),
            confidence=float(confidence),
            uncertainty=float(uncertainty),
            accepted=bool(accepted),
        )


class TeacherStudentConsistency:
    """Consistency objective for target crops without ground-truth 3D."""

    def __init__(self, latent_weight: float = 0.0, point_weight: float = 1.0):
        self.latent_weight = latent_weight
        self.point_weight = point_weight

    def loss(
        self,
        student_points: torch.Tensor,
        teacher_points: torch.Tensor,
        student_latent: torch.Tensor | None = None,
        teacher_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total = student_points.new_tensor(0.0)
        if self.point_weight:
            total = total + self.point_weight * chamfer_distance(student_points, teacher_points.detach())
        if self.latent_weight and student_latent is not None and teacher_latent is not None:
            total = total + self.latent_weight * torch.nn.functional.mse_loss(
                student_latent,
                teacher_latent.detach(),
            )
        return total


@torch.no_grad()
def update_ema_teacher(student: torch.nn.Module, teacher: torch.nn.Module, decay: float = 0.999) -> None:
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
