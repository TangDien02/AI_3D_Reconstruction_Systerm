from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def ensure_pointcloud_array(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float32)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError("points must have shape [N, 3].")
    if points_np.shape[0] == 0:
        raise ValueError("points must not be empty.")
    return points_np


def save_pointcloud_npy(points: np.ndarray | Sequence[Sequence[float]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, ensure_pointcloud_array(points))
    return output_path


def save_pointcloud_ply(points: np.ndarray | Sequence[Sequence[float]], output_path: str | Path) -> Path:
    points_np = ensure_pointcloud_array(points)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points_np)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    body = [f"{x:.7f} {y:.7f} {z:.7f}" for x, y, z in points_np]
    output_path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return output_path

