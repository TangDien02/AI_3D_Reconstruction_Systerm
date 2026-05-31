from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"


def plot_point_cloud(
    points: np.ndarray | Sequence[Sequence[float]],
    output_path: str | Path,
    title: str = "Point cloud",
    sample_size: int = 2048,
) -> Path:
    points_np = np.asarray(points, dtype=np.float32)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError("points must have shape [N, 3].")
    if points_np.shape[0] == 0:
        raise ValueError("points must not be empty.")

    if points_np.shape[0] > sample_size:
        indices = np.linspace(0, points_np.shape[0] - 1, sample_size).astype(int)
        points_np = points_np[indices]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        points_np[:, 2],
        s=2,
        alpha=0.7,
        color="#2f6f5e",
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    max_range = np.ptp(points_np, axis=0).max()
    center = points_np.mean(axis=0)
    half = max(max_range / 2, 1e-6)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
