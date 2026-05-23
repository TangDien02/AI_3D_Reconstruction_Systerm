from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.pointcloud_io import ensure_pointcloud_array


def _sample_points(points: np.ndarray, max_vertices: int | None) -> np.ndarray:
    if max_vertices is None or len(points) <= max_vertices:
        return points
    indices = np.linspace(0, len(points) - 1, max_vertices).astype(np.int64)
    return points[indices]


def convex_hull_faces(points: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import ConvexHull
    except Exception as exc:
        raise RuntimeError("scipy is required for convex-hull OBJ export") from exc

    hull = ConvexHull(points)
    faces = np.asarray(hull.simplices, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("Convex hull did not produce triangular faces.")
    return faces


def save_pointcloud_obj(
    points: np.ndarray | Sequence[Sequence[float]],
    output_path: str | Path,
    max_vertices: int | None = 2048,
    method: str = "convex_hull",
) -> tuple[Path, dict[str, int | str | bool]]:
    points_np = _sample_points(ensure_pointcloud_array(points), max_vertices)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    faces = np.empty((0, 3), dtype=np.int64)
    used_fallback = False
    if method == "convex_hull" and len(points_np) >= 4:
        try:
            faces = convex_hull_faces(points_np)
        except Exception:
            used_fallback = True
    elif method != "vertices":
        used_fallback = True

    lines = [
        "# Generated from reconstructed point cloud",
        f"# vertices={len(points_np)} faces={len(faces)} method={method}",
    ]
    if used_fallback:
        lines.append("# fallback=vertices_only")

    for x, y, z in points_np:
        lines.append(f"v {x:.7f} {y:.7f} {z:.7f}")

    for a, b, c in faces:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path, {
        "method": method,
        "vertices": int(len(points_np)),
        "faces": int(len(faces)),
        "fallback_vertices_only": used_fallback or len(faces) == 0,
    }
