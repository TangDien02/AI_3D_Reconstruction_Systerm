"""
mesh_3d_metrics.py
------------------
3D mesh comparison metrics for evaluation.

Metrics:
  - Chamfer Distance: bidirectional point-to-surface distance
  - Hausdorff Distance: max nearest-neighbor distance
  - F-score: precision/recall at distance threshold

Usage:
    from src.metrics.mesh_3d_metrics import compute_3d_metrics

    result = compute_3d_metrics(
        pred_mesh="results/triposr/mesh.obj",
        gt_mesh="data/pix3d/cad.obj",
        num_samples=50000,
        alignment="icp"
    )
    print(result.chamfer_distance)
    print(result.f_score_tau01)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class Mesh3DMetricsResult:
    """Result of 3D mesh comparison."""
    chamfer_distance: float  # bidirectional
    hausdorff_distance: float
    f_score_tau01: float  # tau=0.01
    f_score_tau001: float  # tau=0.001

    # Per-direction distances (for analysis)
    chamfer_forward: float  # pred → gt
    chamfer_backward: float  # gt → pred

    # Alignment info
    alignment_transform: np.ndarray | None = None  # 4x4 transformation matrix
    alignment_fitness: float = 0.0  # ICP fitness
    scale_ratio: float = 1.0

    def to_dict(self) -> dict:
        return {
            "chamfer_distance": self.chamfer_distance,
            "hausdorff_distance": self.hausdorff_distance,
            "f_score_@0.01": self.f_score_tau01,
            "f_score_@0.001": self.f_score_tau001,
            "chamfer_forward": self.chamfer_forward,
            "chamfer_backward": self.chamfer_backward,
            "alignment_fitness": self.alignment_fitness,
            "scale_ratio": self.scale_ratio,
        }


# ---------------------------------------------------------------------------
# Distance computations
# ---------------------------------------------------------------------------

def _nearest_neighbor_distance(
    src: np.ndarray,
    tgt: np.ndarray,
) -> np.ndarray:
    """
    For each point in src, find nearest neighbor distance to tgt.

    Args:
        src: [N, 3] source points
        tgt: [M, 3] target points

    Returns:
        [N] distances
    """
    from scipy.spatial import cKDTree

    if len(tgt) == 0:
        return np.full(len(src), np.inf)

    tree = cKDTree(tgt)
    distances, _ = tree.query(src, k=1)
    return distances


def chamfer_distance(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Bidirectional Chamfer distance.

    Args:
        pred_pts: [N, 3] predicted points
        gt_pts: [M, 3] ground truth points

    Returns:
        (chamfer, forward, backward)
        - chamfer: bidirectional average
        - forward: mean(dist(pred → gt))
        - backward: mean(dist(gt → pred))
    """
    forward = _nearest_neighbor_distance(pred_pts, gt_pts)
    backward = _nearest_neighbor_distance(gt_pts, pred_pts)

    forward_mean = float(forward.mean())
    backward_mean = float(backward.mean())
    chamfer = (forward_mean + backward_mean) / 2

    return chamfer, forward_mean, backward_mean


def hausdorff_distance(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
) -> float:
    """
    Hausdorff distance: max(max(dist(pred → gt)), max(dist(gt → pred))).

    Args:
        pred_pts: [N, 3] predicted points
        gt_pts: [M, 3] ground truth points

    Returns:
        Hausdorff distance
    """
    forward = _nearest_neighbor_distance(pred_pts, gt_pts)
    backward = _nearest_neighbor_distance(gt_pts, pred_pts)

    return float(max(forward.max(), backward.max()))


def f_score(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    tau: float = 0.01,
) -> float:
    """
    F-score at distance threshold tau.

    F-score = 2 * (precision * recall) / (precision + recall)

    precision = fraction of pred points within tau of GT
    recall = fraction of GT points within tau of pred

    Args:
        pred_pts: [N, 3] predicted points
        gt_pts: [M, 3] ground truth points
        tau: distance threshold

    Returns:
        F-score in [0, 1]
    """
    forward = _nearest_neighbor_distance(pred_pts, gt_pts)
    backward = _nearest_neighbor_distance(gt_pts, pred_pts)

    precision = (forward < tau).sum() / len(pred_pts) if len(pred_pts) > 0 else 0.0
    recall = (backward < tau).sum() / len(gt_pts) if len(gt_pts) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f = 2 * precision * recall / (precision + recall)
    return float(f)


# ---------------------------------------------------------------------------
# Mesh normalization & alignment
# ---------------------------------------------------------------------------

def normalize_mesh_vertices(
    vertices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize mesh: center + scale to unit.

    Returns:
        (normalized_vertices, center, scale)
    """
    center = vertices.mean(axis=0)
    v_centered = vertices - center
    scale = np.linalg.norm(v_centered, axis=1).max()
    scale = max(scale, 1e-8)
    v_normalized = v_centered / scale

    return v_normalized, center, scale


def align_mesh_icp_simple(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    max_iterations: int = 20,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Simple ICP alignment (SVD-based).

    Aligns pred to gt by finding best rotation + isotropic scale.

    Args:
        pred_vertices: [N, 3] predicted mesh vertices
        gt_vertices: [M, 3] ground truth mesh vertices
        max_iterations: ICP iterations

    Returns:
        (pred_aligned, rotation_matrix, scale, fitness)
        - pred_aligned: [N, 3] aligned vertices
        - rotation_matrix: [3, 3] rotation matrix
        - scale: isotropic scale ratio
        - fitness: ICP fitting error (lower is better)
    """
    # Normalize both
    pred_norm, pred_center, pred_scale = normalize_mesh_vertices(pred_vertices)
    gt_norm, gt_center, gt_scale = normalize_mesh_vertices(gt_vertices)

    src = pred_norm.copy()
    tgt = gt_norm

    # ICP iterations
    for iteration in range(max_iterations):
        # Find nearest neighbors
        distances = _nearest_neighbor_distance(src, tgt)

        # Use closest points
        tree_tgt = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(tgt)
        _, indices = tree_tgt.query(src, k=1)
        closest_tgt = tgt[indices]

        # SVD-based best rotation
        H = src.T @ closest_tgt
        try:
            U, _, Vt = np.linalg.svd(H)
            R = (Vt.T @ U.T).T
            # Ensure proper rotation (det=1, not reflection)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = (Vt.T @ U.T).T
        except np.linalg.LinAlgError:
            R = np.eye(3)

        src_old = src.copy()
        src = (src @ R.T)

        # Check convergence
        error = np.linalg.norm(src - src_old)
        if error < 1e-6:
            break

    # Final fitness
    distances = _nearest_neighbor_distance(src, tgt)
    fitness = float(distances.mean())

    # Scale: find best scale factor
    # scale_opt minimizes ||s*src - tgt||^2
    numerator = np.sum(src * closest_tgt)
    denominator = np.sum(src * src)
    scale_factor = numerator / denominator if denominator > 1e-8 else 1.0
    scale_factor = max(scale_factor, 1e-8)

    src_scaled = src * scale_factor

    # Denormalize
    pred_aligned = src_scaled * pred_scale + pred_center

    return pred_aligned, R, scale_factor * (pred_scale / gt_scale), fitness


def align_mesh_open3d(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    pred_faces: np.ndarray | None = None,
    gt_faces: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    ICP alignment using Open3D (more robust).

    Falls back to simple ICP if Open3D not available.

    Returns:
        (pred_aligned, transformation_4x4, scale, fitness)
    """
    try:
        import open3d as o3d
    except ImportError:
        # Fallback to simple ICP
        aligned, R, scale, fitness = align_mesh_icp_simple(
            pred_vertices, gt_vertices
        )
        # Convert to 4x4
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = 0  # translation absorbed in alignment
        return aligned, T, scale, fitness

    # Create point clouds
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_vertices)

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_vertices)

    # ICP registration
    reg = o3d.pipelines.registration.registration_icp(
        pred_pcd,
        gt_pcd,
        max_correspondence_distance=1.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
    )

    T = reg.transformation  # 4x4
    pred_aligned = (T[:3, :3] @ pred_vertices.T).T + T[:3, 3]

    scale = np.cbrt(np.linalg.det(T[:3, :3]))  # Estimate scale from determinant
    fitness = float(reg.fitness)

    return pred_aligned, T, scale, fitness


# ---------------------------------------------------------------------------
# Mesh sampling & conversion
# ---------------------------------------------------------------------------

def sample_mesh_vertices_and_faces(
    mesh_path: str | Path,
    num_samples: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mesh and sample points.

    Returns:
        (vertices, faces, sampled_points)
    """
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)

    # Handle scene
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise ValueError(f"Empty mesh: {mesh_path}")
        mesh = trimesh.util.concatenate(geometries)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Sample points from surface
    sampled_pts, _ = trimesh.sample.sample_surface(mesh, num_samples)
    sampled_pts = np.asarray(sampled_pts, dtype=np.float32)

    return vertices, faces, sampled_pts


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def compute_3d_metrics(
    pred_mesh: str | Path,
    gt_mesh: str | Path,
    num_samples: int = 50000,
    alignment: str = "icp",  # "icp" or "none"
    use_open3d: bool = True,
) -> Mesh3DMetricsResult:
    """
    Compute 3D mesh comparison metrics.

    Args:
        pred_mesh: path to predicted mesh
        gt_mesh: path to ground truth mesh
        num_samples: number of surface points to sample
        alignment: "icp" to align pred to gt, "none" to skip
        use_open3d: use Open3D for ICP (if available)

    Returns:
        Mesh3DMetricsResult with all metrics
    """
    # Load meshes
    pred_verts, pred_faces, pred_pts = sample_mesh_vertices_and_faces(
        pred_mesh, num_samples
    )
    gt_verts, gt_faces, gt_pts = sample_mesh_vertices_and_faces(
        gt_mesh, num_samples
    )

    # Alignment
    if alignment == "icp":
        if use_open3d:
            try:
                pred_aligned, T, scale, fitness = align_mesh_open3d(
                    pred_verts, gt_verts, pred_faces, gt_faces
                )
            except Exception:
                pred_aligned, R, scale, fitness = align_mesh_icp_simple(
                    pred_verts, gt_verts
                )
                T = np.eye(4)
                T[:3, :3] = R
        else:
            pred_aligned, R, scale, fitness = align_mesh_icp_simple(
                pred_verts, gt_verts
            )
            T = np.eye(4)
            T[:3, :3] = R

        # Re-sample from aligned mesh
        pred_pts_aligned = (pred_aligned @ np.eye(3).T).T
        pred_pts_aligned = pred_pts_aligned[
            np.random.choice(len(pred_pts_aligned), num_samples, replace=False)
        ]
    else:
        pred_pts_aligned = pred_pts
        T = None
        scale = 1.0
        fitness = 0.0

    # Metrics
    chamfer, chamfer_fwd, chamfer_bwd = chamfer_distance(pred_pts_aligned, gt_pts)
    hausdorff = hausdorff_distance(pred_pts_aligned, gt_pts)
    f_tau01 = f_score(pred_pts_aligned, gt_pts, tau=0.01)
    f_tau001 = f_score(pred_pts_aligned, gt_pts, tau=0.001)

    return Mesh3DMetricsResult(
        chamfer_distance=chamfer,
        hausdorff_distance=hausdorff,
        f_score_tau01=f_tau01,
        f_score_tau001=f_tau001,
        chamfer_forward=chamfer_fwd,
        chamfer_backward=chamfer_bwd,
        alignment_transform=T,
        alignment_fitness=fitness,
        scale_ratio=scale,
    )
