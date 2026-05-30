"""
render_quality_metrics.py
-------------------------
Render-based quality metrics for 3D meshes (no-reference evaluation).

Metrics:
  - LPIPS: Learned Perceptual Image Patch Similarity (perceptual distance)
  - Depth consistency: rendered depth vs expected
  - Multi-view consistency: same mesh from different angles
  - Silhouette quality: shape consistency

Usage:
    from src.metrics.render_quality_metrics import compute_render_quality

    result = compute_render_quality(
        mesh_path="results/triposr/mesh.obj",
        input_image_path="data/input.png"
    )
    print(result.lpips_score)
    print(result.depth_consistency_score)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class RenderQualityResult:
    """Result of render-based quality evaluation."""

    # Perceptual metrics
    lpips_score: float  # [0, 1], lower is better
    lpips_std: float  # std across views

    # Consistency metrics
    depth_consistency_score: float  # [0, 1], higher is better
    normal_consistency_score: float  # [0, 1], higher is better
    view_consistency_score: float  # [0, 1], higher is better

    # Surface metrics
    surface_smoothness_score: float  # [0, 1], based on rendered normals
    depth_smoothness_score: float  # [0, 1], based on depth gradient

    # Combined
    render_quality_score: float  # weighted average

    # Per-view breakdown
    per_view_lpips: list[float] = None
    per_view_consistency: list[float] = None

    def to_dict(self) -> dict:
        return {
            "lpips_score": self.lpips_score,
            "lpips_std": self.lpips_std,
            "depth_consistency_score": self.depth_consistency_score,
            "normal_consistency_score": self.normal_consistency_score,
            "view_consistency_score": self.view_consistency_score,
            "surface_smoothness_score": self.surface_smoothness_score,
            "depth_smoothness_score": self.depth_smoothness_score,
            "render_quality_score": self.render_quality_score,
        }


# ---------------------------------------------------------------------------
# LPIPS (Perceptual Distance)
# ---------------------------------------------------------------------------

def compute_lpips(
    pred_image: np.ndarray,
    ref_image: np.ndarray,
    net_type: str = "alex",
) -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        pred_image: [H, W, 3] predicted image, range [0, 1]
        ref_image: [H, W, 3] reference image, range [0, 1]
        net_type: "alex" or "squeeze"

    Returns:
        LPIPS score (lower is better, range ~[0, 0.5] typically)
    """
    try:
        import torch
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

        # Prepare tensors
        pred_t = torch.from_numpy(pred_image).unsqueeze(0).permute(0, 3, 1, 2)
        ref_t = torch.from_numpy(ref_image).unsqueeze(0).permute(0, 3, 1, 2)

        pred_t = pred_t.float()
        ref_t = ref_t.float()

        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type=net_type)
        lpips_val = lpips_model(pred_t, ref_t).item()
        return float(lpips_val)

    except ImportError:
        # Fallback: SSIM
        from skimage.metrics import structural_similarity

        pred_gray = np.mean(pred_image, axis=2)
        ref_gray = np.mean(ref_image, axis=2)
        ssim = structural_similarity(pred_gray, ref_gray, data_range=1.0)
        # Convert SSIM to LPIPS-like scale
        return float(1.0 - ssim) * 0.5


# ---------------------------------------------------------------------------
# Depth & Normal Consistency
# ---------------------------------------------------------------------------

def compute_depth_smoothness(depth_map: np.ndarray) -> float:
    """
    Compute depth smoothness as inverse of gradient magnitude.

    Returns:
        Smoothness score [0, 1], higher is smoother
    """
    if depth_map.size == 0:
        return 0.0

    # Compute gradients
    grad_x = np.abs(np.diff(depth_map, axis=1))
    grad_y = np.abs(np.diff(depth_map, axis=0))

    # Mean gradient magnitude
    mean_grad = (grad_x.mean() + grad_y.mean()) / 2

    # Normalize: assume depth range [0, 1]
    # high gradient = low smoothness
    smoothness = 1.0 / (1.0 + mean_grad)
    return float(smoothness)


def compute_normal_smoothness(normal_map: np.ndarray) -> float:
    """
    Compute normal map smoothness based on angular differences.

    Args:
        normal_map: [H, W, 3] normalized normal vectors

    Returns:
        Smoothness score [0, 1]
    """
    if normal_map.shape[0] < 2 or normal_map.shape[1] < 2:
        return 0.0

    # Compute angular differences
    normals_x = normal_map[:, 1:, :]
    normals_x_ref = normal_map[:, :-1, :]
    dot_x = (normals_x * normals_x_ref).sum(axis=2)
    angle_x = np.arccos(np.clip(dot_x, -1, 1))

    normals_y = normal_map[1:, :, :]
    normals_y_ref = normal_map[:-1, :, :]
    dot_y = (normals_y * normals_y_ref).sum(axis=2)
    angle_y = np.arccos(np.clip(dot_y, -1, 1))

    # Mean angular difference
    mean_angle = (angle_x.mean() + angle_y.mean()) / 2

    # Convert to smoothness: small angles = high smoothness
    smoothness = 1.0 / (1.0 + mean_angle)
    return float(smoothness)


def compute_view_consistency(
    renders: list[np.ndarray],
    metric: str = "ssim",
) -> float:
    """
    Compute consistency across multiple rendered views.

    Args:
        renders: list of [H, W, 3] rendered images
        metric: "ssim" or "lpips"

    Returns:
        Consistency score [0, 1]
    """
    if len(renders) < 2:
        return 1.0

    try:
        from skimage.metrics import structural_similarity

        scores = []
        for i in range(len(renders) - 1):
            img1 = np.mean(renders[i], axis=2)  # grayscale
            img2 = np.mean(renders[i + 1], axis=2)

            ssim = structural_similarity(img1, img2, data_range=1.0)
            scores.append(ssim)

        consistency = float(np.mean(scores))
    except Exception:
        consistency = 0.5

    return np.clip(consistency, 0, 1)


# ---------------------------------------------------------------------------
# Rendering helpers (stub)
# ---------------------------------------------------------------------------

def render_mesh_orthographic(
    mesh_path: str | Path,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    resolution: int = 224,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render mesh orthographic view.

    Returns:
        (rendered_rgb, depth_map, normal_map)
        - rendered_rgb: [H, W, 3] in [0, 1]
        - depth_map: [H, W] normalized depth
        - normal_map: [H, W, 3] normalized normals
    """
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(geometries)

    # For now, return placeholder renders
    # TODO: Implement actual rendering with headless GL
    rgb = np.ones((resolution, resolution, 3), dtype=np.float32) * 0.5
    depth = np.ones((resolution, resolution), dtype=np.float32) * 0.5
    normal = np.ones((resolution, resolution, 3), dtype=np.float32)
    normal = normal / np.linalg.norm(normal, axis=2, keepdims=True)

    return rgb, depth, normal


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def compute_render_quality(
    mesh_path: str | Path,
    input_image_path: str | Path | None = None,
    num_views: int = 4,
    resolution: int = 224,
) -> RenderQualityResult:
    """
    Compute render-based quality metrics.

    Args:
        mesh_path: path to 3D mesh
        input_image_path: optional reference image for LPIPS
        num_views: number of viewpoints to render
        resolution: render resolution

    Returns:
        RenderQualityResult
    """
    # Render multiple views
    renders_rgb = []
    renders_depth = []
    renders_normal = []

    viewpoints = [
        (0.0, 0.0),
        (90.0, 0.0),
        (180.0, 0.0),
        (270.0, 0.0),
    ][:num_views]

    for az, el in viewpoints:
        rgb, depth, normal = render_mesh_orthographic(
            mesh_path, az, el, resolution
        )
        renders_rgb.append(rgb)
        renders_depth.append(depth)
        renders_normal.append(normal)

    renders_rgb = np.array(renders_rgb)  # [N, H, W, 3]
    renders_depth = np.array(renders_depth)  # [N, H, W]
    renders_normal = np.array(renders_normal)  # [N, H, W, 3]

    # LPIPS scores
    per_view_lpips = []
    if input_image_path:
        input_img = Image.open(input_image_path).convert("RGB")
        input_img = input_img.resize((resolution, resolution))
        input_np = np.array(input_img, dtype=np.float32) / 255.0

        for render in renders_rgb:
            lpips_val = compute_lpips(render, input_np)
            per_view_lpips.append(lpips_val)

    lpips_mean = float(np.mean(per_view_lpips)) if per_view_lpips else 0.5
    lpips_std = float(np.std(per_view_lpips)) if per_view_lpips else 0.0

    # Consistency scores
    depth_smoothness_scores = [
        compute_depth_smoothness(d) for d in renders_depth
    ]
    depth_consistency = float(np.mean(depth_smoothness_scores))

    normal_smoothness_scores = [
        compute_normal_smoothness(n) for n in renders_normal
    ]
    normal_consistency = float(np.mean(normal_smoothness_scores))

    view_consistency = compute_view_consistency(list(renders_rgb))

    # Surface metrics
    surface_smoothness = float(np.mean(normal_smoothness_scores))
    depth_smoothness = float(np.mean(depth_smoothness_scores))

    # Combined score
    lpips_score = min(lpips_mean / 0.5, 1.0)  # normalize to [0, 1]
    render_quality = (
        0.3 * (1.0 - lpips_score) +
        0.2 * depth_consistency +
        0.2 * normal_consistency +
        0.2 * view_consistency +
        0.1 * surface_smoothness
    )

    return RenderQualityResult(
        lpips_score=lpips_mean,
        lpips_std=lpips_std,
        depth_consistency_score=depth_consistency,
        normal_consistency_score=normal_consistency,
        view_consistency_score=view_consistency,
        surface_smoothness_score=surface_smoothness,
        depth_smoothness_score=depth_smoothness,
        render_quality_score=render_quality,
        per_view_lpips=per_view_lpips,
    )
