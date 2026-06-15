from __future__ import annotations

"""
render_consistency.py
---------------------
No-reference visual quality metric cho TripoSR mesh.

Pipeline:
    input image → extract foreground mask
    TripoSR mesh → render silhouette từ N viewpoints
    So sánh mỗi silhouette với input mask → SSIM + IoU
    Trả về best/mean score qua tất cả viewpoints

Không cần Pix3D GT. Ý tưởng: mesh trông giống ảnh input → score cao.

Sử dụng:
    from src.metrics.render_consistency import compute_render_consistency

    result = compute_render_consistency(
        mesh_path="results/triposr_core/chair/mesh.obj",
        image_path="data/processed_2048/images/chair/0001.png",
    )
    print(result.render_consistency_score)
    print(result.best_view_label)
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


# Các góc nhìn chuẩn: (azimuth_deg, elevation_deg, label)
# Azimuth: 0 = nhìn từ phía trước (-Z), tăng theo chiều kim đồng hồ
# Elevation: 0 = ngang, 90 = nhìn từ trên xuống
STANDARD_VIEWPOINTS: list[tuple[float, float, str]] = [
    (0.0, 0.0, "front"),
    (90.0, 0.0, "right"),
    (180.0, 0.0, "back"),
    (270.0, 0.0, "left"),
    (0.0, 75.0, "top"),
    (45.0, 25.0, "iso_fr"),
    (315.0, 25.0, "iso_fl"),
    (135.0, 25.0, "iso_br"),
]


@dataclass
class ViewRenderResult:
    label: str
    azimuth: float
    elevation: float
    silhouette: np.ndarray  # [H, W] bool — không serialize vào JSON
    ssim: float
    iou: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "ssim": self.ssim,
            "iou": self.iou,
        }


@dataclass
class RenderConsistencyResult:
    best_ssim: float
    mean_ssim: float
    best_iou: float
    mean_iou: float
    best_view_label: str
    render_consistency_score: float  # 0.6 * best_ssim + 0.4 * best_iou
    per_view: list[ViewRenderResult] = field(default_factory=list)

    # Mesh quality metrics (không cần input image)
    mesh_watertight: bool = False
    mesh_vertices: int = 0
    mesh_faces: int = 0

    def to_dict(self) -> dict:
        return {
            "best_ssim": self.best_ssim,
            "mean_ssim": self.mean_ssim,
            "best_iou": self.best_iou,
            "mean_iou": self.mean_iou,
            "best_view_label": self.best_view_label,
            "render_consistency_score": self.render_consistency_score,
            "mesh_watertight": self.mesh_watertight,
            "mesh_vertices": self.mesh_vertices,
            "mesh_faces": self.mesh_faces,
            "per_view": [v.to_dict() for v in self.per_view],
        }


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """
    Camera rotation: azimuth xoay quanh Y, elevation xoay quanh X.
    Dùng để đặt viewpoint trước khi render orthographic.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # Xoay quanh Y (azimuth)
    Ry = np.array([
        [math.cos(az),  0.0, math.sin(az)],
        [0.0,           1.0, 0.0         ],
        [-math.sin(az), 0.0, math.cos(az)],
    ], dtype=np.float64)

    # Xoay quanh X (elevation)
    Rx = np.array([
        [1.0, 0.0,           0.0          ],
        [0.0, math.cos(el), -math.sin(el) ],
        [0.0, math.sin(el),  math.cos(el) ],
    ], dtype=np.float64)

    return Ry @ Rx


# ---------------------------------------------------------------------------
# Silhouette rendering (orthographic, ray casting)
# ---------------------------------------------------------------------------

def render_silhouette(
    mesh,  # trimesh.Trimesh
    azimuth_deg: float,
    elevation_deg: float,
    resolution: int = 224,
    margin: float = 1.25,
) -> np.ndarray:
    """
    Render silhouette nhị phân bằng orthographic ray casting.

    Mesh được chuẩn hóa về unit sphere trước khi render,
    sau đó xoay theo góc nhìn và cast rays theo hướng -Z.

    Returns:
        [H, W] bool — True = pixel thuộc mesh
    """
    import trimesh

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if len(faces) == 0 or len(vertices) == 0:
        return np.zeros((resolution, resolution), dtype=bool)

    # Normalize về unit sphere
    center = vertices.mean(axis=0)
    scale = np.linalg.norm(vertices - center, axis=1).max()
    scale = max(scale, 1e-8)
    v_norm = (vertices - center) / scale  # in [-1, 1] roughly

    # Xoay về góc nhìn
    R = _rotation_matrix(azimuth_deg, elevation_deg)
    v_rot = (R @ v_norm.T).T  # [N, 3]

    # Build normalized mesh
    mesh_view = trimesh.Trimesh(vertices=v_rot, faces=faces, process=False)

    # Orthographic ray grid: x, y từ [-margin, margin], ray đi theo -Z
    xs = np.linspace(-margin, margin, resolution)
    ys = np.linspace(margin, -margin, resolution)  # flip Y — image convention
    grid_x, grid_y = np.meshgrid(xs, ys)
    n_rays = resolution * resolution

    ray_origins = np.column_stack([
        grid_x.ravel(),
        grid_y.ravel(),
        np.full(n_rays, 3.0),
    ])
    ray_directions = np.tile([0.0, 0.0, -1.0], (n_rays, 1))

    try:
        hits = mesh_view.ray.intersects_any(ray_origins, ray_directions)
    except Exception:
        hits = _silhouette_convex_fallback(v_rot, grid_x, grid_y).ravel()

    return hits.reshape(resolution, resolution)


def _silhouette_convex_fallback(
    v_rot: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> np.ndarray:
    """
    Fallback khi ray casting thất bại: dùng convex hull 2D của projected vertices.
    Ít chính xác hơn nhưng không cần OpenGL/Embree.
    """
    from scipy.spatial import ConvexHull, Delaunay

    pts_2d = v_rot[:, :2]  # project XY
    if len(pts_2d) < 4:
        return np.zeros(grid_x.size, dtype=bool)

    try:
        hull = ConvexHull(pts_2d)
        hull_pts = pts_2d[hull.vertices]
        tri = Delaunay(hull_pts)
        query = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        return tri.find_simplex(query) >= 0
    except Exception:
        return np.zeros(grid_x.size, dtype=bool)


# ---------------------------------------------------------------------------
# Input image mask extraction
# ---------------------------------------------------------------------------

def extract_foreground_mask(
    image_path: str | Path,
    resolution: int = 224,
) -> np.ndarray:
    """
    Trích mask foreground từ ảnh input.

    Xử lý 3 loại background:
    - Trắng (255): Pix3D processed images
    - Xám (128): TripoSR processed input (triposr_input.png)
    - Trong suốt (alpha): PNG với alpha channel
    """
    image_path = Path(image_path)
    img = Image.open(image_path)
    img = img.resize((resolution, resolution), Image.Resampling.BILINEAR)
    img_np = np.asarray(img)

    # Alpha channel → dùng trực tiếp
    if img_np.ndim == 3 and img_np.shape[2] == 4:
        return img_np[:, :, 3] > 20

    # Grayscale
    if img_np.ndim == 2:
        mean_val = float(img_np.mean())
        if mean_val > 200:  # white background
            return img_np < 240
        elif 110 < mean_val < 145:  # gray background
            return np.abs(img_np.astype(np.int32) - 128) > 15
        return img_np > 20

    # RGB
    img_rgb = img_np[:, :, :3].astype(np.int32)
    mean_brightness = float(img_rgb.mean())

    if mean_brightness > 200:
        # White background (Pix3D): pixel có màu khác trắng
        return np.any(img_rgb < 240, axis=2)
    elif 110 < mean_brightness < 145:
        # Gray background (TripoSR triposr_input.png): pixel lệch khỏi gray 128
        diff = np.abs(img_rgb - 128).max(axis=2)
        return diff > 20
    else:
        # Dark background
        return np.any(img_rgb > 20, axis=2)


# ---------------------------------------------------------------------------
# SSIM và IoU
# ---------------------------------------------------------------------------

def _compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM giữa hai binary masks. Dùng skimage nếu có, fallback về NCC."""
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(
            pred.astype(np.float32),
            gt.astype(np.float32),
            data_range=1.0,
        ))
    except Exception:
        pass

    # Fallback: Normalized cross-correlation
    p = pred.astype(np.float32) - pred.mean()
    g = gt.astype(np.float32) - gt.mean()
    denom = np.linalg.norm(p) * np.linalg.norm(g)
    if denom < 1e-8:
        return 0.0
    return float(np.clip(np.sum(p * g) / denom, -1.0, 1.0))


def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# Mesh quality (không cần input image)
# ---------------------------------------------------------------------------

def compute_mesh_quality(mesh) -> dict[str, object]:
    """Tính các mesh quality metrics không cần GT."""
    try:
        watertight = bool(mesh.is_watertight)
    except Exception:
        watertight = False

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = getattr(mesh, "faces", np.empty((0, 3)))

    # Dihedral angle variance (surface smoothness proxy)
    face_normals_var = 0.0
    try:
        normals = mesh.face_normals
        if len(normals) > 1:
            face_normals_var = float(np.var(normals, axis=0).mean())
    except Exception:
        pass

    return {
        "mesh_watertight": watertight,
        "mesh_vertices": int(len(vertices)),
        "mesh_faces": int(len(faces)),
        "face_normals_variance": face_normals_var,
    }


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def compute_render_consistency(
    mesh_path: str | Path,
    image_path: str | Path,
    resolution: int = 224,
    viewpoints: list[tuple[float, float, str]] | None = None,
) -> RenderConsistencyResult:
    """
    Tính render consistency score giữa TripoSR mesh và ảnh input.

    Args:
        mesh_path:   Đường dẫn tới mesh file (.obj, .ply, .glb)
        image_path:  Đường dẫn tới ảnh input gốc hoặc triposr_input.png
        resolution:  Độ phân giải render (mặc định 224 khớp với Pix3D processed)
        viewpoints:  Danh sách (azimuth, elevation, label). None = dùng STANDARD_VIEWPOINTS

    Returns:
        RenderConsistencyResult với best/mean SSIM và IoU qua tất cả viewpoints,
        kèm per-view breakdown và mesh quality metrics.
    """
    import trimesh

    if viewpoints is None:
        viewpoints = STANDARD_VIEWPOINTS

    # Load mesh
    loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"Empty scene mesh: {mesh_path}")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded

    if len(getattr(mesh, "faces", [])) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    # Mesh quality metrics
    quality = compute_mesh_quality(mesh)

    # Extract input image foreground mask
    gt_mask = extract_foreground_mask(image_path, resolution=resolution)

    # Render + compare từng viewpoint
    per_view: list[ViewRenderResult] = []
    for az, el, label in viewpoints:
        silhouette = render_silhouette(mesh, az, el, resolution=resolution)
        ssim = _compute_ssim(silhouette, gt_mask)
        iou = _compute_iou(silhouette, gt_mask)
        per_view.append(ViewRenderResult(
            label=label,
            azimuth=az,
            elevation=el,
            silhouette=silhouette,
            ssim=ssim,
            iou=iou,
        ))

    best_view = max(per_view, key=lambda v: v.ssim)
    best_ssim = float(best_view.ssim)
    mean_ssim = float(np.mean([v.ssim for v in per_view]))
    best_iou = float(max(v.iou for v in per_view))
    mean_iou = float(np.mean([v.iou for v in per_view]))

    # Combined score: weighted blend
    render_consistency_score = float(0.6 * best_ssim + 0.4 * best_iou)

    return RenderConsistencyResult(
        best_ssim=best_ssim,
        mean_ssim=mean_ssim,
        best_iou=best_iou,
        mean_iou=mean_iou,
        best_view_label=best_view.label,
        render_consistency_score=render_consistency_score,
        per_view=per_view,
        mesh_watertight=quality["mesh_watertight"],
        mesh_vertices=quality["mesh_vertices"],
        mesh_faces=quality["mesh_faces"],
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_render_consistency_figure(
    result: RenderConsistencyResult,
    image_path: str | Path,
    output_path: str | Path,
    resolution: int = 224,
    max_views: int = 4,
) -> Path:
    """
    Lưu hình so sánh: [input image] [input mask] [best view] [top-N other views].

    Mỗi cột là một panel.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Chuẩn bị dữ liệu
    input_img = np.asarray(
        Image.open(str(image_path)).convert("RGB").resize((resolution, resolution))
    )
    gt_mask = extract_foreground_mask(image_path, resolution=resolution)

    # Sắp xếp views theo SSIM giảm dần
    sorted_views = sorted(result.per_view, key=lambda v: v.ssim, reverse=True)
    show_views = sorted_views[:max_views]

    n_panels = 2 + len(show_views)
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]

    # Panel 0: input image
    axes[0].imshow(input_img)
    axes[0].set_title("Input image", fontsize=9)
    axes[0].axis("off")

    # Panel 1: input mask
    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Input mask", fontsize=9)
    axes[1].axis("off")

    # Panel 2+: rendered silhouettes
    for idx, view in enumerate(show_views):
        ax = axes[2 + idx]
        ax.imshow(view.silhouette, cmap="gray", vmin=0, vmax=1)
        star = "★ " if view.label == result.best_view_label else ""
        ax.set_title(
            f"{star}{view.label}\nSSIM={view.ssim:.3f} IoU={view.iou:.3f}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(
        f"Render Consistency | score={result.render_consistency_score:.3f} "
        f"best_ssim={result.best_ssim:.3f} best_iou={result.best_iou:.3f} "
        f"watertight={result.mesh_watertight}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path