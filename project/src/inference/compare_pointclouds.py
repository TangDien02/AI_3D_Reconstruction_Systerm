from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataloader import ProcessedPix3DDataset
from src.inference.baseline_inference import load_baseline_model, model_points, select_device
from src.metrics.pointcloud_quality import compute_pointcloud_quality_metrics


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[indices]


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    radius = max(np.ptp(points, axis=0).max() / 2, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_comparison_figure(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    output_path: str | Path,
    metrics: dict[str, float],
    sample_id: str,
    max_plot_points: int = 2048,
    show: bool = False,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_plot = sample_points(pred_points, max_plot_points)
    gt_plot = sample_points(gt_points, max_plot_points)
    all_points = np.concatenate([pred_plot, gt_plot], axis=0)

    fig = plt.figure(figsize=(12, 5.5))
    title = (
        "Point cloud comparison | "
        f"sample={sample_id} | "
        f"CD={metrics['chamfer_distance']:.6f} | "
        f"F={metrics['f_score']:.4f} | "
        f"P={metrics['precision']:.4f} | "
        f"R={metrics['recall']:.4f}"
    )
    if "visual_completeness_score" in metrics:
        title += (
            f" | VC={metrics['visual_completeness_score']:.4f} | "
            f"Occ={metrics['occupancy_iou']:.4f} | "
            f"Empty={metrics['empty_space_violation']:.4f}"
        )
    fig.suptitle(title, fontsize=10.5)

    ax_pred = fig.add_subplot(121, projection="3d")
    ax_pred.scatter(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2], s=3, alpha=0.7, c="#2563eb")
    ax_pred.set_title("Predicted point cloud")
    ax_pred.set_xlabel("X")
    ax_pred.set_ylabel("Y")
    ax_pred.set_zlabel("Z")
    set_equal_3d_axes(ax_pred, all_points)

    ax_gt = fig.add_subplot(122, projection="3d")
    ax_gt.scatter(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], s=3, alpha=0.7, c="#dc2626")
    ax_gt.set_title("Ground-truth point cloud")
    ax_gt.set_xlabel("X")
    ax_gt.set_ylabel("Y")
    ax_gt.set_zlabel("Z")
    set_equal_3d_axes(ax_gt, all_points)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def compare_sample(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_DIR / processed_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir

    model, checkpoint = load_baseline_model(checkpoint_path, device)

    categories = args.categories
    if categories is None:
        categories = checkpoint.get("categories")
    input_mode = getattr(args, "input_mode", None) or checkpoint.get("input_mode", "rgb")
    mask_background = getattr(args, "mask_background", None) or checkpoint.get("mask_background", "white")

    dataset = ProcessedPix3DDataset(
        processed_dir=processed_dir,
        split=args.split,
        categories=categories,
        max_samples=args.max_samples,
        expected_num_points=int(checkpoint.get("num_points", 2048)),
        input_mode=input_mode,
        mask_background=mask_background,
        exclude_truncated=bool(getattr(args, "exclude_truncated", False)),
        exclude_occluded=bool(getattr(args, "exclude_occluded", False)),
        exclude_slightly_occluded=bool(getattr(args, "exclude_slightly_occluded", False)),
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No samples found. Check processed_dir, split, categories, and processed image/pointcloud files."
        )

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index must be between 0 and {len(dataset) - 1}.")

    sample = dataset[args.index]
    image = sample["image"].unsqueeze(0).to(device)
    gt_points = sample["points_gt"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_points = model_points(model, image)
        metrics = compute_pointcloud_quality_metrics(
            pred_points,
            gt_points,
            threshold=args.f_threshold,
            fine_threshold=args.fine_threshold,
            loose_threshold=args.loose_threshold,
            density_sample_size=args.density_sample_size,
            voxel_resolution=args.voxel_resolution,
            occupancy_dilation=args.occupancy_dilation,
        )

    pred_np = pred_points[0].detach().cpu().numpy().astype(np.float32)
    gt_np = gt_points[0].detach().cpu().numpy().astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_id = str(dataset.items.iloc[args.index].get("sample_id", f"{args.split}_{args.index:05d}"))
    np.save(output_dir / f"{sample_id}_pred.npy", pred_np)
    np.save(output_dir / f"{sample_id}_gt.npy", gt_np)

    figure_path = save_comparison_figure(
        pred_points=pred_np,
        gt_points=gt_np,
        output_path=output_dir / f"{sample_id}_comparison.png",
        metrics=metrics,
        sample_id=sample_id,
        max_plot_points=args.max_plot_points,
        show=args.show,
    )
    metrics_path = output_dir / f"{sample_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Sample: {sample_id}")
    print(f"Category: {sample['category']}")
    print(f"Input mode: {input_mode} mask_background={mask_background}")
    print(f"Image: {sample['image_path']}")
    print(f"GT pointcloud: {sample['pointcloud_path']}")
    print(f"Pred shape: {pred_np.shape}")
    print(f"GT shape: {gt_np.shape}")
    print(f"Chamfer Distance: {metrics['chamfer_distance']:.8f}")
    print(f"F-score: {metrics['f_score']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"Fine F-score: {metrics['fine_f_score']:.6f}")
    print(f"Fine Recall: {metrics['fine_recall']:.6f}")
    print(f"Occupancy IoU: {metrics['occupancy_iou']:.6f}")
    print(f"Empty-space violation: {metrics['empty_space_violation']:.6f}")
    print(f"Density score: {metrics['density_score']:.6f}")
    print(f"Clump ratio: {metrics['clump_ratio']:.6f}")
    print(f"Surface alignment score: {metrics['surface_alignment_score']:.6f}")
    print(f"Detail preservation score: {metrics['detail_preservation_score']:.6f}")
    print(f"Structure occupancy score: {metrics['structure_occupancy_score']:.6f}")
    print(f"Empty-space score: {metrics['empty_space_score']:.6f}")
    print(f"Density uniformity score: {metrics['density_uniformity_score']:.6f}")
    print(f"Visual completeness score: {metrics['visual_completeness_score']:.6f}")
    print(f"Visual completeness percent: {metrics['visual_completeness_percent']:.2f}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved metrics JSON: {metrics_path}")
    print(f"Saved predicted NPY: {output_dir / f'{sample_id}_pred.npy'}")
    print(f"Saved GT NPY: {output_dir / f'{sample_id}_gt.npy'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize predicted and ground-truth point clouds.")
    parser.add_argument(
        "--checkpoint",
        default="results/all_categories_resnet50_2048pts_30ep_aug/outputs/checkpoints/best_model.pt",
    )
    parser.add_argument("--processed-dir", default="data/processed_2048")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--input-mode", choices=["rgb", "masked_rgb"], default=None)
    parser.add_argument("--mask-background", choices=["white", "black"], default=None)
    parser.add_argument("--exclude-truncated", action="store_true")
    parser.add_argument("--exclude-occluded", action="store_true")
    parser.add_argument("--exclude-slightly-occluded", action="store_true")
    parser.add_argument("--output-dir", default="results/all_categories_resnet50_2048pts_30ep_aug/outputs/comparison")
    parser.add_argument("--f-threshold", type=float, default=0.05)
    parser.add_argument("--fine-threshold", type=float, default=None)
    parser.add_argument("--loose-threshold", type=float, default=None)
    parser.add_argument("--density-sample-size", type=int, default=512)
    parser.add_argument("--voxel-resolution", type=int, default=32)
    parser.add_argument("--occupancy-dilation", type=int, default=1)
    parser.add_argument("--max-plot-points", type=int, default=2048)
    parser.add_argument("--device", default=None, help="Use cuda, cpu, or leave empty for auto.")
    parser.add_argument("--show", action="store_true", help="Open a matplotlib window after saving the figure.")
    return parser.parse_args()


def main() -> None:
    compare_sample(parse_args())


if __name__ == "__main__":
    main()
