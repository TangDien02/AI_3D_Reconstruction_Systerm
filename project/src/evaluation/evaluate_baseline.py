from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataloader import ProcessedPix3DDataset
from src.inference.baseline_inference import load_baseline_model, model_points, select_device
from src.metrics.pointcloud_quality import ALL_POINTCLOUD_METRICS, compute_pointcloud_quality_metrics


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    processed_dir = (PROJECT_DIR / args.processed_dir).resolve()
    checkpoint_path = (PROJECT_DIR / args.checkpoint).resolve()
    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    metric_dir = output_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(getattr(args, "device", "auto"))
    model, checkpoint = load_baseline_model(checkpoint_path, device=device)
    dataset = ProcessedPix3DDataset(
        processed_dir=processed_dir,
        split=args.split,
        categories=args.categories or checkpoint.get("categories"),
        max_samples=args.max_samples,
        expected_num_points=int(checkpoint.get("num_points", 2048)),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    rows = []
    metric_totals = {name: 0.0 for name in ALL_POINTCLOUD_METRICS}
    sample_count = 0

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device)
        points_gt = batch["points_gt"].to(device)
        points_pred = model_points(model, images)

        batch_metrics = compute_pointcloud_quality_metrics(
            points_pred,
            points_gt,
            threshold=args.f_threshold,
            fine_threshold=getattr(args, "fine_threshold", None),
            loose_threshold=getattr(args, "loose_threshold", None),
            density_sample_size=getattr(args, "density_sample_size", 512),
            voxel_resolution=getattr(args, "voxel_resolution", 32),
            occupancy_dilation=getattr(args, "occupancy_dilation", 1),
        )
        current_batch_size = images.shape[0]
        sample_count += current_batch_size
        for metric_name in ALL_POINTCLOUD_METRICS:
            metric_totals[metric_name] += batch_metrics[metric_name] * current_batch_size

        rows.append(
            {
                "batch": batch_index,
                "batch_size": current_batch_size,
                **batch_metrics,
            }
        )
        print(
            f"batch={batch_index} size={current_batch_size} "
            f"cd={batch_metrics['chamfer_distance']:.6f} "
            f"f={batch_metrics['f_score']:.4f} "
            f"vc={batch_metrics['visual_completeness_score']:.4f} "
            f"empty={batch_metrics['empty_space_violation']:.4f}"
        )

    if sample_count == 0:
        raise RuntimeError("Evaluation dataset is empty.")

    batch_metrics_path = metric_dir / f"{args.split}_batch_metrics.csv"
    with batch_metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["batch", "batch_size", *ALL_POINTCLOUD_METRICS],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "categories": args.categories or checkpoint.get("categories"),
        "samples": sample_count,
        "batch_size": args.batch_size,
        "checkpoint_path": str(checkpoint_path),
        **{metric_name: metric_totals[metric_name] / sample_count for metric_name in ALL_POINTCLOUD_METRICS},
        "batch_metrics_path": str(batch_metrics_path),
        "visual_diagnostics": {
            "fine_threshold": getattr(args, "fine_threshold", None) or args.f_threshold * 0.5,
            "loose_threshold": getattr(args, "loose_threshold", None) or args.f_threshold * 2.0,
            "voxel_resolution": getattr(args, "voxel_resolution", 32),
            "occupancy_dilation": getattr(args, "occupancy_dilation", 1),
            "density_sample_size": getattr(args, "density_sample_size", 512),
            "notes": {
                "fine_f_score": "F-score at stricter threshold; useful for thin details such as chair legs.",
                "empty_space_violation": "Predicted occupied voxels outside dilated GT occupancy; lower is better for holes/open spaces.",
                "density_score": "Nearest-neighbor uniformity score; higher is better and lower clumping is expected.",
                "visual_completeness_score": "Unified visual score from surface, detail, structure, empty-space, and density components.",
                "visual_quality_score": "Backward-compatible alias of visual_completeness_score.",
            },
        },
    }
    summary_path = metric_dir / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved evaluation summary to {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the baseline checkpoint on a processed split.")
    parser.add_argument("--processed-dir", default="data/processed_2048")
    parser.add_argument(
        "--checkpoint",
        default="results/all_categories_resnet50_2048pts_30ep_aug/outputs/checkpoints/best_model.pt",
    )
    parser.add_argument("--output-dir", default="results/all_categories_resnet50_2048pts_30ep_aug")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--f-threshold", type=float, default=0.05)
    parser.add_argument("--fine-threshold", type=float, default=None)
    parser.add_argument("--loose-threshold", type=float, default=None)
    parser.add_argument("--density-sample-size", type=int, default=512)
    parser.add_argument("--voxel-resolution", type=int, default=32)
    parser.add_argument("--occupancy-dilation", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    evaluate_checkpoint(parse_args())


if __name__ == "__main__":
    main()

