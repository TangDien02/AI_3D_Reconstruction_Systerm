from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataloader import ProcessedPix3DDataset
from src.inference.baseline_inference import load_baseline_model, model_points, select_device
from src.inference.compare_pointclouds import save_comparison_figure
from src.metrics.pointcloud_quality import ALL_POINTCLOUD_METRICS, compute_pointcloud_quality_metrics


def batch_value(batch: dict, key: str, index: int, default: object = "") -> object:
    if key not in batch:
        return default
    value = batch[key]
    if isinstance(value, (list, tuple)):
        return value[index]
    if torch.is_tensor(value):
        item = value[index]
        return item.item() if item.ndim == 0 else item.detach().cpu().tolist()
    return value


def mean_metric_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    summary = {}
    for metric_name in ALL_POINTCLOUD_METRICS:
        values = [float(row[metric_name]) for row in rows if metric_name in row]
        if values:
            summary[metric_name] = float(sum(values) / len(values))
    return summary


def category_metric_rows(sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    categories = sorted({str(row.get("category", "")) for row in sample_rows})
    output_rows = []
    for category in categories:
        rows = [row for row in sample_rows if str(row.get("category", "")) == category]
        output_rows.append(
            {
                "category": category,
                "samples": len(rows),
                **mean_metric_rows(rows),
            }
        )
    return output_rows


def worst_case_sort_value(row: dict[str, object], metric_name: str) -> float:
    value = row.get(metric_name)
    if value is None:
        raise KeyError(f"Unknown worst-case metric: {metric_name}")
    return float(value)


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

    batch_rows = []
    sample_rows = []
    worst_candidates: list[dict[str, object]] = []
    metric_totals = {name: 0.0 for name in ALL_POINTCLOUD_METRICS}
    sample_count = 0
    dataset_index = 0

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device)
        points_gt = batch["points_gt"].to(device)
        points_pred = model_points(model, images)

        current_batch_size = images.shape[0]
        current_sample_rows = []

        for batch_item in range(current_batch_size):
            pred_sample = points_pred[batch_item : batch_item + 1]
            gt_sample = points_gt[batch_item : batch_item + 1]
            sample_metrics = compute_pointcloud_quality_metrics(
                pred_sample,
                gt_sample,
                threshold=args.f_threshold,
                fine_threshold=getattr(args, "fine_threshold", None),
                loose_threshold=getattr(args, "loose_threshold", None),
                density_sample_size=getattr(args, "density_sample_size", 512),
                voxel_resolution=getattr(args, "voxel_resolution", 32),
                occupancy_dilation=getattr(args, "occupancy_dilation", 1),
            )
            row = {
                "dataset_index": dataset_index,
                "batch": batch_index,
                "batch_item": batch_item,
                "sample_id": batch_value(batch, "sample_id", batch_item, f"{args.split}_{dataset_index:05d}"),
                "category": batch_value(batch, "category", batch_item),
                "model_uid": batch_value(batch, "model_uid", batch_item),
                "image_path": batch_value(batch, "image_path", batch_item),
                "pointcloud_path": batch_value(batch, "pointcloud_path", batch_item),
                **sample_metrics,
            }
            sample_rows.append(row)
            current_sample_rows.append(row)
            sample_count += 1
            for metric_name in ALL_POINTCLOUD_METRICS:
                metric_totals[metric_name] += sample_metrics[metric_name]

            if args.worst_case_count > 0 and not args.skip_worst_gallery:
                pred_np = pred_sample[0].detach().cpu().numpy().astype(np.float32)
                gt_np = gt_sample[0].detach().cpu().numpy().astype(np.float32)
                worst_candidates.append(
                    {
                        "row": row,
                        "pred_points": pred_np,
                        "gt_points": gt_np,
                    }
                )
                reverse = args.worst_case_mode == "max"
                worst_candidates.sort(
                    key=lambda item: worst_case_sort_value(item["row"], args.worst_case_metric),
                    reverse=reverse,
                )
                worst_candidates = worst_candidates[: args.worst_case_count]

            dataset_index += 1

        batch_metrics = mean_metric_rows(current_sample_rows)
        batch_rows.append(
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
        writer.writerows(batch_rows)

    sample_metrics_path = metric_dir / f"{args.split}_sample_metrics.csv"
    with sample_metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset_index",
                "batch",
                "batch_item",
                "sample_id",
                "category",
                "model_uid",
                "image_path",
                "pointcloud_path",
                *ALL_POINTCLOUD_METRICS,
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    category_metrics_path = metric_dir / f"{args.split}_category_metrics.csv"
    category_rows = category_metric_rows(sample_rows)
    with category_metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["category", "samples", *ALL_POINTCLOUD_METRICS],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(category_rows)

    worst_cases_dir = output_dir / "outputs" / f"{args.split}_worst_cases"
    worst_cases_metrics_path = None
    if worst_candidates:
        worst_cases_dir.mkdir(parents=True, exist_ok=True)
        worst_rows = []
        for rank, item in enumerate(worst_candidates, start=1):
            row = dict(item["row"])
            sample_id = str(row.get("sample_id", f"sample_{rank:02d}"))
            prefix = f"{rank:02d}_{sample_id}"
            figure_path = save_comparison_figure(
                pred_points=item["pred_points"],
                gt_points=item["gt_points"],
                output_path=worst_cases_dir / f"{prefix}_comparison.png",
                metrics=row,
                sample_id=sample_id,
                max_plot_points=args.max_plot_points,
                show=False,
            )
            metrics_path = worst_cases_dir / f"{prefix}_metrics.json"
            row["worst_rank"] = rank
            row["comparison_path"] = str(figure_path)
            metrics_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            worst_rows.append(row)

        worst_cases_metrics_path = worst_cases_dir / "worst_cases.csv"
        with worst_cases_metrics_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "worst_rank",
                    "dataset_index",
                    "sample_id",
                    "category",
                    "model_uid",
                    "image_path",
                    "pointcloud_path",
                    "comparison_path",
                    *ALL_POINTCLOUD_METRICS,
                ],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(worst_rows)

    summary = {
        "split": args.split,
        "categories": args.categories or checkpoint.get("categories"),
        "samples": sample_count,
        "batch_size": args.batch_size,
        "checkpoint_path": str(checkpoint_path),
        **{metric_name: metric_totals[metric_name] / sample_count for metric_name in ALL_POINTCLOUD_METRICS},
        "batch_metrics_path": str(batch_metrics_path),
        "sample_metrics_path": str(sample_metrics_path),
        "category_metrics_path": str(category_metrics_path),
        "worst_cases_dir": str(worst_cases_dir) if worst_candidates else None,
        "worst_cases_metrics_path": str(worst_cases_metrics_path) if worst_cases_metrics_path else None,
        "worst_case_metric": args.worst_case_metric,
        "worst_case_mode": args.worst_case_mode,
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
    parser.add_argument("--worst-case-count", type=int, default=20)
    parser.add_argument("--worst-case-metric", default="visual_completeness_score")
    parser.add_argument("--worst-case-mode", choices=["min", "max"], default="min")
    parser.add_argument("--skip-worst-gallery", action="store_true")
    parser.add_argument("--max-plot-points", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    evaluate_checkpoint(parse_args())


if __name__ == "__main__":
    main()

