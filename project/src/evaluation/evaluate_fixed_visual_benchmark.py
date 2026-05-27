from __future__ import annotations

import argparse
import csv
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
from src.inference.compare_pointclouds import save_comparison_figure
from src.metrics.pointcloud_quality import ALL_POINTCLOUD_METRICS, compute_pointcloud_quality_metrics


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_DIR / path


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Fixed benchmark manifest not found: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"Fixed benchmark manifest is empty: {manifest_path}")
    required_columns = {"benchmark_index", "dataset_index", "sample_id", "category"}
    missing = required_columns.difference(rows[0].keys())
    if missing:
        raise KeyError(f"Manifest is missing columns: {sorted(missing)}")
    return rows


def mean_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    summary = {}
    for metric_name in ALL_POINTCLOUD_METRICS:
        values = [float(row[metric_name]) for row in rows if metric_name in row]
        if values:
            summary[metric_name] = float(sum(values) / len(values))
    return summary


@torch.no_grad()
def evaluate_fixed_visual_benchmark(args: argparse.Namespace) -> dict:
    manifest_path = resolve_project_path(args.manifest)
    checkpoint_path = resolve_project_path(args.checkpoint)
    processed_dir = resolve_project_path(args.processed_dir)
    output_dir = resolve_project_path(args.output_dir)
    metric_dir = output_dir / "metrics"
    comparison_dir = output_dir / "outputs" / "fixed_visual_comparison"
    metric_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_comparison:
        comparison_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_manifest(manifest_path)
    categories = sorted({row["category"] for row in manifest_rows})

    device = select_device(args.device)
    model, checkpoint = load_baseline_model(checkpoint_path, device=device)
    input_mode = args.input_mode or checkpoint.get("input_mode", "rgb")
    mask_background = args.mask_background or checkpoint.get("mask_background", "white")
    dataset = ProcessedPix3DDataset(
        processed_dir=processed_dir,
        split=args.split,
        categories=categories,
        max_samples=None,
        expected_num_points=int(checkpoint.get("num_points", 2048)),
        input_mode=input_mode,
        mask_background=mask_background,
    )

    output_rows: list[dict[str, object]] = []
    for manifest_row in manifest_rows:
        benchmark_index = int(manifest_row["benchmark_index"])
        dataset_index = int(manifest_row["dataset_index"])
        if dataset_index < 0 or dataset_index >= len(dataset):
            raise IndexError(f"dataset_index={dataset_index} is outside dataset length {len(dataset)}")

        sample = dataset[dataset_index]
        sample_id = str(sample.get("sample_id", f"{args.split}_{dataset_index:05d}"))
        if sample_id != manifest_row["sample_id"]:
            raise RuntimeError(
                "Fixed benchmark manifest no longer matches the processed split: "
                f"index={dataset_index} expected={manifest_row['sample_id']} actual={sample_id}"
            )

        image = sample["image"].unsqueeze(0).to(device)
        gt_points = sample["points_gt"].unsqueeze(0).to(device)
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
        prefix = f"{benchmark_index:02d}_{sample_id}"

        if not args.skip_artifacts:
            np.save(comparison_dir / f"{prefix}_pred.npy", pred_np)
            np.save(comparison_dir / f"{prefix}_gt.npy", gt_np)

        if not args.skip_comparison:
            save_comparison_figure(
                pred_points=pred_np,
                gt_points=gt_np,
                output_path=comparison_dir / f"{prefix}_comparison.png",
                metrics=metrics,
                sample_id=sample_id,
                max_plot_points=args.max_plot_points,
                show=False,
            )

        output_row = {
            **manifest_row,
            "image_path": sample["image_path"],
            "pointcloud_path": sample["pointcloud_path"],
            **metrics,
        }
        output_rows.append(output_row)
        print(
            f"[{benchmark_index:02d}] {sample_id} "
            f"cd={metrics['chamfer_distance']:.6f} "
            f"f={metrics['f_score']:.4f} "
            f"fine={metrics['fine_f_score']:.4f} "
            f"occ={metrics['occupancy_iou']:.4f} "
            f"empty={metrics['empty_space_violation']:.4f} "
            f"vc={metrics['visual_completeness_score']:.4f}"
        )

    csv_path = metric_dir / "fixed_visual_benchmark.csv"
    fieldnames = [
        "benchmark_index",
        "dataset_index",
        "sample_id",
        "category",
        "processed_image",
        "pointcloud",
        "model_uid",
        "reason",
        "image_path",
        "pointcloud_path",
        *ALL_POINTCLOUD_METRICS,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "split": args.split,
        "samples": len(output_rows),
        "categories": categories,
        "manifest_path": str(manifest_path),
        "processed_dir": str(processed_dir),
        "checkpoint_path": str(checkpoint_path),
        "input_mode": input_mode,
        "mask_background": mask_background,
        "metrics_path": str(csv_path),
        "comparison_dir": str(comparison_dir) if not args.skip_comparison else None,
        "visual_diagnostics": {
            "fine_threshold": args.fine_threshold or args.f_threshold * 0.5,
            "loose_threshold": args.loose_threshold or args.f_threshold * 2.0,
            "voxel_resolution": args.voxel_resolution,
            "occupancy_dilation": args.occupancy_dilation,
            "density_sample_size": args.density_sample_size,
        },
        **mean_metrics(output_rows),
    }
    summary_path = metric_dir / "fixed_visual_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved fixed benchmark CSV: {csv_path}")
    print(f"Saved fixed benchmark summary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fixed visual benchmark sample list.")
    parser.add_argument("--manifest", default="benchmarks/fixed_test_samples_chair.csv")
    parser.add_argument("--processed-dir", default="data/processed_2048")
    parser.add_argument(
        "--checkpoint",
        default="results/all_categories_resnet50_2048pts_30ep_aug/outputs/checkpoints/best_model.pt",
    )
    parser.add_argument("--output-dir", default="results/fixed_visual_benchmark")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--input-mode", choices=["rgb", "masked_rgb"], default=None)
    parser.add_argument("--mask-background", choices=["white", "black"], default=None)
    parser.add_argument("--f-threshold", type=float, default=0.05)
    parser.add_argument("--fine-threshold", type=float, default=None)
    parser.add_argument("--loose-threshold", type=float, default=None)
    parser.add_argument("--density-sample-size", type=int, default=512)
    parser.add_argument("--voxel-resolution", type=int, default=32)
    parser.add_argument("--occupancy-dilation", type=int, default=1)
    parser.add_argument("--max-plot-points", type=int, default=2048)
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    evaluate_fixed_visual_benchmark(parse_args())


if __name__ == "__main__":
    main()
