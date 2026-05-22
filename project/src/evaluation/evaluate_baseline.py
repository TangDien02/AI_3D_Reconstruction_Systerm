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
from src.metrics.losses import chamfer_distance, f_score


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
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    rows = []
    total_cd = 0.0
    total_f = 0.0
    total_precision = 0.0
    total_recall = 0.0
    sample_count = 0

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device)
        points_gt = batch["points_gt"].to(device)
        points_pred = model_points(model, images)

        batch_cd = chamfer_distance(points_pred, points_gt).item()
        batch_f, batch_precision, batch_recall = f_score(
            points_pred,
            points_gt,
            threshold=args.f_threshold,
        )
        current_batch_size = images.shape[0]
        sample_count += current_batch_size
        total_cd += batch_cd * current_batch_size
        total_f += batch_f * current_batch_size
        total_precision += batch_precision * current_batch_size
        total_recall += batch_recall * current_batch_size

        rows.append(
            {
                "batch": batch_index,
                "batch_size": current_batch_size,
                "chamfer_distance": batch_cd,
                "f_score": batch_f,
                "precision": batch_precision,
                "recall": batch_recall,
            }
        )
        print(
            f"batch={batch_index} size={current_batch_size} "
            f"cd={batch_cd:.6f} f={batch_f:.4f}"
        )

    if sample_count == 0:
        raise RuntimeError("Evaluation dataset is empty.")

    batch_metrics_path = metric_dir / f"{args.split}_batch_metrics.csv"
    with batch_metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["batch", "batch_size", "chamfer_distance", "f_score", "precision", "recall"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "categories": args.categories or checkpoint.get("categories"),
        "samples": sample_count,
        "batch_size": args.batch_size,
        "checkpoint_path": str(checkpoint_path),
        "chamfer_distance": total_cd / sample_count,
        "f_score": total_f / sample_count,
        "precision": total_precision / sample_count,
        "recall": total_recall / sample_count,
        "batch_metrics_path": str(batch_metrics_path),
    }
    summary_path = metric_dir / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved evaluation summary to {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the baseline checkpoint on a processed split.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--checkpoint",
        default="results/chair_resnet_baseline/outputs/checkpoints/best_model.pt",
    )
    parser.add_argument("--output-dir", default="results/chair_resnet_baseline")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--f-threshold", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    evaluate_checkpoint(parse_args())


if __name__ == "__main__":
    main()

