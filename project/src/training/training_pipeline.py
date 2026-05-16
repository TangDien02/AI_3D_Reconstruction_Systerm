from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

# Ghi chu:
# File nay la training pipeline cho bai toan single-view 3D reconstruction.
# Dau vao la anh 2D da duoc dataloader xu ly, dau ra la point cloud 3D du doan.
#
# Luong xu ly:
# 1. Tao Pix3DDataset tu project/data/raw/pix3d.json.
# 2. Chia du lieu thanh train/validation.
# 3. Dung DataLoader tao batch anh va point cloud ground truth.
# 4. Dua anh vao TransformerPointCloudNet de du doan point cloud.
# 5. Tinh Chamfer Distance giua point cloud du doan va ground truth.
# 6. Backpropagation va cap nhat trong so model bang Adam.
# 7. Danh gia bang Chamfer Distance va F-score, sau do luu metric/checkpoint.
#
# Hien tai day la skeleton de chuan bi cho tuan 3, chua phai model toi uu.

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataloader import Pix3DDataset, ProcessedPix3DDataset
from src.metrics.losses import chamfer_distance, f_score
from src.models.transformer_pointcloud import TransformerPointCloudNet


def setup_baseline_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("BaselineTraining")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = log_dir / "baseline.log"
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(type(handler) is logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def save_training_curves(metrics_path: Path, output_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    rows = []
    with metrics_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_chamfer_distance": float(row["val_chamfer_distance"]),
                    "val_f_score": float(row["val_f_score"]),
                }
            )

    if not rows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o", label="Train loss")
    axes[0].plot(
        epochs,
        [row["val_chamfer_distance"] for row in rows],
        marker="o",
        label="Val Chamfer",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss / Chamfer")
    axes[0].legend()

    axes[1].plot(epochs, [row["val_f_score"] for row in rows], marker="o", color="#2f6f5e")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val F-score")

    fig.suptitle("Baseline training metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        points_gt = batch["points_gt"].to(device)

        points_pred = model(images)
        loss = chamfer_distance(points_pred, points_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device, threshold):
    model.eval()
    total_cd = 0.0
    total_f = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        points_gt = batch["points_gt"].to(device)

        points_pred = model(images)
        total_cd += chamfer_distance(points_pred, points_gt).item()
        total_f += f_score(points_pred, points_gt, threshold=threshold)[0]

    num_batches = max(len(dataloader), 1)
    return {
        "chamfer_distance": total_cd / num_batches,
        "f_score": total_f / num_batches,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train single-view 3D point cloud baseline")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--dataset-mode", choices=["raw", "processed"], default="processed")
    parser.add_argument("--split", default="train")
    parser.add_argument("--categories", nargs="+", default=["chair"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--f-threshold", type=float, default=0.05)
    parser.add_argument(
        "--best-metric",
        choices=["val_chamfer_distance", "val_f_score"],
        default="val_chamfer_distance",
        help="Validation metric used to save checkpoints/best_model.pt.",
    )
    return parser.parse_args()


def build_checkpoint(model, args, epoch, train_loss=None, val_metrics=None, best_metric=None, best_score=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "categories": args.categories,
        "num_points": args.num_points,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "transformer_depth": args.transformer_depth,
        "num_heads": args.num_heads,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_score": best_score,
    }
    if train_loss is not None:
        checkpoint["train_loss"] = train_loss
    if val_metrics is not None:
        checkpoint["val_chamfer_distance"] = val_metrics["chamfer_distance"]
        checkpoint["val_f_score"] = val_metrics["f_score"]
    return checkpoint


def is_better_score(metric_name, score, best_score):
    if best_score is None:
        return True
    if metric_name == "val_chamfer_distance":
        return score < best_score
    return score > best_score


def run_training(args):
    raw_dir = (PROJECT_DIR / args.raw_dir).resolve()
    processed_dir = (PROJECT_DIR / args.processed_dir).resolve()
    output_dir = (PROJECT_DIR / args.output_dir).resolve()
    checkpoint_dir = output_dir / "outputs" / "checkpoints"
    metric_dir = output_dir / "metrics"
    log_dir = output_dir / "logs"
    artifact_dir = output_dir / "outputs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_baseline_logger(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting baseline training on device=%s", device)
    logger.info("Output directory: %s", output_dir)
    if args.dataset_mode == "processed":
        dataset = ProcessedPix3DDataset(
            processed_dir=processed_dir,
            split=args.split,
            categories=args.categories,
            max_samples=args.max_samples,
        )
    else:
        dataset = Pix3DDataset(
            root_dir=raw_dir,
            categories=args.categories,
            image_size=args.image_size,
            num_points=args.num_points,
            max_samples=args.max_samples,
        )

    if len(dataset) < 2:
        raise RuntimeError("Dataset needs at least 2 samples for train/validation split.")
    logger.info(
        "Dataset ready: mode=%s split=%s categories=%s samples=%s",
        args.dataset_mode,
        args.split,
        ",".join(args.categories),
        len(dataset),
    )

    train_size = max(1, int(len(dataset) * 0.8))
    val_size = len(dataset) - train_size
    if val_size == 0:
        train_size -= 1
        val_size = 1

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info("Train samples: %s | Val samples: %s", train_size, val_size)

    model = TransformerPointCloudNet(
        num_points=args.num_points,
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.transformer_depth,
        num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = metric_dir / "training_metrics.csv"
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    best_score = None
    best_epoch = None
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_chamfer_distance", "val_f_score"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device, threshold=args.f_threshold)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_chamfer_distance": val_metrics["chamfer_distance"],
                "val_f_score": val_metrics["f_score"],
            }
            writer.writerow(row)

            current_score = row[args.best_metric]
            if is_better_score(args.best_metric, current_score, best_score):
                best_score = current_score
                best_epoch = epoch
                torch.save(
                    build_checkpoint(
                        model=model,
                        args=args,
                        epoch=epoch,
                        train_loss=train_loss,
                        val_metrics=val_metrics,
                        best_metric=args.best_metric,
                        best_score=best_score,
                    ),
                    best_checkpoint_path,
                )
                best_text = f" best_{args.best_metric}={best_score:.6f}"
            else:
                best_text = ""

            message = (
                f"epoch={epoch} "
                f"train_loss={train_loss:.6f} "
                f"val_cd={val_metrics['chamfer_distance']:.6f} "
                f"val_f={val_metrics['f_score']:.4f}"
                f"{best_text}"
            )
            print(message)
            logger.info(message)

    checkpoint_path = checkpoint_dir / "transformer_pointcloud_net.pt"
    torch.save(
        build_checkpoint(
            model=model,
            args=args,
            epoch=args.epochs,
            best_metric=args.best_metric,
            best_score=best_score,
        ),
        checkpoint_path,
    )
    plot_path = save_training_curves(metrics_path, artifact_dir / "training_curves.png")
    summary_path = artifact_dir / "baseline_summary.json"
    summary = {
        "dataset_mode": args.dataset_mode,
        "split": args.split,
        "categories": args.categories,
        "max_samples": args.max_samples,
        "num_points": args.num_points,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_metric": args.best_metric,
        "best_score": best_score,
        "training_curves_path": str(plot_path) if plot_path else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved checkpoint to {checkpoint_path}")
    if best_score is not None:
        print(f"Saved best checkpoint to {best_checkpoint_path} (epoch={best_epoch}, {args.best_metric}={best_score:.6f})")
    print(f"Saved summary to {summary_path}")
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)
    if best_score is not None:
        logger.info(
            "Saved best checkpoint to %s (epoch=%s, %s=%.6f)",
            best_checkpoint_path,
            best_epoch,
            args.best_metric,
            best_score,
        )
    logger.info("Saved summary to %s", summary_path)
    if plot_path:
        logger.info("Saved training curves to %s", plot_path)
    return {
        "metrics_path": metrics_path,
        "checkpoint_path": checkpoint_path,
        "best_checkpoint_path": best_checkpoint_path,
        "summary_path": summary_path,
        "plot_path": plot_path,
    }


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
