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
# 1. Tao dataset tu raw Pix3D hoac data/processed.
# 2. Voi data processed, dung truc tiep splits/train.csv va splits/val.csv.
#    Voi data raw, fallback ve random split train/validation.
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
    parser.add_argument("--output-dir", default="results/chair_baseline")
    parser.add_argument("--dataset-mode", choices=["raw", "processed"], default="processed")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Processed split used for training.",
    )
    parser.add_argument(
        "--val-split",
        default="val",
        choices=["train", "val", "test"],
        help="Processed split used for validation.",
    )
    parser.add_argument("--categories", nargs="+", default=["chair"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--val-max-samples",
        type=int,
        default=None,
        help="Limit validation samples. Defaults to --max-samples for processed smoke tests.",
    )
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
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. Use cpu when the local GPU/PyTorch build is not compatible.",
    )
    parser.add_argument(
        "--best-metric",
        choices=["val_chamfer_distance", "val_f_score"],
        default="val_chamfer_distance",
        help="Validation metric used to save checkpoints/best_model.pt.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help=(
            "Checkpoint to resume from. Defaults to outputs/checkpoints/best_model.pt "
            "inside --output-dir when it exists."
        ),
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start from a fresh model even when best_model.pt already exists.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Do not run test evaluation after training.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Do not generate predicted-vs-ground-truth point cloud comparison after training.",
    )
    parser.add_argument("--post-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--comparison-index", type=int, default=0)
    parser.add_argument("--max-plot-points", type=int, default=2048)
    parser.set_defaults(resume=True)
    return parser.parse_args()


def build_checkpoint(
    model,
    args,
    epoch,
    train_loss=None,
    val_metrics=None,
    best_metric=None,
    best_score=None,
    best_epoch=None,
    optimizer=None,
    resumed_from_checkpoint=None,
):
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
        "best_epoch": best_epoch,
        "learning_rate": args.lr,
        "dataset_mode": args.dataset_mode,
        "split": args.split,
        "val_split": getattr(args, "val_split", None),
        "max_samples": args.max_samples,
        "val_max_samples": getattr(args, "val_max_samples", None),
        "resumed_from_checkpoint": str(resumed_from_checkpoint) if resumed_from_checkpoint else None,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
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


def resolve_resume_checkpoint(args, checkpoint_dir: Path) -> Path | None:
    if not getattr(args, "resume", True):
        return None

    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    if resume_checkpoint:
        checkpoint_path = Path(resume_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_DIR / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    default_checkpoint = checkpoint_dir / "best_model.pt"
    if default_checkpoint.is_file():
        return default_checkpoint
    return None


def validate_resume_checkpoint(checkpoint: dict, args, checkpoint_path: Path) -> None:
    def normalize_categories(categories):
        if categories is None:
            return None
        if isinstance(categories, str):
            return {categories}
        return set(categories)

    architecture_keys = [
        "num_points",
        "image_size",
        "patch_size",
        "embed_dim",
        "transformer_depth",
        "num_heads",
    ]
    mismatches = []
    for key in architecture_keys:
        if key in checkpoint and int(checkpoint[key]) != int(getattr(args, key)):
            mismatches.append(f"{key}: checkpoint={checkpoint[key]} current={getattr(args, key)}")

    checkpoint_categories = checkpoint.get("categories")
    if checkpoint_categories is not None and args.categories is not None:
        if normalize_categories(checkpoint_categories) != normalize_categories(args.categories):
            mismatches.append(
                f"categories: checkpoint={checkpoint_categories} current={args.categories}"
            )

    if mismatches:
        raise RuntimeError(
            "Cannot resume because checkpoint config does not match the current training config: "
            + "; ".join(mismatches)
            + f". Use a different --output-dir or pass --no-resume. Checkpoint: {checkpoint_path}"
        )


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def read_last_metrics_epoch(metrics_path: Path) -> int:
    if not metrics_path.is_file():
        return 0

    last_epoch = 0
    with metrics_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                last_epoch = max(last_epoch, int(row["epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
    return last_epoch


def select_training_device(device_name: str | None = None) -> torch.device:
    requested = (device_name or "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in {"auto", "cuda"}:
        raise ValueError(f"Unsupported device: {device_name}")

    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cpu")

    try:
        if torch.cuda.device_count() <= 0:
            raise RuntimeError("torch.cuda.device_count() is 0.")
        torch.empty(1, device="cuda")
        return torch.device("cuda")
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError(f"CUDA was requested, but it is not usable: {exc}") from exc
        print(f"CUDA is not usable, falling back to CPU: {exc}")
        return torch.device("cpu")


def save_metric_line_chart(
    csv_path: Path,
    output_path: Path,
    x_column: str,
    metric_columns: list[str],
    title: str,
) -> Path | None:
    if not csv_path.is_file():
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    if not rows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [float(row[x_column]) for row in rows] if x_column in rows[0] else list(range(1, len(rows) + 1))

    plt.figure(figsize=(9, 4.8))
    for column in metric_columns:
        if column not in rows[0]:
            continue
        y_values = [float(row[column]) for row in rows]
        plt.plot(x_values, y_values, marker="o", label=column)
    plt.title(title)
    plt.xlabel(x_column if x_column in rows[0] else "step")
    plt.ylabel("Metric value")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_summary_bar_chart(summary_path: Path, output_path: Path) -> Path | None:
    if not summary_path.is_file():
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metric_names = ["chamfer_distance", "f_score", "precision", "recall"]
    metrics = {
        name: float(summary[name])
        for name in metric_names
        if name in summary and isinstance(summary[name], (int, float))
    }
    if not metrics:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.4))
    plt.bar(metrics.keys(), metrics.values(), color=["#2563eb", "#2f6f5e", "#f59e0b", "#dc2626"][: len(metrics)])
    plt.title("Test summary metrics")
    plt.ylabel("Metric value")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_post_training_metric_charts(metric_dir: Path, artifact_dir: Path) -> dict[str, Path | None]:
    return {
        "test_batch_metrics_chart": save_metric_line_chart(
            metric_dir / "test_batch_metrics.csv",
            artifact_dir / "test_batch_metrics.png",
            x_column="batch",
            metric_columns=["chamfer_distance", "f_score", "precision", "recall"],
            title="Test batch metrics",
        ),
        "test_summary_metrics_chart": save_summary_bar_chart(
            metric_dir / "test_summary.json",
            artifact_dir / "test_summary_metrics.png",
        ),
    }


def run_post_training_outputs(
    args,
    checkpoint_path: Path,
    output_dir: Path,
    metric_dir: Path,
    artifact_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, object]:
    post_outputs: dict[str, object] = {}
    if args.dataset_mode != "processed":
        logger.info("Skipping post-training evaluation/comparison because dataset_mode is not processed.")
        return post_outputs

    if not getattr(args, "skip_evaluation", False):
        from types import SimpleNamespace
        from src.evaluation.evaluate_baseline import evaluate_checkpoint

        logger.info("Post-training evaluation on %s split.", args.post_split)
        evaluation_summary = evaluate_checkpoint(
            SimpleNamespace(
                processed_dir=args.processed_dir,
                checkpoint=str(checkpoint_path),
                output_dir=args.output_dir,
                split=args.post_split,
                categories=args.categories,
                max_samples=getattr(args, "eval_max_samples", None),
                batch_size=args.batch_size,
                f_threshold=args.f_threshold,
                device=str(device),
            )
        )
        post_outputs["evaluation_summary"] = evaluation_summary
        post_outputs.update(save_post_training_metric_charts(metric_dir, artifact_dir))

    if not getattr(args, "skip_comparison", False):
        from types import SimpleNamespace
        from src.inference.compare_pointclouds import compare_sample

        comparison_dir = artifact_dir / "comparison"
        logger.info("Post-training point cloud comparison on %s split.", args.post_split)
        compare_sample(
            SimpleNamespace(
                checkpoint=str(checkpoint_path),
                processed_dir=args.processed_dir,
                output_dir=str(comparison_dir),
                split=args.post_split,
                categories=args.categories,
                index=getattr(args, "comparison_index", 0),
                max_samples=None,
                f_threshold=args.f_threshold,
                max_plot_points=getattr(args, "max_plot_points", 2048),
                device=str(device),
                show=False,
            )
        )
        post_outputs["comparison_dir"] = comparison_dir

    return post_outputs


def run_training(args):
    if not hasattr(args, "best_metric"):
        args.best_metric = "val_chamfer_distance"
    if not hasattr(args, "resume"):
        args.resume = True
    if not hasattr(args, "resume_checkpoint"):
        args.resume_checkpoint = None
    if not hasattr(args, "device"):
        args.device = "auto"
    if not hasattr(args, "skip_evaluation"):
        args.skip_evaluation = False
    if not hasattr(args, "skip_comparison"):
        args.skip_comparison = False
    if not hasattr(args, "post_split"):
        args.post_split = "test"
    if not hasattr(args, "eval_max_samples"):
        args.eval_max_samples = None
    if not hasattr(args, "val_split"):
        args.val_split = "val"
    if not hasattr(args, "val_max_samples"):
        args.val_max_samples = None
    if not hasattr(args, "comparison_index"):
        args.comparison_index = 0
    if not hasattr(args, "max_plot_points"):
        args.max_plot_points = 2048

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

    device = select_training_device(args.device)
    logger.info("Starting baseline training on device=%s", device)
    logger.info("Output directory: %s", output_dir)
    if args.dataset_mode == "processed":
        train_dataset = ProcessedPix3DDataset(
            processed_dir=processed_dir,
            split=args.split,
            categories=args.categories,
            max_samples=args.max_samples,
        )
        validation_max_samples = args.val_max_samples
        if validation_max_samples is None:
            validation_max_samples = args.max_samples
        val_dataset = ProcessedPix3DDataset(
            processed_dir=processed_dir,
            split=args.val_split,
            categories=args.categories,
            max_samples=validation_max_samples,
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

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    if train_size < 1:
        raise RuntimeError(f"Training split is empty: {args.split}")
    if val_size < 1:
        raise RuntimeError(f"Validation split is empty: {args.val_split}")

    logger.info(
        "Dataset ready: mode=%s train_split=%s val_split=%s categories=%s train_samples=%s val_samples=%s",
        args.dataset_mode,
        args.split,
        args.val_split if args.dataset_mode == "processed" else "random",
        ",".join(args.categories),
        train_size,
        val_size,
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
    resume_checkpoint_path = resolve_resume_checkpoint(args, checkpoint_dir)
    best_score = None
    best_epoch = None
    resumed_from_epoch = None
    optimizer_resumed = False

    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
        validate_resume_checkpoint(checkpoint, args, resume_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
            optimizer_resumed = True

        best_score = checkpoint.get("best_score")
        if best_score is None and args.best_metric in checkpoint:
            best_score = checkpoint[args.best_metric]
        best_epoch = checkpoint.get("best_epoch") or checkpoint.get("epoch")
        resumed_from_epoch = checkpoint.get("epoch")

        message = (
            f"Resuming from {resume_checkpoint_path} "
            f"(checkpoint_epoch={resumed_from_epoch}, best_{args.best_metric}={best_score})"
        )
        print(message)
        logger.info(message)
        if not optimizer_resumed:
            message = "Resume checkpoint has no optimizer_state_dict; continuing with a fresh Adam optimizer."
            print(message)
            logger.info(message)

    last_metrics_epoch = read_last_metrics_epoch(metrics_path) if resume_checkpoint_path else 0
    checkpoint_epoch = int(resumed_from_epoch or 0)
    start_epoch = max(last_metrics_epoch, checkpoint_epoch) + 1
    end_epoch = start_epoch + args.epochs - 1

    metrics_mode = "a" if resume_checkpoint_path is not None and metrics_path.is_file() else "w"
    with metrics_path.open(metrics_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_chamfer_distance", "val_f_score"])
        if metrics_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, end_epoch + 1):
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
                        best_epoch=best_epoch,
                        optimizer=optimizer,
                        resumed_from_checkpoint=resume_checkpoint_path,
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
            epoch=end_epoch,
            best_metric=args.best_metric,
            best_score=best_score,
            best_epoch=best_epoch,
            optimizer=optimizer,
            resumed_from_checkpoint=resume_checkpoint_path,
        ),
        checkpoint_path,
    )
    plot_path = save_training_curves(metrics_path, artifact_dir / "training_curves.png")
    best_or_last_checkpoint_path = best_checkpoint_path if best_checkpoint_path.is_file() else checkpoint_path
    post_outputs = run_post_training_outputs(
        args=args,
        checkpoint_path=best_or_last_checkpoint_path,
        output_dir=output_dir,
        metric_dir=metric_dir,
        artifact_dir=artifact_dir,
        device=device,
        logger=logger,
    )
    summary_path = artifact_dir / "baseline_summary.json"
    summary = {
        "dataset_mode": args.dataset_mode,
        "split": args.split,
        "val_split": args.val_split,
        "categories": args.categories,
        "max_samples": args.max_samples,
        "val_max_samples": args.val_max_samples,
        "train_samples": train_size,
        "val_samples": val_size,
        "num_points": args.num_points,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "learning_rate": args.lr,
        "device": str(device),
        "resume_enabled": getattr(args, "resume", True),
        "resumed_from_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path else None,
        "resumed_from_epoch": resumed_from_epoch,
        "optimizer_resumed": optimizer_resumed,
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_metric": args.best_metric,
        "best_score": best_score,
        "training_curves_path": str(plot_path) if plot_path else None,
        "test_batch_metrics_chart_path": str(post_outputs.get("test_batch_metrics_chart"))
        if post_outputs.get("test_batch_metrics_chart")
        else None,
        "test_summary_metrics_chart_path": str(post_outputs.get("test_summary_metrics_chart"))
        if post_outputs.get("test_summary_metrics_chart")
        else None,
        "comparison_dir": str(post_outputs.get("comparison_dir")) if post_outputs.get("comparison_dir") else None,
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
        "post_outputs": post_outputs,
        "resumed_from_checkpoint": resume_checkpoint_path,
        "resumed_from_epoch": resumed_from_epoch,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "optimizer_resumed": optimizer_resumed,
    }


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
