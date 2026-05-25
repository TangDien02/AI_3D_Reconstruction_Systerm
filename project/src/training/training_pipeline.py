from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Ghi chu:
# File nay la training pipeline cho bai toan single-view 3D reconstruction.
# Dau vao la anh 2D da duoc dataloader xu ly, dau ra la point cloud 3D du doan.
#
# Luong xu ly:
# 1. Tao dataset tu raw Pix3D hoac data/processed.
# 2. Voi data processed, dung truc tiep splits/train.csv va splits/val.csv.
#    Voi data raw, fallback ve random split train/validation.
# 3. Dung DataLoader tao batch anh va point cloud ground truth.
# 4. Dua anh vao ResNet encoder + point cloud decoder de du doan point cloud.
# 5. Tinh Chamfer Distance giua point cloud du doan va ground truth.
# 6. Backpropagation va cap nhat trong so model bang Adam.
# 7. Danh gia bang Chamfer Distance va F-score, sau do luu metric/checkpoint.
#
# Pipeline chinh dung object-level ResNet reconstruction baseline.

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataloader import Pix3DDataset, ProcessedPix3DDataset
from src.metrics.losses import (
    chamfer_distance,
    f_score,
    point_repulsion_loss,
    weighted_chamfer_distance,
)
from src.models.object_reconstruction import build_object_reconstruction_model


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


def model_points(model, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    return output.points if hasattr(output, "points") else output


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type)


def build_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


class AddGaussianNoise:
    def __init__(self, std: float = 0.0):
        self.std = max(0.0, float(std))

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    amp_enabled=False,
    grad_scaler=None,
    chamfer_gt_weight=1.0,
    repulsion_weight=0.0,
    repulsion_k=8,
    repulsion_radius=0.03,
    repulsion_sample_size=512,
):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        points_gt = batch["points_gt"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            points_pred = model_points(model, images)
            loss = weighted_chamfer_distance(
                points_pred,
                points_gt,
                gt_weight=chamfer_gt_weight,
            )
            if repulsion_weight > 0:
                repulsion = point_repulsion_loss(
                    points_pred,
                    k=repulsion_k,
                    radius=repulsion_radius,
                    sample_size=repulsion_sample_size,
                )
                loss = loss + repulsion_weight * repulsion

        if amp_enabled and grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device, threshold, amp_enabled=False):
    model.eval()
    total_cd = 0.0
    total_f = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        points_gt = batch["points_gt"].to(device)

        with autocast_context(device, amp_enabled):
            points_pred = model_points(model, images)
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
    parser.add_argument("--output-dir", default="results/chair_resnet_baseline")
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
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--encoder-name", choices=["conv", "resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=None)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision when training on CUDA. CPU runs keep FP32.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau"],
        default="plateau",
        help="Learning-rate scheduler. plateau uses ReduceLROnPlateau on the selected validation metric.",
    )
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.7)
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler-threshold", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--chamfer-gt-weight",
        type=float,
        default=1.25,
        help="Weight for the ground-truth-to-prediction Chamfer term. Higher values improve surface coverage.",
    )
    parser.add_argument(
        "--repulsion-weight",
        type=float,
        default=0.01,
        help="Weight for predicted point repulsion regularization.",
    )
    parser.add_argument("--repulsion-k", type=int, default=8)
    parser.add_argument("--repulsion-radius", type=float, default=0.03)
    parser.add_argument(
        "--repulsion-sample-size",
        type=int,
        default=512,
        help="Number of predicted points sampled for repulsion loss. Use 0 to use all points.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply train-only image augmentation for processed datasets.",
    )
    parser.add_argument("--augment-brightness", type=float, default=0.15)
    parser.add_argument("--augment-contrast", type=float, default=0.15)
    parser.add_argument("--augment-noise-std", type=float, default=0.01)
    parser.add_argument("--augment-erasing-prob", type=float, default=0.10)
    parser.add_argument("--augment-erasing-scale", type=float, default=0.12)
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=None,
        help="Absolute epoch at which to unfreeze the ResNet encoder. Omit to keep the initial freeze setting.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop after this many epochs without a meaningful validation improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=12,
        help="Minimum epochs to run in this invocation before early stopping can trigger.",
    )
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
    args = parser.parse_args()
    for field in (
        "max_samples",
        "val_max_samples",
        "eval_max_samples",
        "unfreeze_epoch",
    ):
        if getattr(args, field) is not None and getattr(args, field) < 0:
            setattr(args, field, None)
    for field in ("early_stopping_patience", "early_stopping_min_epochs"):
        if getattr(args, field) is not None and getattr(args, field) < 0:
            setattr(args, field, 0)
    if args.early_stopping_min_delta < 0:
        args.early_stopping_min_delta = 0.0
    if args.lr_scheduler_factor <= 0 or args.lr_scheduler_factor >= 1:
        args.lr_scheduler_factor = 0.7
    if args.lr_scheduler_patience < 0:
        args.lr_scheduler_patience = 0
    if args.lr_scheduler_threshold < 0:
        args.lr_scheduler_threshold = 0.0
    if args.lr_scheduler_min_lr < 0:
        args.lr_scheduler_min_lr = 0.0
    if args.chamfer_gt_weight <= 0:
        args.chamfer_gt_weight = 1.0
    if args.repulsion_weight < 0:
        args.repulsion_weight = 0.0
    if args.repulsion_k < 0:
        args.repulsion_k = 0
    if args.repulsion_radius < 0:
        args.repulsion_radius = 0.0
    if args.repulsion_sample_size < 0:
        args.repulsion_sample_size = 0
    return args


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
    scheduler=None,
    grad_scaler=None,
    resumed_from_checkpoint=None,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "resnet_pointcloud",
        "categories": args.categories,
        "num_points": args.num_points,
        "image_size": args.image_size,
        "encoder_name": args.encoder_name,
        "feature_dim": args.feature_dim,
        "pretrained": args.pretrained,
        "freeze_encoder": args.freeze_encoder,
        "encoder_unfrozen": getattr(args, "encoder_unfrozen", None),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "learning_rate": args.lr,
        "decoder_lr": getattr(args, "decoder_lr", None),
        "encoder_lr": getattr(args, "encoder_lr", None),
        "weight_decay": getattr(args, "weight_decay", 0.0),
        "amp": getattr(args, "amp", True),
        "lr_scheduler": getattr(args, "lr_scheduler", "plateau"),
        "lr_scheduler_factor": getattr(args, "lr_scheduler_factor", 0.7),
        "lr_scheduler_patience": getattr(args, "lr_scheduler_patience", 5),
        "lr_scheduler_threshold": getattr(args, "lr_scheduler_threshold", 1e-4),
        "lr_scheduler_min_lr": getattr(args, "lr_scheduler_min_lr", 1e-6),
        "chamfer_gt_weight": getattr(args, "chamfer_gt_weight", 1.25),
        "repulsion_weight": getattr(args, "repulsion_weight", 0.01),
        "repulsion_k": getattr(args, "repulsion_k", 8),
        "repulsion_radius": getattr(args, "repulsion_radius", 0.03),
        "repulsion_sample_size": getattr(args, "repulsion_sample_size", 512),
        "unfreeze_epoch": getattr(args, "unfreeze_epoch", None),
        "early_stopping_patience": getattr(args, "early_stopping_patience", 0),
        "early_stopping_min_delta": getattr(args, "early_stopping_min_delta", 0.0),
        "early_stopping_min_epochs": getattr(args, "early_stopping_min_epochs", 0),
        "dataset_mode": args.dataset_mode,
        "split": args.split,
        "val_split": getattr(args, "val_split", None),
        "max_samples": args.max_samples,
        "val_max_samples": getattr(args, "val_max_samples", None),
        "resumed_from_checkpoint": str(resumed_from_checkpoint) if resumed_from_checkpoint else None,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["lr_scheduler_state_dict"] = scheduler.state_dict()
    if grad_scaler is not None:
        checkpoint["grad_scaler_state_dict"] = grad_scaler.state_dict()
    if train_loss is not None:
        checkpoint["train_loss"] = train_loss
    if val_metrics is not None:
        checkpoint["val_chamfer_distance"] = val_metrics["chamfer_distance"]
        checkpoint["val_f_score"] = val_metrics["f_score"]
    return checkpoint


def is_better_score(metric_name, score, best_score, min_delta: float = 0.0):
    if best_score is None:
        return True
    if metric_name == "val_chamfer_distance":
        return score < best_score - min_delta
    return score > best_score + min_delta


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

    mismatches = []
    checkpoint_model_type = checkpoint.get("model_type")
    if checkpoint_model_type is not None and checkpoint_model_type != "resnet_pointcloud":
        mismatches.append(f"model_type: checkpoint={checkpoint_model_type} current=resnet_pointcloud")
    if checkpoint_model_type is None and any(
        key in checkpoint for key in ("patch_size", "embed_dim", "transformer_depth", "num_heads")
    ):
        mismatches.append("model_type: checkpoint appears to be an old Transformer checkpoint")

    for key in ["num_points", "image_size", "feature_dim"]:
        if key in checkpoint and int(checkpoint[key]) != int(getattr(args, key)):
            mismatches.append(f"{key}: checkpoint={checkpoint[key]} current={getattr(args, key)}")

    for key in ["encoder_name", "pretrained", "freeze_encoder"]:
        if key in checkpoint and checkpoint[key] != getattr(args, key):
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


def validate_processed_splits(processed_dir: Path, train_split: str, val_split: str) -> dict[str, int]:
    import pandas as pd

    split_dir = processed_dir / "splits"
    split_frames = {}
    required_columns = {"model_uid", "pointcloud", "processed_image", "category"}
    for split_name in {train_split, val_split, "test"}:
        split_path = split_dir / f"{split_name}.csv"
        if not split_path.is_file():
            continue
        frame = pd.read_csv(split_path)
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise RuntimeError(f"{split_path} is missing required columns: {sorted(missing_columns)}")
        split_frames[split_name] = frame

    if train_split in split_frames and val_split in split_frames:
        train_models = set(split_frames[train_split]["model_uid"].astype(str))
        val_models = set(split_frames[val_split]["model_uid"].astype(str))
        overlap = train_models & val_models
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise RuntimeError(f"CAD model leakage between {train_split} and {val_split}: {examples}")

    if "test" in split_frames:
        for split_name, frame in split_frames.items():
            if split_name == "test":
                continue
            overlap = set(frame["model_uid"].astype(str)) & set(split_frames["test"]["model_uid"].astype(str))
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise RuntimeError(f"CAD model leakage between {split_name} and test: {examples}")

    counts = {}
    for split_name, frame in split_frames.items():
        counts[f"{split_name}_samples"] = len(frame)
        counts[f"{split_name}_models"] = frame["model_uid"].nunique()
    return counts


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_optimizer(model, args) -> torch.optim.Optimizer:
    decoder_lr = getattr(args, "decoder_lr", None) or args.lr
    encoder_lr = getattr(args, "encoder_lr", None) or decoder_lr
    weight_decay = getattr(args, "weight_decay", 0.0)

    encoder_param_ids = {id(param) for param in model.encoder.parameters()}
    encoder_params = [
        param for param in model.encoder.parameters()
        if param.requires_grad
    ]
    decoder_params = [
        param for param in model.parameters()
        if param.requires_grad and id(param) not in encoder_param_ids
    ]

    param_groups = []
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": decoder_lr, "name": "decoder"})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr, "name": "encoder"})
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, args):
    if getattr(args, "lr_scheduler", "plateau") == "none":
        return None

    mode = "min" if args.best_metric == "val_chamfer_distance" else "max"
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        threshold=args.lr_scheduler_threshold,
        threshold_mode="abs",
        min_lr=args.lr_scheduler_min_lr,
    )


def optimizer_lr_summary(optimizer: torch.optim.Optimizer) -> str:
    parts = []
    for index, group in enumerate(optimizer.param_groups):
        name = group.get("name") or f"group_{index}"
        parts.append(f"{name}={group['lr']:.2e}")
    return ",".join(parts)


def optimizer_lr_values(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    values = {}
    for index, group in enumerate(optimizer.param_groups):
        name = group.get("name") or f"group_{index}"
        values[name] = float(group["lr"])
    return values


def step_lr_scheduler(scheduler, optimizer: torch.optim.Optimizer, metric: float) -> bool:
    if scheduler is None:
        return False

    before = [float(group["lr"]) for group in optimizer.param_groups]
    scheduler.step(metric)
    after = [float(group["lr"]) for group in optimizer.param_groups]
    return any(new_lr < old_lr for old_lr, new_lr in zip(before, after))


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
    if not hasattr(args, "encoder_name"):
        args.encoder_name = "resnet18"
    if not hasattr(args, "feature_dim"):
        args.feature_dim = 512
    if not hasattr(args, "pretrained"):
        args.pretrained = True
    if not hasattr(args, "freeze_encoder"):
        args.freeze_encoder = True
    if not hasattr(args, "decoder_lr"):
        args.decoder_lr = None
    if not hasattr(args, "encoder_lr"):
        args.encoder_lr = 1e-5
    if not hasattr(args, "weight_decay"):
        args.weight_decay = 1e-4
    if not hasattr(args, "amp"):
        args.amp = True
    if not hasattr(args, "lr_scheduler"):
        args.lr_scheduler = "plateau"
    if not hasattr(args, "lr_scheduler_factor"):
        args.lr_scheduler_factor = 0.7
    if not hasattr(args, "lr_scheduler_patience"):
        args.lr_scheduler_patience = 5
    if not hasattr(args, "lr_scheduler_threshold"):
        args.lr_scheduler_threshold = 1e-4
    if not hasattr(args, "lr_scheduler_min_lr"):
        args.lr_scheduler_min_lr = 1e-6
    if not hasattr(args, "chamfer_gt_weight"):
        args.chamfer_gt_weight = 1.25
    if not hasattr(args, "repulsion_weight"):
        args.repulsion_weight = 0.01
    if not hasattr(args, "repulsion_k"):
        args.repulsion_k = 8
    if not hasattr(args, "repulsion_radius"):
        args.repulsion_radius = 0.03
    if not hasattr(args, "repulsion_sample_size"):
        args.repulsion_sample_size = 512
    if not hasattr(args, "unfreeze_epoch"):
        args.unfreeze_epoch = None
    if not hasattr(args, "augment"):
        args.augment = True
    if not hasattr(args, "augment_brightness"):
        args.augment_brightness = 0.15
    if not hasattr(args, "augment_contrast"):
        args.augment_contrast = 0.15
    if not hasattr(args, "augment_noise_std"):
        args.augment_noise_std = 0.01
    if not hasattr(args, "augment_erasing_prob"):
        args.augment_erasing_prob = 0.10
    if not hasattr(args, "augment_erasing_scale"):
        args.augment_erasing_scale = 0.12
    if not hasattr(args, "early_stopping_patience"):
        args.early_stopping_patience = 8
    if not hasattr(args, "early_stopping_min_delta"):
        args.early_stopping_min_delta = 1e-4
    if not hasattr(args, "early_stopping_min_epochs"):
        args.early_stopping_min_epochs = 12

    args.early_stopping_patience = max(0, int(args.early_stopping_patience or 0))
    args.early_stopping_min_delta = max(0.0, float(args.early_stopping_min_delta or 0.0))
    args.early_stopping_min_epochs = max(0, int(args.early_stopping_min_epochs or 0))
    if args.lr_scheduler not in {"none", "plateau"}:
        args.lr_scheduler = "plateau"
    args.lr_scheduler_factor = float(args.lr_scheduler_factor or 0.7)
    if args.lr_scheduler_factor <= 0 or args.lr_scheduler_factor >= 1:
        args.lr_scheduler_factor = 0.7
    args.lr_scheduler_patience = max(0, int(args.lr_scheduler_patience or 0))
    args.lr_scheduler_threshold = max(0.0, float(args.lr_scheduler_threshold or 0.0))
    args.lr_scheduler_min_lr = max(0.0, float(args.lr_scheduler_min_lr or 0.0))
    args.chamfer_gt_weight = max(1e-8, float(args.chamfer_gt_weight or 1.0))
    args.repulsion_weight = max(0.0, float(args.repulsion_weight or 0.0))
    args.repulsion_k = max(0, int(args.repulsion_k or 0))
    args.repulsion_radius = max(0.0, float(args.repulsion_radius or 0.0))
    args.repulsion_sample_size = max(0, int(args.repulsion_sample_size or 0))

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
    split_validation = {}
    if args.dataset_mode == "processed":
        split_validation = validate_processed_splits(processed_dir, args.split, args.val_split)
        logger.info("Validated processed splits: %s", split_validation)

        train_transform = None
        if args.augment:
            erasing_scale_max = min(1.0, max(0.02, float(args.augment_erasing_scale)))
            transform_steps = [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=max(0.0, float(args.augment_brightness)),
                    contrast=max(0.0, float(args.augment_contrast)),
                    saturation=0.2,
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                transforms.ToTensor(),
                AddGaussianNoise(std=float(args.augment_noise_std)),
            ]
            if float(args.augment_erasing_prob) > 0:
                transform_steps.append(
                    transforms.RandomErasing(
                        p=min(1.0, max(0.0, float(args.augment_erasing_prob))),
                        scale=(0.02, erasing_scale_max),
                    )
                )
            train_transform = transforms.Compose(transform_steps)
            logger.info(
                "Train augmentation enabled: brightness=%s contrast=%s noise_std=%s erasing_prob=%s erasing_scale_max=%s",
                args.augment_brightness,
                args.augment_contrast,
                args.augment_noise_std,
                args.augment_erasing_prob,
                erasing_scale_max,
            )
        else:
            logger.info("Train augmentation disabled.")

        train_dataset = ProcessedPix3DDataset(
            processed_dir=processed_dir,
            split=args.split,
            categories=args.categories,
            max_samples=args.max_samples,
            expected_num_points=args.num_points,
            transform=train_transform,
        )
        validation_max_samples = args.val_max_samples
        if validation_max_samples is None:
            validation_max_samples = args.max_samples
        # Val dataset khong dung augmentation
        val_dataset = ProcessedPix3DDataset(
            processed_dir=processed_dir,
            split=args.val_split,
            categories=args.categories,
            max_samples=validation_max_samples,
            expected_num_points=args.num_points,
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
    if args.max_samples is not None or getattr(args, "val_max_samples", None) is not None:
        logger.warning(
            "Limited-sample training run: max_samples=%s val_max_samples=%s. Treat metrics as smoke-test only.",
            args.max_samples,
            getattr(args, "val_max_samples", None),
        )

    model = build_object_reconstruction_model(
        encoder_name=args.encoder_name,
        pretrained=args.pretrained,
        feature_dim=args.feature_dim,
        num_points=args.num_points,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    logger.info(
        "Model ready: resnet_pointcloud encoder=%s pretrained=%s freeze_encoder=%s feature_dim=%s num_points=%s trainable_params=%s",
        args.encoder_name,
        args.pretrained,
        args.freeze_encoder,
        args.feature_dim,
        args.num_points,
        model.trainable_parameter_count(),
    )
    encoder_unfrozen = not args.freeze_encoder
    args.encoder_unfrozen = encoder_unfrozen
    optimizer = build_optimizer(model, args)
    scheduler = build_lr_scheduler(optimizer, args)
    amp_enabled = bool(args.amp and device.type == "cuda")
    grad_scaler = build_grad_scaler(amp_enabled)
    scheduler_resumed = False
    grad_scaler_resumed = False

    if args.amp and not amp_enabled:
        logger.info("AMP requested but inactive because device=%s. Training uses FP32.", device)
    elif amp_enabled:
        logger.info("AMP mixed precision enabled for CUDA training.")
    if scheduler is not None:
        logger.info(
            "ReduceLROnPlateau enabled: metric=%s factor=%s patience=%s threshold=%s min_lr=%s",
            args.best_metric,
            args.lr_scheduler_factor,
            args.lr_scheduler_patience,
            args.lr_scheduler_threshold,
            args.lr_scheduler_min_lr,
        )
    else:
        logger.info("LR scheduler disabled.")
    logger.info(
        "Training loss: weighted_chamfer gt_weight=%s repulsion_weight=%s repulsion_k=%s repulsion_radius=%s repulsion_sample_size=%s",
        args.chamfer_gt_weight,
        args.repulsion_weight,
        args.repulsion_k,
        args.repulsion_radius,
        args.repulsion_sample_size,
    )

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

        if checkpoint.get("encoder_unfrozen") and not encoder_unfrozen:
            model.unfreeze_encoder()
            encoder_unfrozen = True
            args.encoder_unfrozen = True
            optimizer = build_optimizer(model, args)
            scheduler = build_lr_scheduler(optimizer, args)
            logger.info("Resume checkpoint was saved after encoder unfreeze; rebuilt optimizer param groups.")

        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                move_optimizer_state_to_device(optimizer, device)
                optimizer_resumed = True
            except ValueError as exc:
                message = f"Could not resume optimizer_state_dict ({exc}); continuing with a fresh AdamW optimizer."
                print(message)
                logger.warning(message)

        if scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                scheduler_resumed = True
            except ValueError as exc:
                message = f"Could not resume lr_scheduler_state_dict ({exc}); continuing with a fresh scheduler."
                print(message)
                logger.warning(message)

        if amp_enabled and "grad_scaler_state_dict" in checkpoint:
            try:
                grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
                grad_scaler_resumed = True
            except ValueError as exc:
                message = f"Could not resume grad_scaler_state_dict ({exc}); continuing with a fresh GradScaler."
                print(message)
                logger.warning(message)

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
            message = "Resume checkpoint has no optimizer_state_dict; continuing with a fresh AdamW optimizer."
            print(message)
            logger.info(message)

    last_metrics_epoch = read_last_metrics_epoch(metrics_path) if resume_checkpoint_path else 0
    checkpoint_epoch = int(resumed_from_epoch or 0)
    start_epoch = max(last_metrics_epoch, checkpoint_epoch) + 1
    planned_end_epoch = start_epoch + args.epochs - 1
    last_completed_epoch = start_epoch - 1
    early_stopped = False
    early_stop_epoch = None
    epochs_without_improvement = 0
    early_stopping_score = best_score

    if args.early_stopping_patience > 0:
        logger.info(
            "Early stopping enabled: patience=%s min_delta=%s min_epochs=%s metric=%s",
            args.early_stopping_patience,
            args.early_stopping_min_delta,
            args.early_stopping_min_epochs,
            args.best_metric,
        )

    metrics_mode = "a" if resume_checkpoint_path is not None and metrics_path.is_file() else "w"
    with metrics_path.open(metrics_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_chamfer_distance", "val_f_score"])
        if metrics_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, planned_end_epoch + 1):
            last_completed_epoch = epoch
            if (
                not encoder_unfrozen
                and args.unfreeze_epoch is not None
                and epoch >= args.unfreeze_epoch
            ):
                model.unfreeze_encoder()
                encoder_unfrozen = True
                args.encoder_unfrozen = True
                optimizer = build_optimizer(model, args)
                scheduler = build_lr_scheduler(optimizer, args)
                message = (
                    f"Unfroze encoder at epoch={epoch}; "
                    f"encoder_lr={args.encoder_lr} decoder_lr={args.decoder_lr or args.lr}"
                )
                print(message)
                logger.info(message)

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                amp_enabled=amp_enabled,
                grad_scaler=grad_scaler,
                chamfer_gt_weight=args.chamfer_gt_weight,
                repulsion_weight=args.repulsion_weight,
                repulsion_k=args.repulsion_k,
                repulsion_radius=args.repulsion_radius,
                repulsion_sample_size=args.repulsion_sample_size,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                threshold=args.f_threshold,
                amp_enabled=amp_enabled,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_chamfer_distance": val_metrics["chamfer_distance"],
                "val_f_score": val_metrics["f_score"],
            }
            writer.writerow(row)
            file.flush()

            current_score = row[args.best_metric]
            is_best_score = is_better_score(args.best_metric, current_score, best_score)
            is_early_stopping_improvement = is_better_score(
                args.best_metric,
                current_score,
                early_stopping_score,
                min_delta=args.early_stopping_min_delta,
            )
            lr_reduced = step_lr_scheduler(scheduler, optimizer, current_score)
            if is_best_score:
                best_score = current_score
                best_epoch = epoch
                args.encoder_unfrozen = encoder_unfrozen
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
                        scheduler=scheduler,
                        grad_scaler=grad_scaler,
                        resumed_from_checkpoint=resume_checkpoint_path,
                    ),
                    best_checkpoint_path,
                )
                best_text = f" best_{args.best_metric}={best_score:.6f}"
            else:
                best_text = ""

            if is_early_stopping_improvement:
                early_stopping_score = current_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            message = (
                f"epoch={epoch} "
                f"train_loss={train_loss:.6f} "
                f"val_cd={val_metrics['chamfer_distance']:.6f} "
                f"val_f={val_metrics['f_score']:.4f}"
                f"{best_text}"
            )
            if args.early_stopping_patience > 0:
                message += f" patience={epochs_without_improvement}/{args.early_stopping_patience}"
            message += f" lr={optimizer_lr_summary(optimizer)}"
            if lr_reduced:
                message += " lr_reduced=true"
            print(message)
            logger.info(message)

            epochs_run = epoch - start_epoch + 1
            if (
                args.early_stopping_patience > 0
                and epochs_run >= args.early_stopping_min_epochs
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                early_stopped = True
                early_stop_epoch = epoch
                message = (
                    f"Early stopping at epoch={epoch}: no {args.best_metric} improvement "
                    f">= {args.early_stopping_min_delta} for {epochs_without_improvement} epochs. "
                    f"Best epoch={best_epoch}, best_score={best_score}."
                )
                print(message)
                logger.info(message)
                break

    checkpoint_path = checkpoint_dir / "resnet_pointcloud_net.pt"
    args.encoder_unfrozen = encoder_unfrozen
    final_epoch = max(last_completed_epoch, start_epoch - 1)
    torch.save(
        build_checkpoint(
            model=model,
            args=args,
            epoch=final_epoch,
            best_metric=args.best_metric,
            best_score=best_score,
            best_epoch=best_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
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
        "split_validation": split_validation,
        "run_scale": "limited_smoke" if args.max_samples is not None else "full_split",
        "num_points": args.num_points,
        "image_size": args.image_size,
        "model_type": "resnet_pointcloud",
        "encoder_name": args.encoder_name,
        "feature_dim": args.feature_dim,
        "pretrained": args.pretrained,
        "freeze_encoder": args.freeze_encoder,
        "encoder_unfrozen": encoder_unfrozen,
        "decoder_lr": args.decoder_lr or args.lr,
        "encoder_lr": args.encoder_lr,
        "weight_decay": args.weight_decay,
        "amp_requested": bool(args.amp),
        "amp_enabled": amp_enabled,
        "lr_scheduler": args.lr_scheduler,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_threshold": args.lr_scheduler_threshold,
        "lr_scheduler_min_lr": args.lr_scheduler_min_lr,
        "final_learning_rates": optimizer_lr_values(optimizer),
        "chamfer_gt_weight": args.chamfer_gt_weight,
        "repulsion_weight": args.repulsion_weight,
        "repulsion_k": args.repulsion_k,
        "repulsion_radius": args.repulsion_radius,
        "repulsion_sample_size": args.repulsion_sample_size,
        "unfreeze_epoch": args.unfreeze_epoch,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "start_epoch": start_epoch,
        "end_epoch": final_epoch,
        "planned_end_epoch": planned_end_epoch,
        "early_stopping_enabled": args.early_stopping_patience > 0,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "early_stopping_min_epochs": args.early_stopping_min_epochs,
        "early_stopped": early_stopped,
        "early_stop_epoch": early_stop_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "learning_rate": args.lr,
        "device": str(device),
        "resume_enabled": getattr(args, "resume", True),
        "resumed_from_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path else None,
        "resumed_from_epoch": resumed_from_epoch,
        "optimizer_resumed": optimizer_resumed,
        "lr_scheduler_resumed": scheduler_resumed,
        "grad_scaler_resumed": grad_scaler_resumed,
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
        "end_epoch": final_epoch,
        "planned_end_epoch": planned_end_epoch,
        "early_stopped": early_stopped,
        "early_stop_epoch": early_stop_epoch,
        "optimizer_resumed": optimizer_resumed,
        "lr_scheduler_resumed": scheduler_resumed,
        "grad_scaler_resumed": grad_scaler_resumed,
    }


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
