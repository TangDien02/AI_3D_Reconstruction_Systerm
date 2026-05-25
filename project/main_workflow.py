from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils.logger import get_logger


logger = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Pix3D preprocessing and baseline training workflow."
    )
    parser.add_argument("--raw-dir", default="data/raw/pix3d")
    parser.add_argument("--processed-dir", default="data/processed_2048")
    parser.add_argument("--output-dir", default="results/chair_resnet_baseline")
    parser.add_argument("--categories", nargs="+", default=["chair"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples for quick smoke tests. Omit or pass -1 to use all samples.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=None)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="plateau")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.7)
    parser.add_argument("--lr-scheduler-patience", type=int, default=5)
    parser.add_argument("--lr-scheduler-threshold", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--chamfer-gt-weight", type=float, default=1.25)
    parser.add_argument("--repulsion-weight", type=float, default=0.01)
    parser.add_argument("--repulsion-k", type=int, default=8)
    parser.add_argument("--repulsion-radius", type=float, default=0.03)
    parser.add_argument("--repulsion-sample-size", type=int, default=512)
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply train-only image augmentation. Validation/test remain unchanged.",
    )
    parser.add_argument("--augment-brightness", type=float, default=0.15)
    parser.add_argument("--augment-contrast", type=float, default=0.15)
    parser.add_argument("--augment-noise-std", type=float, default=0.01)
    parser.add_argument("--augment-erasing-prob", type=float, default=0.10)
    parser.add_argument("--augment-erasing-scale", type=float, default=0.12)
    parser.add_argument("--unfreeze-epoch", type=int, default=10)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop training after this many epochs without validation improvement.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=12)
    parser.add_argument("--f-threshold", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--best-metric",
        choices=["val_chamfer_distance", "val_f_score"],
        default="val_chamfer_distance",
        help="Validation metric used to update outputs/checkpoints/best_model.pt.",
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
    parser.set_defaults(resume=True)
    parser.add_argument("--encoder-name", choices=["conv", "resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-pointclouds", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 0:
        args.max_samples = None
    if args.early_stopping_patience < 0:
        args.early_stopping_patience = 0
    if args.early_stopping_min_epochs < 0:
        args.early_stopping_min_epochs = 0
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


def run_preprocessing(args: argparse.Namespace) -> dict[str, Path]:
    from src.preprocessing.build_processed_dataset import build_processed_images
    from src.preprocessing.mesh_processor import build_pointclouds_from_metadata
    from src.preprocessing.metadata_cleaner import (
        clean_pix3d_metadata,
        save_metadata_and_splits,
    )

    raw_dir = PROJECT_DIR / args.raw_dir
    processed_dir = PROJECT_DIR / args.processed_dir

    logger.info("Bước 1/2: làm sạch metadata và build processed dataset.")
    metadata = clean_pix3d_metadata(raw_dir, categories=args.categories)
    paths = save_metadata_and_splits(
        metadata,
        processed_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    logger.info("Clean samples: %s", len(metadata))
    logger.info("Metadata: %s", paths["metadata"])
    logger.info("Splits: %s, %s, %s", paths["train"], paths["val"], paths["test"])

    if not args.skip_images:
        image_count = build_processed_images(
            metadata,
            raw_dir=raw_dir,
            output_dir=processed_dir,
            image_size=args.image_size,
            overwrite=args.overwrite,
            max_samples=args.max_samples,
            progress_interval=args.progress_interval,
        )
        logger.info("Processed images/masks: %s", image_count)

    if not args.skip_pointclouds:
        point_paths = build_pointclouds_from_metadata(
            metadata_csv=paths["metadata"],
            raw_dir=raw_dir,
            output_dir=processed_dir,
            num_points=args.num_points,
            seed=args.seed,
            overwrite=args.overwrite,
            progress_interval=args.progress_interval,
            max_models=args.max_samples,
        )
        logger.info("Point clouds ready: %s", len(point_paths))

    return paths


def make_training_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        raw_dir="data/raw",
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        dataset_mode="processed",
        split="train",
        categories=args.categories,
        max_samples=args.max_samples,
        num_points=args.num_points,
        image_size=args.image_size,
        encoder_name=args.encoder_name,
        feature_dim=args.feature_dim,
        pretrained=args.pretrained,
        freeze_encoder=args.freeze_encoder,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        decoder_lr=args.decoder_lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        amp=args.amp,
        lr_scheduler=args.lr_scheduler,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_threshold=args.lr_scheduler_threshold,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr,
        chamfer_gt_weight=args.chamfer_gt_weight,
        repulsion_weight=args.repulsion_weight,
        repulsion_k=args.repulsion_k,
        repulsion_radius=args.repulsion_radius,
        repulsion_sample_size=args.repulsion_sample_size,
        augment=args.augment,
        augment_brightness=args.augment_brightness,
        augment_contrast=args.augment_contrast,
        augment_noise_std=args.augment_noise_std,
        augment_erasing_prob=args.augment_erasing_prob,
        augment_erasing_scale=args.augment_erasing_scale,
        unfreeze_epoch=args.unfreeze_epoch,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        f_threshold=args.f_threshold,
        best_metric=args.best_metric,
        device=args.device,
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint,
    )


def ensure_training_dependencies() -> None:
    missing = [
        package
        for package in ("torch", "torchvision", "numpy", "pandas", "PIL")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing training dependencies: {names}. "
            "Install project/requirements.txt in a Python environment that supports PyTorch."
        )


def main() -> None:
    global logger
    args = parse_args()
    logger = get_logger("MainWorkflow", Path(args.output_dir) / "logs")

    if args.skip_preprocessing:
        logger.info("Bước 1/2: bỏ qua preprocessing, dùng dữ liệu có sẵn trong %s.", args.processed_dir)
    else:
        run_preprocessing(args)

    if args.skip_training:
        logger.info("Bước 2/2: bỏ qua training baseline theo tham số --skip-training.")
        return

    logger.info("Bước 2/2: train baseline ResNet point cloud.")
    ensure_training_dependencies()
    from src.training.training_pipeline import run_training

    outputs = run_training(make_training_args(args))
    logger.info("Metrics: %s", outputs["metrics_path"])
    logger.info("Checkpoint: %s", outputs["checkpoint_path"])
    logger.info("Summary: %s", outputs["summary_path"])
    if outputs["plot_path"]:
        logger.info("Training curves: %s", outputs["plot_path"])
    logger.info("Hoàn tất workflow preprocessing -> baseline.")


if __name__ == "__main__":
    main()
