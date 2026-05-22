from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils.logger import get_logger


logger = get_logger("MainWorkflow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Pix3D preprocessing and baseline training workflow."
    )
    parser.add_argument("--raw-dir", default="data/raw/pix3d")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results/baseline")
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--f-threshold", type=float, default=0.05)
    parser.add_argument(
        "--best-metric",
        choices=["val_chamfer_distance", "val_f_score"],
        default="val_chamfer_distance",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-pointclouds", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 0:
        args.max_samples = None
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
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        transformer_depth=args.transformer_depth,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        f_threshold=args.f_threshold,
        best_metric=args.best_metric,
    )


def ensure_training_dependencies() -> None:
    missing = [
        package
        for package in ("torch", "numpy", "pandas", "PIL")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing training dependencies: {names}. "
            "Install project/requirements.txt in a Python environment that supports PyTorch."
        )


def main() -> None:
    args = parse_args()

    if args.skip_preprocessing:
        logger.info("Bước 1/2: bỏ qua preprocessing, dùng dữ liệu có sẵn trong %s.", args.processed_dir)
    else:
        run_preprocessing(args)

    if args.skip_training:
        logger.info("Bước 2/2: bỏ qua training baseline theo tham số --skip-training.")
        return

    logger.info("Bước 2/2: train baseline Transformer point cloud.")
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
