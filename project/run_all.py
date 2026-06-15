from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preprocessing, training, evaluation, and point-cloud comparison."
    )
    parser.add_argument("--category", default="chair")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default="results/chair_resnet_baseline")
    parser.add_argument("--encoder-name", choices=["conv", "resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decoder-lr", type=float, default=None)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze-epoch", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke test: 8 samples, 1 epoch.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Train from scratch even when output-dir already has best_model.pt.",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def checkpoint_for_next_step(train_outputs: dict) -> Path:
    checkpoint_path = Path(train_outputs["best_checkpoint_path"])
    if not checkpoint_path.is_file():
        checkpoint_path = Path(train_outputs["checkpoint_path"])
    return checkpoint_path


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = parse_args()
    if args.quick:
        args.max_samples = 8
        args.epochs = 1
        args.output_dir = "results/smoke_test"

    raw_dir = PROJECT_DIR / "data" / "raw" / "pix3d"
    processed_dir = PROJECT_DIR / "data" / "processed"
    output_dir = PROJECT_DIR / args.output_dir

    from src.preprocessing.metadata_cleaner import clean_pix3d_metadata, save_metadata_and_splits
    from src.preprocessing.build_processed_dataset import build_processed_images
    from src.preprocessing.mesh_processor import build_pointclouds_from_metadata
    from src.training.training_pipeline import run_training
    from src.evaluation.evaluate_baseline import evaluate_checkpoint
    from src.inference.compare_pointclouds import compare_sample

    if args.skip_preprocessing:
        print("Step 1/5 - Skipping preprocessing and using existing data/processed")
    else:
        print("Step 1/5 - Cleaning metadata and creating splits")
        metadata = clean_pix3d_metadata(raw_dir, categories=[args.category])
        paths = save_metadata_and_splits(metadata, processed_dir, train_ratio=0.7, val_ratio=0.15, seed=42)
        print(f"Clean metadata records: {len(metadata)}")

        print("Step 2/5 - Building processed images, masks, and point clouds")
        build_processed_images(
            metadata,
            raw_dir=raw_dir,
            output_dir=processed_dir,
            image_size=args.image_size,
            overwrite=args.overwrite,
            max_samples=args.max_samples,
            progress_interval=50,
        )
        build_pointclouds_from_metadata(
            metadata_csv=paths["metadata"],
            raw_dir=raw_dir,
            output_dir=processed_dir,
            num_points=args.num_points,
            seed=42,
            overwrite=args.overwrite,
            progress_interval=20,
            max_models=args.max_samples,
        )

    print("Step 3/5 - Training baseline")
    train_outputs = run_training(
        SimpleNamespace(
            raw_dir="data/raw/pix3d",
            processed_dir="data/processed",
            output_dir=args.output_dir,
            dataset_mode="processed",
            split="train",
            categories=[args.category],
            max_samples=args.max_samples,
            num_points=args.num_points,
            image_size=args.image_size,
            encoder_name=args.encoder_name,
            feature_dim=args.feature_dim,
            pretrained=args.pretrained,
            freeze_encoder=args.freeze_encoder,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=1e-4,
            decoder_lr=args.decoder_lr,
            encoder_lr=args.encoder_lr,
            weight_decay=args.weight_decay,
            unfreeze_epoch=args.unfreeze_epoch,
            f_threshold=0.05,
            best_metric="val_chamfer_distance",
            device=args.device,
            resume=args.resume,
            resume_checkpoint=args.resume_checkpoint,
            skip_evaluation=True,
            skip_comparison=True,
            post_split="test",
            eval_max_samples=None,
            comparison_index=0,
            max_plot_points=2048,
        )
    )
    checkpoint_path = checkpoint_for_next_step(train_outputs)

    print("Step 4/5 - Evaluating on test split")
    evaluation_summary = evaluate_checkpoint(
        SimpleNamespace(
            processed_dir="data/processed",
            checkpoint=str(checkpoint_path),
            output_dir=args.output_dir,
            split="test",
            categories=[args.category],
            max_samples=None,
            batch_size=args.batch_size,
            f_threshold=0.05,
            device=args.device,
        )
    )

    print("Step 5/5 - Comparing predicted point cloud with ground truth")
    comparison_dir = output_dir / "outputs" / "comparison"
    compare_sample(
        SimpleNamespace(
            checkpoint=str(checkpoint_path),
            processed_dir=str(processed_dir),
            output_dir=str(comparison_dir),
            split="test",
            categories=[args.category],
            index=0,
            max_samples=None,
            f_threshold=0.05,
            max_plot_points=2048,
            device=args.device,
            show=False,
        )
    )

    summary = {
        "category": args.category,
        "max_samples": args.max_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "training_metrics": str(train_outputs["metrics_path"]),
        "evaluation_summary": evaluation_summary,
        "comparison_dir": str(comparison_dir),
    }
    summary_path = output_dir / "outputs" / "run_all_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
