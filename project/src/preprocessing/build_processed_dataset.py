from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.preprocessing.metadata_cleaner import clean_pix3d_metadata, save_metadata_and_splits


def parse_bbox(
    value: object,
    image_size: tuple[int, int],
    padding_ratio: float = 0.0,
) -> tuple[int, int, int, int]:
    width, height = image_size
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return 0, 0, width, height

    x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    if padding_ratio > 0:
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        pad_x = int(round(box_width * padding_ratio))
        pad_y = int(round(box_height * padding_ratio))
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def letterbox_image_and_mask(
    image: "Image.Image",
    mask: "Image.Image",
    image_size: int,
) -> tuple["Image.Image", "Image.Image"]:
    from PIL import Image

    width, height = image.size
    scale = image_size / max(width, height)
    resized_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resized_image = image.resize(resized_size, Image.Resampling.BILINEAR)
    resized_mask = mask.resize(resized_size, Image.Resampling.NEAREST)

    output_image = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    output_mask = Image.new("L", (image_size, image_size), 0)
    offset = (
        (image_size - resized_size[0]) // 2,
        (image_size - resized_size[1]) // 2,
    )
    output_image.paste(resized_image, offset)
    output_mask.paste(resized_mask, offset)
    return output_image, output_mask


def preprocess_image_and_mask(
    image_path: str | Path,
    mask_path: str | Path,
    output_image_path: str | Path,
    output_mask_path: str | Path,
    bbox: object,
    image_size: int = 224,
    crop_padding: float = 0.10,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    import numpy as np
    from PIL import Image

    output_image_path = Path(output_image_path)
    output_mask_path = Path(output_mask_path)
    if output_image_path.is_file() and output_mask_path.is_file() and not overwrite:
        return output_image_path, output_mask_path

    image = load_image_safely(image_path, mode="RGB")
    mask = load_image_safely(mask_path, mode="L").resize(image.size)
    crop_box = parse_bbox(bbox, image.size, padding_ratio=crop_padding)

    image = image.crop(crop_box)
    mask = mask.crop(crop_box)

    image_np = np.asarray(image).astype(np.uint8)
    mask_np = np.asarray(mask) > 0
    masked_np = np.full_like(image_np, 255)
    masked_np[mask_np] = image_np[mask_np]

    masked_image = Image.fromarray(masked_np)
    binary_mask = Image.fromarray(mask_np.astype(np.uint8) * 255)
    processed_image, processed_mask = letterbox_image_and_mask(masked_image, binary_mask, image_size)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    processed_image.save(output_image_path)
    processed_mask.save(output_mask_path)
    return output_image_path, output_mask_path


def load_image_safely(image_path: str | Path, mode: str) -> "Image.Image":
    from PIL import Image

    image = Image.open(image_path)
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    return image.convert(mode)


def build_processed_images(
    metadata: pd.DataFrame,
    raw_dir: str | Path,
    output_dir: str | Path,
    image_size: int = 224,
    crop_padding: float = 0.10,
    overwrite: bool = False,
    max_samples: int | None = None,
    progress_interval: int = 100,
) -> int:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    rows = metadata.head(max_samples) if max_samples is not None else metadata
    total = len(rows)

    processed_count = 0
    for row in rows.itertuples(index=False):
        preprocess_image_and_mask(
            image_path=raw_dir / str(row.img),
            mask_path=raw_dir / str(row.mask),
            output_image_path=output_dir / str(row.processed_image),
            output_mask_path=output_dir / str(row.processed_mask),
            bbox=row.bbox,
            image_size=image_size,
            crop_padding=crop_padding,
            overwrite=overwrite,
        )
        processed_count += 1
        if progress_interval > 0 and (
            processed_count == 1
            or processed_count % progress_interval == 0
            or processed_count == total
        ):
            print(f"Processed images/masks: {processed_count}/{total}", flush=True)

    return processed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the processed Pix3D dataset.")
    parser.add_argument("--raw-dir", default="data/raw/pix3d")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-padding", type=float, default=0.10)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-pointclouds", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 0:
        args.max_samples = None
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    metadata = clean_pix3d_metadata(raw_dir, categories=args.categories)
    paths = save_metadata_and_splits(
        metadata,
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Clean samples: {len(metadata)}")
    print(f"Metadata: {paths['metadata']}")
    print(f"Splits: {paths['train']}, {paths['val']}, {paths['test']}")

    if args.metadata_only:
        return

    if not args.skip_images:
        count = build_processed_images(
            metadata,
            raw_dir=raw_dir,
            output_dir=output_dir,
            image_size=args.image_size,
            crop_padding=max(0.0, args.crop_padding),
            overwrite=args.overwrite,
            max_samples=args.max_samples,
            progress_interval=args.progress_interval,
        )
        print(f"Processed images/masks: {count}")

    if not args.skip_pointclouds:
        from src.preprocessing.mesh_processor import build_pointclouds_from_metadata

        point_paths = build_pointclouds_from_metadata(
            metadata_csv=paths["metadata"],
            raw_dir=raw_dir,
            output_dir=output_dir,
            num_points=args.num_points,
            seed=args.seed,
            overwrite=args.overwrite,
            progress_interval=args.progress_interval,
            max_models=args.max_samples,
        )
        print(f"Point clouds ready: {len(point_paths)}")


if __name__ == "__main__":
    main()
