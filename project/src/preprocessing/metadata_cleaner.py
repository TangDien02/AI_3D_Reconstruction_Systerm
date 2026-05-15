from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ["img", "mask", "model", "category", "bbox"]


def load_pix3d_json(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    json_path = raw_dir / "pix3d.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Pix3D metadata not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    return pd.DataFrame(records)


def _relative_processed_image_path(row: pd.Series) -> str:
    img_path = Path(str(row["img"]))
    return str(Path("images") / str(row["category"]) / f"{img_path.stem}.png").replace("\\", "/")


def _relative_processed_mask_path(row: pd.Series) -> str:
    mask_path = Path(str(row["mask"]))
    return str(Path("masks") / str(row["category"]) / f"{mask_path.stem}.png").replace("\\", "/")


def _relative_point_path(row: pd.Series) -> str:
    model_path = Path(str(row["model"]))
    model_id = model_path.parent.name or model_path.stem
    return str(Path("points") / str(row["category"]) / f"{model_id}.npy").replace("\\", "/")


def clean_pix3d_metadata(raw_dir: str | Path, categories: Iterable[str] | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    data = load_pix3d_json(raw_dir)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required Pix3D columns: {missing_columns}")

    if categories:
        category_set = set(categories)
        data = data[data["category"].isin(category_set)].copy()
    else:
        data = data.copy()

    data["img_exists"] = data["img"].apply(lambda path: (raw_dir / str(path)).is_file())
    data["mask_exists"] = data["mask"].apply(lambda path: (raw_dir / str(path)).is_file())
    data["model_exists"] = data["model"].apply(lambda path: (raw_dir / str(path)).is_file())

    clean_data = data[
        data["img"].notna()
        & data["mask"].notna()
        & data["model"].notna()
        & data["category"].notna()
        & data["bbox"].notna()
        & data["img_exists"]
        & data["mask_exists"]
        & data["model_exists"]
    ].copy()

    clean_data = clean_data.reset_index(drop=True)
    clean_data.insert(0, "sample_id", clean_data.index.map(lambda idx: f"pix3d_{idx:05d}"))
    clean_data["processed_image"] = clean_data.apply(_relative_processed_image_path, axis=1)
    clean_data["processed_mask"] = clean_data.apply(_relative_processed_mask_path, axis=1)
    clean_data["pointcloud"] = clean_data.apply(_relative_point_path, axis=1)

    return clean_data


def make_stratified_splits(
    metadata: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    if "category" not in metadata.columns:
        raise KeyError("metadata must contain a 'category' column.")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Use ratios where train_ratio > 0, val_ratio >= 0, and train + val < 1.")

    shuffled = metadata.groupby("category", group_keys=False).sample(frac=1, random_state=seed)
    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in shuffled.groupby("category", sort=False):
        n_items = len(group)
        train_end = int(n_items * train_ratio)
        val_end = train_end + int(n_items * val_ratio)

        if n_items >= 3:
            train_end = max(1, min(train_end, n_items - 2))
            val_end = max(train_end + 1, min(val_end, n_items - 1))
        elif n_items == 2:
            train_end = 1
            val_end = 1
        else:
            train_end = 1
            val_end = 1

        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    return {
        "train": pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True),
        "val": pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True),
        "test": pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True),
    }


def save_metadata_and_splits(
    metadata: pd.DataFrame,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "pix3d_clean_metadata.csv"
    _safe_to_csv(metadata, metadata_path)

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    paths = {"metadata": metadata_path}
    for split_name, split_data in make_stratified_splits(metadata, train_ratio, val_ratio, seed).items():
        split_path = split_dir / f"{split_name}.csv"
        _safe_to_csv(split_data, split_path)
        paths[split_name] = split_path

    return paths


def _safe_to_csv(data: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        data.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, output_path)
    except PermissionError as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        if output_path.exists():
            print(
                f"Warning: {output_path} is locked. Keeping the existing file and continuing.",
                flush=True,
            )
            return output_path
        raise PermissionError(
            f"Cannot write {output_path}. Close the file if it is open in Excel, VS Code preview, "
            "or another program, then run the command again."
        ) from exc

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Pix3D metadata and create train/val/test splits.")
    parser.add_argument("--raw-dir", default="data/raw/pix3d")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = clean_pix3d_metadata(args.raw_dir, categories=args.categories)
    paths = save_metadata_and_splits(
        metadata,
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Clean samples: {len(metadata)}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
