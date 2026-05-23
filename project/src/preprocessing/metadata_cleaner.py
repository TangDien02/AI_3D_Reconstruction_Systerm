from __future__ import annotations

import argparse
import hashlib
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


def _stable_model_uid(model_rel_path: str) -> str:
    normalized_path = str(model_rel_path).replace("\\", "/").strip()
    path_hash = hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()[:12]
    model_stem = Path(normalized_path).stem or "model"
    return f"{model_stem}_{path_hash}"


def _relative_point_path(row: pd.Series) -> str:
    model_uid = str(row["model_uid"])
    return str(Path("points") / str(row["category"]) / f"{model_uid}.npy").replace("\\", "/")


def validate_metadata_paths(metadata: pd.DataFrame) -> None:
    model_uid_collisions = metadata.groupby("model_uid")["model"].nunique()
    collided_uids = model_uid_collisions[model_uid_collisions > 1]
    if not collided_uids.empty:
        examples = ", ".join(collided_uids.head(5).index.astype(str))
        raise RuntimeError(f"model_uid collision detected for: {examples}")

    pointcloud_collisions = metadata.groupby("pointcloud")["model"].nunique()
    collided_paths = pointcloud_collisions[pointcloud_collisions > 1]
    if not collided_paths.empty:
        examples = ", ".join(collided_paths.head(5).index.astype(str))
        raise RuntimeError(f"pointcloud path collision detected for: {examples}")


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
    clean_data["model_uid"] = clean_data["model"].apply(_stable_model_uid)
    clean_data["processed_image"] = clean_data.apply(_relative_processed_image_path, axis=1)
    clean_data["processed_mask"] = clean_data.apply(_relative_processed_mask_path, axis=1)
    clean_data["pointcloud"] = clean_data.apply(_relative_point_path, axis=1)
    validate_metadata_paths(clean_data)

    return clean_data


def make_stratified_splits(
    metadata: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    required_cols = {"category", "model_uid"}
    missing_cols = required_cols - set(metadata.columns)

    if missing_cols:
        raise KeyError(f"metadata must contain columns: {missing_cols}")

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "Use ratios where train_ratio > 0, val_ratio >= 0, and train + val < 1."
        )

    train_parts = []
    val_parts = []
    test_parts = []

    for category, group in metadata.groupby("category", sort=False):
        unique_models = (
            group[["model_uid"]]
            .drop_duplicates()
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

        n_models = len(unique_models)

        if n_models == 0:
            continue

        if n_models == 1:
            train_model_ids = set(unique_models["model_uid"])
            val_model_ids = set()
            test_model_ids = set()

        elif n_models == 2:
            train_model_ids = set(unique_models.iloc[:1]["model_uid"])
            val_model_ids = set()
            test_model_ids = set(unique_models.iloc[1:]["model_uid"])

        else:
            train_end = int(n_models * train_ratio)
            val_end = train_end + int(n_models * val_ratio)

            train_end = max(1, min(train_end, n_models - 2))
            val_end = max(train_end + 1, min(val_end, n_models - 1))

            train_model_ids = set(unique_models.iloc[:train_end]["model_uid"])
            val_model_ids = set(unique_models.iloc[train_end:val_end]["model_uid"])
            test_model_ids = set(unique_models.iloc[val_end:]["model_uid"])

        train_parts.append(group[group["model_uid"].isin(train_model_ids)])
        val_parts.append(group[group["model_uid"].isin(val_model_ids)])
        test_parts.append(group[group["model_uid"].isin(test_model_ids)])

    def _concat_or_empty(parts: list[pd.DataFrame]) -> pd.DataFrame:
        valid_parts = [part for part in parts if len(part) > 0]

        if not valid_parts:
            return metadata.iloc[0:0].copy().reset_index(drop=True)

        return (
            pd.concat(valid_parts)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

    return {
        "train": _concat_or_empty(train_parts),
        "val": _concat_or_empty(val_parts),
        "test": _concat_or_empty(test_parts),
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
