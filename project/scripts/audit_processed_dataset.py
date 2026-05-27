from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]


def cad_uid_from_model(path: object) -> str:
    normalized = str(path).replace("\\", "/").strip()
    return Path(normalized).parent.as_posix() or normalized


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    frame = pd.read_csv(path)
    if "cad_uid" not in frame.columns and "model" in frame.columns:
        frame["cad_uid"] = frame["model"].apply(cad_uid_from_model)
    return frame


def check_metadata(metadata: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required_columns = {"model", "model_uid", "cad_uid", "pointcloud", "processed_image", "category"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        return [f"metadata missing columns: {sorted(missing_columns)}"]

    pointcloud_collisions = metadata.groupby("pointcloud")["model"].nunique()
    for pointcloud, model_count in pointcloud_collisions[pointcloud_collisions > 1].items():
        errors.append(f"pointcloud collision: {pointcloud} maps to {model_count} CAD models")

    uid_collisions = metadata.groupby("model_uid")["model"].nunique()
    for model_uid, model_count in uid_collisions[uid_collisions > 1].items():
        errors.append(f"model_uid collision: {model_uid} maps to {model_count} CAD models")

    return errors


def check_split_leakage(split_dir: Path) -> list[str]:
    errors: list[str] = []
    frames = {name: load_csv(split_dir / f"{name}.csv") for name in ("train", "val", "test")}
    for name, frame in frames.items():
        if "model_uid" not in frame.columns:
            errors.append(f"{name}.csv missing model_uid")
        if "cad_uid" not in frame.columns:
            errors.append(f"{name}.csv missing cad_uid")

    if errors:
        return errors

    for key_name in ("model_uid", "cad_uid"):
        for left_name, right_name in (("train", "val"), ("train", "test"), ("val", "test")):
            left = set(frames[left_name][key_name].astype(str))
            right = set(frames[right_name][key_name].astype(str))
            overlap = sorted(left & right)
            if overlap:
                errors.append(
                    f"{key_name} leakage {left_name}-{right_name}: {', '.join(overlap[:5])}"
                )
    return errors


def add_split_counts(metadata: pd.DataFrame) -> None:
    print(f"Metadata rows: {len(metadata)}")
    print(f"CAD folders: {metadata['cad_uid'].nunique() if 'cad_uid' in metadata else 'n/a'}")
    print(f"CAD models: {metadata['model_uid'].nunique() if 'model_uid' in metadata else 'n/a'}")
    print(f"Pointcloud paths: {metadata['pointcloud'].nunique() if 'pointcloud' in metadata else 'n/a'}")


def check_artifacts(processed_dir: Path, metadata: pd.DataFrame, num_points: int | None) -> list[str]:
    errors: list[str] = []
    image_paths = metadata[["processed_image"]].drop_duplicates()
    for row in image_paths.itertuples(index=False):
        image_path = processed_dir / str(row.processed_image)
        if not image_path.is_file():
            errors.append(f"missing processed image: {image_path}")

    pointcloud_paths = metadata[["pointcloud"]].drop_duplicates()
    for row in pointcloud_paths.itertuples(index=False):
        pointcloud_path = processed_dir / str(row.pointcloud)
        if not pointcloud_path.is_file():
            errors.append(f"missing pointcloud: {pointcloud_path}")
            continue
        if num_points is None:
            continue
        try:
            points = np.load(pointcloud_path, mmap_mode="r")
        except Exception as exc:
            errors.append(f"cannot load pointcloud {pointcloud_path}: {exc}")
            continue
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] != num_points:
            errors.append(
                f"bad pointcloud shape {tuple(points.shape)} expected ({num_points}, 3): {pointcloud_path}"
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit processed Pix3D metadata, splits, and artifacts.")
    parser.add_argument("--processed-dir", default=str(PROJECT_DIR / "data" / "processed"))
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--max-errors", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    metadata = load_csv(processed_dir / "pix3d_clean_metadata.csv")

    errors = []
    errors.extend(check_metadata(metadata))
    errors.extend(check_split_leakage(processed_dir / "splits"))
    if not args.skip_artifacts:
        errors.extend(check_artifacts(processed_dir, metadata, args.num_points))

    add_split_counts(metadata)

    if errors:
        print("AUDIT FAILED")
        for error in errors[: args.max_errors]:
            print(f"- {error}")
        remaining = len(errors) - args.max_errors
        if remaining > 0:
            print(f"... {remaining} more errors")
        sys.exit(1)

    print("AUDIT OK: no pointcloud collisions, no model/cad leakage, artifacts match expected shape.")


if __name__ == "__main__":
    main()
