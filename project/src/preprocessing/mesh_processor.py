from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import trimesh


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


def load_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {mesh_path}")
    return mesh


def sample_mesh_points(mesh_path: str | Path, num_points: int = 2048, seed: int | None = 42) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    mesh = load_mesh(mesh_path)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return normalize_points(points)


def save_pointcloud(
    mesh_path: str | Path,
    output_path: str | Path,
    num_points: int = 2048,
    seed: int | None = 42,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    expected_metadata = {
        "source_mesh": str(Path(mesh_path).as_posix()),
        "num_points": int(num_points),
        "seed": seed,
        "normalization": "center_max_norm_v1",
        "format_version": 1,
    }

    if output_path.is_file() and not overwrite:
        try:
            points = np.load(output_path, mmap_mode="r")
            point_count_matches = points.ndim == 2 and points.shape[1] == 3 and points.shape[0] == num_points
        except Exception:
            point_count_matches = False

        if point_count_matches:
            if not metadata_path.is_file():
                metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")
            return output_path

        print(
            f"Regenerating incompatible point cloud artifact: {output_path}",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = sample_mesh_points(mesh_path, num_points=num_points, seed=seed)
    np.save(output_path, points)
    metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")
    return output_path

def clean_pointcloud_dir(output_dir: str | Path) -> None:
    points_dir = Path(output_dir) / "points"
    if points_dir.exists():
        shutil.rmtree(points_dir)
        print(f"Removed old point clouds: {points_dir}", flush=True)

def validate_metadata_columns(metadata: pd.DataFrame) -> None:
    required_cols = {"model", "pointcloud"}
    missing_cols = required_cols - set(metadata.columns)

    if missing_cols:
        raise KeyError(f"metadata must contain columns: {missing_cols}")


def normalize_relative_path(path: object) -> str:
    return str(path).replace("\\", "/").strip()


def get_unique_pointcloud_rows(
    metadata: pd.DataFrame,
    max_models: int | None = None,
) -> pd.DataFrame:
    validate_metadata_columns(metadata)

    metadata = metadata.copy()
    metadata["_model_norm"] = metadata["model"].apply(normalize_relative_path)
    metadata["_pointcloud_norm"] = metadata["pointcloud"].apply(normalize_relative_path)

    collision_check = (
        metadata.groupby("_pointcloud_norm")["_model_norm"]
        .nunique()
    )

    collisions = collision_check[collision_check > 1]

    if len(collisions) > 0:
        raise ValueError(
            "Pointcloud path collision detected. "
            "Different CAD models are mapped to the same .npy file."
        )

    unique_rows = (
        metadata[["model", "pointcloud", "_pointcloud_norm"]]
        .drop_duplicates("_pointcloud_norm")
        .reset_index(drop=True)
    )

    if max_models is not None:
        unique_rows = unique_rows.head(max_models)

    return unique_rows

def check_pointcloud_shapes(
    points_dir: str | Path,
    expected_num_points: int,
) -> None:
    points_dir = Path(points_dir)
    expected_shape = (expected_num_points, 3)

    shape_counter = Counter()
    bad_files = []

    for path in points_dir.rglob("*.npy"):
        points = np.load(path)
        shape = tuple(points.shape)
        shape_counter[shape] += 1

        if shape != expected_shape:
            bad_files.append((path, shape))

    print("Point cloud shape summary:", flush=True)
    for shape, count in shape_counter.items():
        print(f"{shape}: {count} files", flush=True)

    if bad_files:
        examples = "\n".join(
            f"- {path}: {shape}" for path, shape in bad_files[:10]
        )

        raise ValueError(
            f"Mixed point cloud shapes detected. "
            f"Expected only {expected_shape}.\n{examples}"
        )

def build_pointclouds_from_metadata(
    metadata_csv: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
    num_points: int = 2048,
    seed: int | None = 42,
    overwrite: bool = False,
    clean_output: bool = False,
    progress_interval: int = 50,
    max_models: int | None = None,
) -> list[Path]:
    metadata_csv = Path(metadata_csv)
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw Pix3D directory not found: {raw_dir}")

    if clean_output:
        clean_pointcloud_dir(output_dir)

    metadata = pd.read_csv(metadata_csv)

    unique_rows = get_unique_pointcloud_rows(
        metadata=metadata,
        max_models=max_models,
    )

    saved_paths = []
    total = len(unique_rows)

    for index, row in enumerate(unique_rows.itertuples(index=False), start=1):
        mesh_path = raw_dir / str(row.model)
        point_path = output_dir / str(row.pointcloud)

        saved_path = save_pointcloud(
            mesh_path=mesh_path,
            output_path=point_path,
            num_points=num_points,
            seed=seed,
            overwrite=overwrite,
        )

        saved_paths.append(saved_path)

        if progress_interval > 0 and (
            index == 1
            or index % progress_interval == 0
            or index == total
        ):
            print(f"Point clouds ready: {index}/{total}", flush=True)

    check_pointcloud_shapes(
        points_dir=output_dir / "points",
        expected_num_points=num_points,
    )

    return saved_paths

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample normalized Pix3D mesh point clouds.")
    parser.add_argument("--metadata-csv", default="data/processed/pix3d_clean_metadata.csv")
    parser.add_argument("--raw-dir", default="data/raw/pix3d")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--max-models", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = build_pointclouds_from_metadata(
        metadata_csv=args.metadata_csv,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        num_points=args.num_points,
        seed=args.seed,
        overwrite=args.overwrite,
        clean_output=args.clean_output,
        progress_interval=args.progress_interval,
        max_models=args.max_models,
    )
    print(f"Point clouds ready: {len(saved_paths)}")


if __name__ == "__main__":
    main()
