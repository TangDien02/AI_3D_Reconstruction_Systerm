from __future__ import annotations

import argparse
from pathlib import Path

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
    if output_path.is_file() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = sample_mesh_points(mesh_path, num_points=num_points, seed=seed)
    np.save(output_path, points)
    return output_path


def build_pointclouds_from_metadata(
    metadata_csv: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
    num_points: int = 2048,
    seed: int | None = 42,
    overwrite: bool = False,
    progress_interval: int = 50,
    max_models: int | None = None,
) -> list[Path]:
    metadata = pd.read_csv(metadata_csv)
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    if "model" not in metadata.columns or "pointcloud" not in metadata.columns:
        raise KeyError("metadata must contain 'model' and 'pointcloud' columns.")

    saved_paths = []
    unique_pointclouds = metadata[["model", "pointcloud"]].drop_duplicates("pointcloud")
    if max_models is not None:
        unique_pointclouds = unique_pointclouds.head(max_models)
    total = len(unique_pointclouds)
    for index, row in enumerate(unique_pointclouds.itertuples(index=False), start=1):
        mesh_path = raw_dir / str(row.model)
        point_path = output_dir / str(row.pointcloud)
        saved_paths.append(
            save_pointcloud(
                mesh_path,
                point_path,
                num_points=num_points,
                seed=seed,
                overwrite=overwrite,
            )
        )
        if progress_interval > 0 and (
            index == 1
            or index % progress_interval == 0
            or index == total
        ):
            print(f"Point clouds ready: {index}/{total}", flush=True)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample normalized Pix3D mesh point clouds.")
    parser.add_argument("--metadata-csv", default="data/processed/pix3d_clean_metadata.csv")
    parser.add_argument("--raw-dir", default="data/raw/pix3d")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
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
        progress_interval=args.progress_interval,
        max_models=args.max_models,
    )
    print(f"Point clouds ready: {len(saved_paths)}")


if __name__ == "__main__":
    main()
