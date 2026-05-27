from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]


def cad_uid_from_model(path: object) -> str:
    normalized = str(path).replace("\\", "/").strip()
    return Path(normalized).parent.as_posix() or normalized


def load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / "splits" / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split CSV: {path}")
    frame = pd.read_csv(path)
    if "cad_uid" not in frame.columns and "model" in frame.columns:
        frame["cad_uid"] = frame["model"].apply(cad_uid_from_model)
    if "cad_uid" not in frame.columns:
        raise KeyError(f"{path} must contain cad_uid or model.")
    return frame


def filter_categories(frame: pd.DataFrame, categories: list[str] | None) -> pd.DataFrame:
    if not categories:
        return frame.copy()
    return frame[frame["category"].isin(categories)].copy()


def hard_case_mask(frame: pd.DataFrame) -> pd.Series:
    hard = pd.Series(False, index=frame.index)
    for column in ("truncated", "occluded", "slightly_occluded"):
        if column in frame.columns:
            hard = hard | frame[column].fillna(False).astype(bool)
    return hard


def choose_rows(frame: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_samples:
        return frame.copy()
    return frame.sample(n=max_samples, random_state=seed).sort_values("_dataset_index").copy()


def write_manifest(frame: pd.DataFrame, output_path: Path, reason: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = frame.copy().reset_index(drop=True)
    manifest.insert(0, "benchmark_index", range(len(manifest)))
    manifest["dataset_index"] = manifest["_dataset_index"].astype(int)
    manifest["reason"] = reason
    columns = [
        "benchmark_index",
        "dataset_index",
        "sample_id",
        "category",
        "processed_image",
        "pointcloud",
        "model_uid",
        "cad_uid",
        "reason",
    ]
    manifest[columns].to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def build_manifests(args: argparse.Namespace) -> dict[str, Path]:
    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_DIR / processed_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir

    train = filter_categories(load_split(processed_dir, "train"), args.categories).reset_index(drop=True)
    train["_dataset_index"] = train.index
    target = filter_categories(load_split(processed_dir, args.split), args.categories).reset_index(drop=True)
    target["_dataset_index"] = target.index

    train_cads = set(train["cad_uid"].astype(str))
    target_cads = target["cad_uid"].astype(str)
    hard = hard_case_mask(target)
    train_hard = hard_case_mask(train)

    groups = {
        "seen_cad_train": (train[~train_hard], "train"),
        f"unseen_cad_{args.split}": (target[~target_cads.isin(train_cads) & ~hard], args.split),
        f"hard_cases_{args.split}": (target[hard], args.split),
    }

    paths = {}
    for name, (group, split_name) in groups.items():
        selected = choose_rows(group, max_samples=args.max_samples, seed=args.seed)
        output_path = output_dir / f"{name}.csv"
        paths[name] = write_manifest(selected, output_path=output_path, reason=name)
        print(f"{name}: {len(selected)} samples from split={split_name} -> {paths[name]}")

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Pix3D fixed benchmark manifests by CAD difficulty.")
    parser.add_argument("--processed-dir", default="data/processed_2048")
    parser.add_argument("--output-dir", default="benchmarks/generated")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--categories", nargs="*", default=["chair"])
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_samples <= 0:
        args.max_samples = 20
    return args


def main() -> None:
    build_manifests(parse_args())


if __name__ == "__main__":
    main()
