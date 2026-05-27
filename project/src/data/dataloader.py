import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Ghi chu:
# File nay la dataloader cho Pix3D. Nhiem vu chinh la bien du lieu tho
# thanh sample sach de dua vao training pipeline.
#
# Moi sample tra ve gom:
# - image: anh RGB da tach nen bang mask, resize va chuyen thanh tensor [3, H, W]
# - category: nhan lop cua vat the, vi du chair/table/sofa
# - points_gt: point cloud ground truth duoc sample tu CAD model .obj
# - model_path, image_path: duong dan de truy vet du lieu khi debug/bao cao
#


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


def load_and_sample_obj(model_path: str, num_points: int = 2048) -> torch.Tensor:
    import trimesh

    mesh = trimesh.load_mesh(model_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    points = normalize_points(points)
    return torch.from_numpy(points)


class Pix3DDataset(Dataset):
    def __init__(
        self,
        root_dir,
        categories=None,
        split="train",
        image_size=224,
        num_points=2048,
        max_samples=None,
        transform=None,
        use_mask_channel: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.categories = set(categories) if categories else None
        self.split = split
        self.image_size = image_size
        self.num_points = num_points
        self.transform = transform
        self.use_mask_channel = bool(use_mask_channel)

        with open(self.root_dir / "pix3d.json", "r", encoding="utf-8") as f:
            self.items = json.load(f)

        self.items = [
            item for item in self.items
            if item.get("img")
            and item.get("mask")
            and item.get("model")
            and (self.categories is None or item.get("category") in self.categories)
            and (self.root_dir / item["img"]).is_file()
            and (self.root_dir / item["mask"]).is_file()
            and (self.root_dir / item["model"]).is_file()
        ]
        if max_samples is not None:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        image_path = self.root_dir / item["img"]
        mask_path = self.root_dir / item["mask"]
        model_path = self.root_dir / item["model"]

        image = self._load_image_safely(image_path, mode="RGB")
        mask = self._load_image_safely(mask_path, mode="L").resize(image.size)
        image = self._apply_mask(image, mask)
        image = image.resize((self.image_size, self.image_size))
        image_tensor = self._to_tensor(image)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        if self.use_mask_channel:
            mask_tensor = self._mask_to_tensor(mask.resize((self.image_size, self.image_size)))
            image_tensor = torch.cat([image_tensor, mask_tensor], dim=0)

        points_gt = load_and_sample_obj(str(model_path), num_points=self.num_points)

        return {
            "image": image_tensor,
            "category": item["category"],
            "points_gt": points_gt,
            "model_path": str(model_path),
            "image_path": str(image_path),
        }

    @staticmethod
    def _apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
        image_np = np.asarray(image).astype(np.uint8)
        mask_np = np.asarray(mask) > 0
        masked = np.full_like(image_np, 255)
        masked[mask_np] = image_np[mask_np]
        return Image.fromarray(masked)

    @staticmethod
    def _load_image_safely(image_path: str | Path, mode: str) -> Image.Image:
        image = Image.open(image_path)
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        return image.convert(mode)

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    @staticmethod
    def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
        arr = (np.asarray(mask).astype(np.float32) > 0).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

def _resolve_existing_flags(
    items: pd.DataFrame,
    processed_dir: Path,
    file_columns: list[str],
) -> pd.DataFrame:
    items = items.copy()

    for column in file_columns:
        flag_column = f"_{column}_exists"
        items[flag_column] = items[column].apply(
            lambda path: (processed_dir / str(path)).is_file()
        )

    return items


def _raise_if_missing_files(
    items: pd.DataFrame,
    split: str,
    processed_dir: Path,
    file_columns: list[str],
) -> None:
    missing_parts = []

    for column in file_columns:
        flag_column = f"_{column}_exists"
        missing_rows = items[~items[flag_column]]

        if len(missing_rows) > 0:
            examples = missing_rows[column].head(10).tolist()
            missing_parts.append(
                f"{column}: missing {len(missing_rows)}/{len(items)} files\n"
                f"examples: {examples}"
            )

    if missing_parts:
        raise FileNotFoundError(
            f"Missing processed dataset files in split='{split}'.\n"
            f"processed_dir={processed_dir}\n\n"
            + "\n\n".join(missing_parts)
            + "\n\nFix: run build_processed_dataset again, or use the correct --processed-dir."
        )

class ProcessedPix3DDataset(Dataset):
    def __init__(
        self,
        processed_dir,
        split="train",
        categories=None,
        max_samples=None,
        transform=None,
        require_files=True,
        allow_missing_files=False,
        expected_num_points: int | None = None,
        use_mask_channel: bool = False,
        exclude_truncated: bool = False,
        exclude_occluded: bool = False,
        exclude_slightly_occluded: bool = False,
    ):
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.categories = set(categories) if categories else None
        self.transform = transform
        self.expected_num_points = expected_num_points
        self.use_mask_channel = bool(use_mask_channel)

        split_path = self.processed_dir / "splits" / f"{split}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        items = pd.read_csv(split_path)
        original_count = len(items)
        if "cad_uid" not in items.columns and "model" in items.columns:
            items["cad_uid"] = items["model"].apply(
                lambda path: Path(str(path).replace("\\", "/").strip()).parent.as_posix()
            )

        if self.categories:
            items = items[items["category"].isin(self.categories)].copy()

        for column, enabled in (
            ("truncated", exclude_truncated),
            ("occluded", exclude_occluded),
            ("slightly_occluded", exclude_slightly_occluded),
        ):
            if enabled and column in items.columns:
                items = items[~items[column].fillna(False).astype(bool)].copy()

        after_category_count = len(items)

        required_columns = [
            "processed_image",
            "processed_mask",
            "pointcloud",
            "category",
        ]
        missing_columns = [col for col in required_columns if col not in items.columns]
        if missing_columns:
            raise KeyError(f"Split file is missing columns: {missing_columns}")

        file_columns = [
            "processed_image",
            "processed_mask",
            "pointcloud",
        ]

        if require_files:
            items = _resolve_existing_flags(
                items=items,
                processed_dir=self.processed_dir,
                file_columns=file_columns,
            )

            if not allow_missing_files:
                _raise_if_missing_files(
                    items=items,
                    split=split,
                    processed_dir=self.processed_dir,
                    file_columns=file_columns,
                )

            before_filter_count = len(items)

            keep_mask = True
            for column in file_columns:
                keep_mask = keep_mask & items[f"_{column}_exists"]

            items = items[keep_mask].copy()
            after_filter_count = len(items)

            if before_filter_count != after_filter_count:
                print(
                    f"Warning: split='{split}' filtered missing files: "
                    f"{before_filter_count} -> {after_filter_count}",
                    flush=True,
                )

        if len(items) == 0:
            raise ValueError(
                f"No usable samples left in split='{split}'. "
                f"original={original_count}, after_category_filter={after_category_count}. "
                f"Check processed_dir={self.processed_dir}"
            )

        if max_samples is not None:
            items = items.head(max_samples).copy()

        internal_cols = [col for col in items.columns if col.startswith("_") and col.endswith("_exists")]
        if internal_cols:
            items = items.drop(columns=internal_cols)

        self.items = items.reset_index(drop=True)

        print(
            f"Loaded ProcessedPix3DDataset split='{split}': "
            f"original={original_count}, "
            f"after_category_filter={after_category_count}, "
            f"final={len(self.items)}, "
            f"processed_dir={self.processed_dir}",
            flush=True,
        )
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items.iloc[idx]

        image_path = self.processed_dir / str(item["processed_image"])
        mask_path = self.processed_dir / str(item["processed_mask"])
        pointcloud_path = self.processed_dir / str(item["pointcloud"])

        image = Pix3DDataset._load_image_safely(image_path, mode="RGB")
        image_tensor = Pix3DDataset._to_tensor(image)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        if self.use_mask_channel:
            mask = Pix3DDataset._load_image_safely(mask_path, mode="L")
            mask_tensor = Pix3DDataset._mask_to_tensor(mask)
            image_tensor = torch.cat([image_tensor, mask_tensor], dim=0)

        points_np = np.load(pointcloud_path).astype(np.float32)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            raise RuntimeError(f"Invalid point cloud shape {points_np.shape}: {pointcloud_path}")
        if self.expected_num_points is not None and points_np.shape[0] != self.expected_num_points:
            raise RuntimeError(
                f"Point cloud artifact has {points_np.shape[0]} points but training expects "
                f"{self.expected_num_points}: {pointcloud_path}. Regenerate processed point clouds."
            )
        points_gt = torch.from_numpy(points_np)

        sample = {
            "image": image_tensor,
            "category": item["category"],
            "points_gt": points_gt,
            "pointcloud_path": str(pointcloud_path),
            "image_path": str(image_path),
        }
        if "sample_id" in item:
            sample["sample_id"] = str(item["sample_id"])
        if "model_uid" in item:
            sample["model_uid"] = str(item["model_uid"])
        if "cad_uid" in item:
            sample["cad_uid"] = str(item["cad_uid"])
        if "model" in item:
            sample["model_path"] = str(item["model"])

        return sample


