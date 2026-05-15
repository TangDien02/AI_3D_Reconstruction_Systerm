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
    ):
        self.root_dir = Path(root_dir)
        self.categories = set(categories) if categories else None
        self.split = split
        self.image_size = image_size
        self.num_points = num_points
        self.transform = transform

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


class ProcessedPix3DDataset(Dataset):
    def __init__(
        self,
        processed_dir,
        split="train",
        categories=None,
        max_samples=None,
        transform=None,
        require_files=True,
    ):
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.categories = set(categories) if categories else None
        self.transform = transform

        split_path = self.processed_dir / "splits" / f"{split}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        items = pd.read_csv(split_path)
        if self.categories:
            items = items[items["category"].isin(self.categories)]

        required_columns = ["processed_image", "pointcloud", "category"]
        missing_columns = [col for col in required_columns if col not in items.columns]
        if missing_columns:
            raise KeyError(f"Split file is missing columns: {missing_columns}")

        if require_files:
            items = items[
                items["processed_image"].apply(lambda path: (self.processed_dir / str(path)).is_file())
                & items["pointcloud"].apply(lambda path: (self.processed_dir / str(path)).is_file())
            ]

        if max_samples is not None:
            items = items.head(max_samples)

        self.items = items.reset_index(drop=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items.iloc[idx]
        image_path = self.processed_dir / str(item["processed_image"])
        pointcloud_path = self.processed_dir / str(item["pointcloud"])

        image = Pix3DDataset._load_image_safely(image_path, mode="RGB")
        image_tensor = Pix3DDataset._to_tensor(image)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        points_gt = torch.from_numpy(np.load(pointcloud_path).astype(np.float32))

        sample = {
            "image": image_tensor,
            "category": item["category"],
            "points_gt": points_gt,
            "pointcloud_path": str(pointcloud_path),
            "image_path": str(image_path),
        }
        if "model" in item:
            sample["model_path"] = str(item["model"])
        return sample


