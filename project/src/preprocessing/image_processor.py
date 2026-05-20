from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from src.utils.logger import get_logger


logger = get_logger("ImagePreprocessor")


@dataclass(frozen=True)
class SceneObject:
    """Object-level record produced by scene preparation."""

    object_id: str
    bbox: tuple[int, int, int, int]
    label: str = "unknown"
    confidence: float = 0.0
    mask_path: str | None = None
    crop_path: str | None = None
    source: str = "metadata"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenePreparationResult:
    scene_id: str
    image_path: str
    objects: list[SceneObject]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "image_path": self.image_path,
            "objects": [obj.to_dict() for obj in self.objects],
        }


@dataclass(frozen=True)
class ObjectTensorSample:
    object_id: str
    image: torch.Tensor
    mask: torch.Tensor | None
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float
    domain_label: str = "unknown"


@dataclass(frozen=True)
class ScenePreparationConfig:
    image_size: int = 224
    crop_padding_ratio: float = 0.10
    min_confidence: float = 0.25
    min_box_area_ratio: float = 0.0005
    background_value: int = 255
    use_yolo_fallback: bool = True
    yolo_model_name: str = "yolov8n-seg.pt"


class StrongAugmentation:
    """Lightweight PIL/torch augmentation without adding torchvision as a hard dependency."""

    def __init__(
        self,
        brightness: float = 0.25,
        contrast: float = 0.25,
        noise_std: float = 0.03,
        blur_radius: float = 1.2,
        erase_prob: float = 0.15,
        seed: int | None = None,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std
        self.blur_radius = blur_radius
        self.erase_prob = erase_prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.brightness > 0:
            factor = float(self.rng.uniform(1 - self.brightness, 1 + self.brightness))
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast > 0:
            factor = float(self.rng.uniform(1 - self.contrast, 1 + self.contrast))
            image = ImageEnhance.Contrast(image).enhance(factor)
        if self.blur_radius > 0 and self.rng.random() < 0.35:
            radius = float(self.rng.uniform(0.1, self.blur_radius))
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        arr = np.asarray(image).astype(np.float32) / 255.0
        if self.noise_std > 0:
            arr = arr + self.rng.normal(0.0, self.noise_std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)

        if self.erase_prob > 0 and self.rng.random() < self.erase_prob:
            height, width = arr.shape[:2]
            erase_h = max(1, int(height * float(self.rng.uniform(0.05, 0.18))))
            erase_w = max(1, int(width * float(self.rng.uniform(0.05, 0.18))))
            top = int(self.rng.integers(0, max(height - erase_h + 1, 1)))
            left = int(self.rng.integers(0, max(width - erase_w + 1, 1)))
            arr[top : top + erase_h, left : left + erase_w] = 1.0

        return Image.fromarray((arr * 255).astype(np.uint8))


class ImagePreprocessor:
    """Scene-level and object-level preparation for reconstruction pipelines."""

    def __init__(self, config: ScenePreparationConfig | None = None):
        self.config = config or ScenePreparationConfig()
        self.yolo_model = None
        if self.config.use_yolo_fallback:
            self._try_load_yolo()

    def _try_load_yolo(self) -> None:
        try:
            from ultralytics import YOLO

            self.yolo_model = YOLO(self.config.yolo_model_name)
            logger.info("Loaded YOLO fallback model: %s", self.config.yolo_model_name)
        except Exception as exc:
            logger.warning("YOLO fallback unavailable: %s", exc)

    def get_class_from_pre_data(self, image_id: str | None, metadata_dict: dict[str, Any] | None) -> str | None:
        if image_id and metadata_dict and image_id in metadata_dict:
            value = metadata_dict[image_id]
            if isinstance(value, dict):
                return value.get("label") or value.get("category")
            return str(value)
        return None

    def process(self, image_path: str | Path, pre_data_class: str | None = None) -> str | None:
        """Backward-compatible class lookup used by the old template baseline."""

        if pre_data_class:
            logger.info("Using class from metadata: %s", pre_data_class)
            return pre_data_class

        objects = self.detect_objects(image_path)
        if not objects:
            return None
        return objects[0].label if objects[0].label != "unknown" else None

    def prepare_scene(
        self,
        image_path: str | Path,
        scene_id: str | None = None,
        metadata_objects: list[dict[str, Any]] | None = None,
        output_dir: str | Path | None = None,
    ) -> ScenePreparationResult:
        image_path = Path(image_path)
        scene_id = scene_id or image_path.stem
        objects = self.objects_from_metadata(metadata_objects) if metadata_objects else self.detect_objects(image_path)

        if output_dir is not None:
            output_dir = Path(output_dir)
            saved_objects = []
            for obj in objects:
                crop_path, mask_path = self.save_object_crop(image_path, obj, output_dir, scene_id)
                saved_objects.append(
                    SceneObject(
                        object_id=obj.object_id,
                        bbox=obj.bbox,
                        label=obj.label,
                        confidence=obj.confidence,
                        mask_path=str(mask_path) if mask_path else obj.mask_path,
                        crop_path=str(crop_path),
                        source=obj.source,
                    )
                )
            objects = saved_objects

        return ScenePreparationResult(scene_id=scene_id, image_path=str(image_path), objects=objects)

    def objects_from_metadata(self, records: list[dict[str, Any]] | None) -> list[SceneObject]:
        objects: list[SceneObject] = []
        for index, record in enumerate(records or []):
            bbox = self._coerce_bbox(record.get("bbox"))
            if bbox is None:
                continue
            label = str(record.get("label") or record.get("category") or "unknown")
            confidence = float(record.get("confidence", 1.0))
            objects.append(
                SceneObject(
                    object_id=str(record.get("object_id") or f"obj_{index:03d}"),
                    bbox=bbox,
                    label=label,
                    confidence=confidence,
                    mask_path=record.get("mask_path"),
                    crop_path=record.get("crop_path"),
                    source="metadata",
                )
            )
        return objects

    def detect_objects(self, image_path: str | Path) -> list[SceneObject]:
        if self.yolo_model is None:
            return []

        image_path = Path(image_path)
        results = self.yolo_model(str(image_path), verbose=False)
        if not results:
            return []

        image = self._load_image(image_path, mode="RGB")
        image_area = image.width * image.height
        names = getattr(self.yolo_model, "names", {})
        objects: list[SceneObject] = []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return objects

        for index, box in enumerate(boxes):
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            bbox = self._coerce_bbox(xyxy)
            if bbox is None:
                continue
            confidence = float(box.conf[0].detach().cpu().item()) if getattr(box, "conf", None) is not None else 0.0
            if confidence < self.config.min_confidence:
                continue
            if self._bbox_area_ratio(bbox, image_area) < self.config.min_box_area_ratio:
                continue

            class_id = int(box.cls[0].detach().cpu().item()) if getattr(box, "cls", None) is not None else -1
            label = str(names.get(class_id, "unknown")) if isinstance(names, dict) else "unknown"
            objects.append(
                SceneObject(
                    object_id=f"{image_path.stem}_obj_{index:03d}",
                    bbox=bbox,
                    label=label,
                    confidence=confidence,
                    source="yolo",
                )
            )
        return objects

    def save_object_crop(
        self,
        image_path: str | Path,
        obj: SceneObject,
        output_dir: str | Path,
        scene_id: str,
    ) -> tuple[Path, Path | None]:
        output_dir = Path(output_dir)
        crop_dir = output_dir / "crops"
        mask_dir = output_dir / "masks"
        crop_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        image = self._load_image(image_path, mode="RGB")
        mask = self._load_image(obj.mask_path, mode="L") if obj.mask_path else None
        crop, crop_mask = self.crop_object(image, obj.bbox, mask=mask)

        crop_path = crop_dir / f"{scene_id}_{obj.object_id}.png"
        crop.save(crop_path)

        mask_path = None
        if crop_mask is not None:
            mask_path = mask_dir / f"{scene_id}_{obj.object_id}.png"
            crop_mask.save(mask_path)
        return crop_path, mask_path

    def prepare_object_tensor(
        self,
        image_path: str | Path,
        obj: SceneObject,
        domain_label: str = "unknown",
        augment: StrongAugmentation | None = None,
    ) -> ObjectTensorSample:
        image = self._load_image(obj.crop_path or image_path, mode="RGB")
        mask = self._load_image(obj.mask_path, mode="L") if obj.mask_path else None

        if obj.crop_path is None:
            image, mask = self.crop_object(image, obj.bbox, mask=mask)
        if mask is not None:
            image = self.apply_mask(image, mask)

        image = self.letterbox(image, self.config.image_size)
        mask_tensor = None
        if mask is not None:
            mask = self.letterbox(
                mask,
                self.config.image_size,
                fill_value=0,
                resample=Image.Resampling.NEAREST,
            )
            mask_tensor = self.mask_to_tensor(mask)

        if augment is not None:
            image = augment(image)

        return ObjectTensorSample(
            object_id=obj.object_id,
            image=self.image_to_tensor(image),
            mask=mask_tensor,
            label=obj.label,
            bbox=obj.bbox,
            confidence=obj.confidence,
            domain_label=domain_label,
        )

    def crop_object(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        mask: Image.Image | None = None,
    ) -> tuple[Image.Image, Image.Image | None]:
        padded = self.pad_bbox(bbox, image.size, self.config.crop_padding_ratio)
        crop = image.crop(padded)
        crop_mask = mask.resize(image.size, Image.Resampling.NEAREST).crop(padded) if mask is not None else None
        return crop, crop_mask

    def pad_bbox(
        self,
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
        padding_ratio: float,
    ) -> tuple[int, int, int, int]:
        width, height = image_size
        x1, y1, x2, y2 = bbox
        pad_x = int(round((x2 - x1) * padding_ratio))
        pad_y = int(round((y2 - y1) * padding_ratio))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

    def apply_mask(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image_np = np.asarray(image).astype(np.uint8)
        mask_np = np.asarray(mask.resize(image.size, Image.Resampling.NEAREST)) > 0
        masked = np.full_like(image_np, self.config.background_value)
        masked[mask_np] = image_np[mask_np]
        return Image.fromarray(masked)

    @staticmethod
    def letterbox(
        image: Image.Image,
        image_size: int,
        fill_value: int = 255,
        resample: int | Image.Resampling = Image.Resampling.BILINEAR,
    ) -> Image.Image:
        width, height = image.size
        scale = min(image_size / width, image_size / height)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized = image.resize(new_size, resample)
        if image.mode == "L":
            canvas = Image.new("L", (image_size, image_size), color=fill_value)
        else:
            canvas = Image.new(image.mode, (image_size, image_size), color=(fill_value,) * len(image.getbands()))
        offset = ((image_size - new_size[0]) // 2, (image_size - new_size[1]) // 2)
        canvas.paste(resized, offset)
        return canvas

    @staticmethod
    def image_to_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    @staticmethod
    def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
        arr = (np.asarray(mask.convert("L")).astype(np.float32) / 255.0)[None, ...]
        return torch.from_numpy(arr)

    @staticmethod
    def _load_image(image_path: str | Path, mode: str) -> Image.Image:
        image = Image.open(image_path)
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        return image.convert(mode)

    @staticmethod
    def _coerce_bbox(value: Any) -> tuple[int, int, int, int] | None:
        if value is None:
            return None
        if isinstance(value, str):
            import ast

            value = ast.literal_eval(value)
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _bbox_area_ratio(bbox: tuple[int, int, int, int], image_area: int) -> float:
        x1, y1, x2, y2 = bbox
        return ((x2 - x1) * (y2 - y1)) / max(image_area, 1)
