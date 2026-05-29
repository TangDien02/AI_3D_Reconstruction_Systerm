from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from threading import Lock
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

app = FastAPI(title="3DRecon API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVER_DIR = Path(__file__).resolve().parent
REPO_DIR = SERVER_DIR.parent
PROJECT_DIR = REPO_DIR / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

UPLOAD_DIR = SERVER_DIR / "uploads"
MODEL_OUTPUT_DIR = SERVER_DIR / "models"
SEGMENT_OUTPUT_DIR = SERVER_DIR / "segment_outputs"
YOLO_WEIGHTS = SERVER_DIR / "weights" / "yolo26n-seg.pt"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DETECTION_CONFIDENCE = env_float("YOLO_DETECTION_CONFIDENCE", 0.20)
DETECTION_IMAGE_SIZE = env_int("YOLO_DETECTION_IMAGE_SIZE", 640)
DETECTION_MAX_OBJECTS = env_int("YOLO_DETECTION_MAX_OBJECTS", 20)
DETECTION_IOU = env_float("YOLO_DETECTION_IOU", 0.45)
MODEL_INPUT_IMAGE_SIZE = env_int("RECON_MODEL_INPUT_IMAGE_SIZE", 224)
MODEL_INPUT_MARGIN_RATIO = env_float("RECON_MODEL_INPUT_MARGIN_RATIO", 0.08)
MODEL_INPUT_MIN_MARGIN_PX = env_int("RECON_MODEL_INPUT_MIN_MARGIN_PX", 8)
TRIPOSR_MODEL_NAME_OR_PATH = os.environ.get("TRIPOSR_MODEL_NAME_OR_PATH", "stabilityai/TripoSR")
TRIPOSR_REPO_DIR = os.environ.get("TRIPOSR_REPO_DIR")
TRIPOSR_DEVICE = os.environ.get("TRIPOSR_DEVICE", "auto")
TRIPOSR_CHUNK_SIZE = env_int("TRIPOSR_CHUNK_SIZE", 8192)
TRIPOSR_MC_RESOLUTION = env_int("TRIPOSR_MC_RESOLUTION", 256)
TRIPOSR_FOREGROUND_RATIO = env_float("TRIPOSR_FOREGROUND_RATIO", 0.85)
TRIPOSR_NUM_POINTS = env_int("TRIPOSR_NUM_POINTS", 2048)
TRIPOSR_MODEL_SAVE_FORMAT = os.environ.get("TRIPOSR_MODEL_SAVE_FORMAT", "glb").lower()
TRIPOSR_REMOVE_BACKGROUND = env_bool("TRIPOSR_REMOVE_BACKGROUND", True)
TRIPOSR_SAVE_PREVIEW = env_bool("TRIPOSR_SAVE_PREVIEW", True)
TRIPOSR_CROP_MARGIN_RATIO = env_float("TRIPOSR_CROP_MARGIN_RATIO", 0.16)
TRIPOSR_CROP_MIN_MARGIN_PX = env_int("TRIPOSR_CROP_MIN_MARGIN_PX", 16)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEGMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/segment-outputs", StaticFiles(directory=SEGMENT_OUTPUT_DIR), name="segment_outputs")
app.mount("/models", StaticFiles(directory=MODEL_OUTPUT_DIR), name="models")

_yolo_model = None
_yolo_model_lock = Lock()
_triposr_core = None
_triposr_core_lock = Lock()


def get_yolo_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    return "0" if torch.cuda.is_available() else "cpu"


def get_yolo_model():
    global _yolo_model

    if _yolo_model is not None:
        return _yolo_model

    if not YOLO_WEIGHTS.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"YOLO weights not found: {YOLO_WEIGHTS}",
        )

    with _yolo_model_lock:
        if _yolo_model is None:
            try:
                from ultralytics import YOLO
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"YOLO dependencies are not ready: {exc}",
                ) from exc
            _yolo_model = YOLO(str(YOLO_WEIGHTS))
            try:
                _yolo_model.fuse()
            except Exception:
                pass
    return _yolo_model


def build_triposr_config():
    try:
        from src.reconstruction.triposr_runner import TripoSRConfig
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"TripoSR core is not importable: {exc}",
        ) from exc

    return TripoSRConfig(
        model_name_or_path=TRIPOSR_MODEL_NAME_OR_PATH,
        triposr_repo_dir=TRIPOSR_REPO_DIR,
        device=TRIPOSR_DEVICE,
        chunk_size=TRIPOSR_CHUNK_SIZE,
        mc_resolution=TRIPOSR_MC_RESOLUTION,
        foreground_ratio=TRIPOSR_FOREGROUND_RATIO,
        remove_background=TRIPOSR_REMOVE_BACKGROUND,
        num_points=TRIPOSR_NUM_POINTS,
        model_save_format=TRIPOSR_MODEL_SAVE_FORMAT,
        normalize_points=True,
    )


def get_triposr_core():
    global _triposr_core

    if _triposr_core is not None:
        return _triposr_core

    with _triposr_core_lock:
        if _triposr_core is None:
            try:
                from src.reconstruction.triposr_runner import TripoSRCore
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"TripoSR dependencies are not ready: {exc}",
                ) from exc
            try:
                _triposr_core = TripoSRCore(config=build_triposr_config())
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"TripoSR init failed: {exc}") from exc
    return _triposr_core


def clamp_bbox_xyxy(
    xyxy: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), float(image_width)))
    y1 = max(0.0, min(float(y1), float(image_height)))
    x2 = max(x1, min(float(x2), float(image_width)))
    y2 = max(y1, min(float(y2), float(image_height)))
    return x1, y1, x2, y2


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def union_bbox_xyxy(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if box_b is None:
        return box_a
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)


def expand_bbox_xyxy(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin_ratio: float = MODEL_INPUT_MARGIN_RATIO,
    min_margin_px: int = MODEL_INPUT_MIN_MARGIN_PX,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    margin_ratio = max(0.0, min(float(margin_ratio), 0.40))
    margin_x = max(float(min_margin_px), width * margin_ratio)
    margin_y = max(float(min_margin_px), height * margin_ratio)
    return clamp_bbox_xyxy(
        [x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y],
        image_width=image_width,
        image_height=image_height,
    )


def bbox_to_crop_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    crop_x1 = max(0, min(int(math.floor(x1)), image_width - 1))
    crop_y1 = max(0, min(int(math.floor(y1)), image_height - 1))
    crop_x2 = max(crop_x1 + 1, min(int(math.ceil(x2)), image_width))
    crop_y2 = max(crop_y1 + 1, min(int(math.ceil(y2)), image_height))
    return crop_x1, crop_y1, crop_x2, crop_y2


def crop_box_payload(crop_box: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = crop_box
    return {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
    }


def make_mask_from_polygon(
    polygon: list,
    image_width: int,
    image_height: int,
    fallback_bbox: tuple[float, float, float, float],
) -> Image.Image:
    mask = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)

    if polygon is not None and len(polygon) >= 3:
        points = [(float(x), float(y)) for x, y in polygon]
        draw.polygon(points, fill=255)
    else:
        x1, y1, x2, y2 = fallback_bbox
        draw.rectangle((x1, y1, x2, y2), fill=255)

    return mask


def safe_slug(value: object, default: str = "object") -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip().lower()).strip("-")
    return (text[:48] or default).strip("-") or default


def build_job_id(label: object = "object") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{safe_slug(label)}_{uuid.uuid4().hex[:8]}"


def job_output_dir(root_dir: Path, job_id: str) -> Path:
    date_part = job_id[:8] if len(job_id) >= 8 else datetime.now().strftime("%Y%m%d")
    return root_dir / date_part / job_id


def mounted_url(root_dir: Path, mount_path: str, path: Path) -> str:
    relative_path = Path(path).relative_to(root_dir).as_posix()
    return f"{mount_path}/{relative_path}"


def to_relative_url(path: Path) -> str:
    return mounted_url(SEGMENT_OUTPUT_DIR, "/segment-outputs", path)


def to_model_url(path: Path) -> str:
    return mounted_url(MODEL_OUTPUT_DIR, "/models", path)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


async def read_upload_image(image: UploadFile) -> Image.Image:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc


def detect_and_select_object(
    pil_image: Image.Image,
    object_id: int | None = None,
    bbox_x: float | None = None,
    bbox_y: float | None = None,
    bbox_width: float | None = None,
    bbox_height: float | None = None,
):
    image_width, image_height = pil_image.size
    model = get_yolo_model()

    try:
        results = model.predict(
            pil_image,
            conf=DETECTION_CONFIDENCE,
            imgsz=DETECTION_IMAGE_SIZE,
            max_det=DETECTION_MAX_OBJECTS,
            iou=DETECTION_IOU,
            device=get_yolo_device(),
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YOLO segmentation failed: {exc}") from exc

    result = results[0] if results else None
    if result is None or result.boxes is None or len(result.boxes) == 0:
        raise HTTPException(status_code=404, detail="No object detected.")

    detections = []
    for index, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = clamp_bbox_xyxy(
            [float(value) for value in box.xyxy[0].tolist()],
            image_width=image_width,
            image_height=image_height,
        )
        detections.append(
            {
                "index": index,
                "label": model.names.get(cls_id, str(cls_id)),
                "confidence": confidence,
                "bbox_xyxy": (x1, y1, x2, y2),
            }
        )

    selected_bbox = None
    if None not in (bbox_x, bbox_y, bbox_width, bbox_height):
        selected_bbox = clamp_bbox_xyxy(
            [
                float(bbox_x),
                float(bbox_y),
                float(bbox_x + bbox_width),
                float(bbox_y + bbox_height),
            ],
            image_width=image_width,
            image_height=image_height,
        )

    if selected_bbox is not None:
        selected_detection = max(
            detections,
            key=lambda detection: bbox_iou(detection["bbox_xyxy"], selected_bbox),
        )
    elif object_id is not None and 0 <= object_id < len(detections):
        selected_detection = detections[object_id]
    else:
        selected_detection = max(detections, key=lambda detection: detection["confidence"])

    return result, selected_detection, detections


def compose_masked_crop(
    pil_image: Image.Image,
    full_mask: Image.Image,
    crop_box: tuple[int, int, int, int],
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    crop = pil_image.crop(crop_box)
    mask_crop = full_mask.crop(crop_box)
    masked_crop = Image.new("RGB", crop.size, (255, 255, 255))
    masked_crop.paste(crop, mask=mask_crop)
    transparent_crop = crop.convert("RGBA")
    transparent_crop.putalpha(mask_crop)
    return crop, mask_crop, masked_crop, transparent_crop


def square_pad_image(
    image: Image.Image,
    fill: tuple[int, int, int] | int = (255, 255, 255),
) -> tuple[Image.Image, dict]:
    width, height = image.size
    side = max(width, height)
    if image.mode == "L":
        padded = Image.new("L", (side, side), int(fill) if isinstance(fill, int) else 0)
    else:
        padded = Image.new(image.mode, (side, side), fill)
    left = (side - width) // 2
    top = (side - height) // 2
    padded.paste(image, (left, top))
    padding = {
        "left": left,
        "top": top,
        "right": side - width - left,
        "bottom": side - height - top,
    }
    return padded, padding


def build_segment_model_input(
    pil_image: Image.Image,
    full_mask: Image.Image,
    selected_xyxy: tuple[float, float, float, float],
) -> tuple[Image.Image, Image.Image, dict]:
    image_width, image_height = pil_image.size
    mask_bbox = full_mask.getbbox()
    base_bbox = union_bbox_xyxy(
        selected_xyxy,
        tuple(float(value) for value in mask_bbox) if mask_bbox else None,
    )
    expanded_bbox = expand_bbox_xyxy(
        base_bbox,
        image_width=image_width,
        image_height=image_height,
    )
    model_crop_box = bbox_to_crop_box(expanded_bbox, image_width, image_height)
    _, model_mask_crop, model_masked_crop, _ = compose_masked_crop(pil_image, full_mask, model_crop_box)
    padded_image, padding = square_pad_image(model_masked_crop, fill=(255, 255, 255))
    padded_mask, _ = square_pad_image(model_mask_crop, fill=0)
    model_input = padded_image.resize(
        (MODEL_INPUT_IMAGE_SIZE, MODEL_INPUT_IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )
    model_input_mask = padded_mask.resize(
        (MODEL_INPUT_IMAGE_SIZE, MODEL_INPUT_IMAGE_SIZE),
        Image.Resampling.NEAREST,
    )
    metadata = {
        "mode": "segmented_mask_crop_square_pad",
        "image_size": MODEL_INPUT_IMAGE_SIZE,
        "background": "white",
        "margin_ratio": MODEL_INPUT_MARGIN_RATIO,
        "min_margin_px": MODEL_INPUT_MIN_MARGIN_PX,
        "base_bbox": crop_box_payload(bbox_to_crop_box(base_bbox, image_width, image_height)),
        "model_crop_bbox": crop_box_payload(model_crop_box),
        "square_padding": padding,
    }
    return model_input, model_input_mask, metadata


def build_triposr_crop_input(
    pil_image: Image.Image,
    selected_xyxy: tuple[float, float, float, float],
) -> tuple[Image.Image, dict]:
    image_width, image_height = pil_image.size
    expanded_bbox = expand_bbox_xyxy(
        selected_xyxy,
        image_width=image_width,
        image_height=image_height,
        margin_ratio=TRIPOSR_CROP_MARGIN_RATIO,
        min_margin_px=TRIPOSR_CROP_MIN_MARGIN_PX,
    )
    crop_box = bbox_to_crop_box(expanded_bbox, image_width, image_height)
    crop = pil_image.crop(crop_box).convert("RGB")
    metadata = {
        "mode": "bbox_crop_for_triposr_rembg",
        "crop_strategy": "yolo_bbox_with_margin",
        "background_handling": "triposr_rembg" if TRIPOSR_REMOVE_BACKGROUND else "input_background",
        "margin_ratio": TRIPOSR_CROP_MARGIN_RATIO,
        "min_margin_px": TRIPOSR_CROP_MIN_MARGIN_PX,
        "base_bbox": crop_box_payload(bbox_to_crop_box(selected_xyxy, image_width, image_height)),
        "triposr_crop_bbox": crop_box_payload(crop_box),
    }
    return crop, metadata


def build_plain_model_input(pil_image: Image.Image) -> tuple[Image.Image, dict]:
    padded_image, padding = square_pad_image(pil_image.convert("RGB"), fill=(255, 255, 255))
    model_input = padded_image.resize(
        (MODEL_INPUT_IMAGE_SIZE, MODEL_INPUT_IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )
    metadata = {
        "mode": "plain_square_pad",
        "image_size": MODEL_INPUT_IMAGE_SIZE,
        "background": "white",
        "square_padding": padding,
    }
    return model_input, metadata


def save_segment_artifacts(
    pil_image: Image.Image,
    result,
    selected_detection: dict,
    job_id: str,
) -> tuple[dict, Path]:
    image_width, image_height = pil_image.size
    selected_index = selected_detection["index"]
    selected_xyxy = selected_detection["bbox_xyxy"]
    x1, y1, x2, y2 = selected_xyxy
    crop_box = bbox_to_crop_box(selected_xyxy, image_width, image_height)

    polygon = None
    if result.masks is not None and result.masks.xy is not None and selected_index < len(result.masks.xy):
        polygon = result.masks.xy[selected_index]
    full_mask = make_mask_from_polygon(polygon, image_width, image_height, selected_xyxy)

    crop, mask_crop, masked_crop, transparent_crop = compose_masked_crop(pil_image, full_mask, crop_box)
    triposr_crop, triposr_crop_metadata = build_triposr_crop_input(
        pil_image=pil_image,
        selected_xyxy=selected_xyxy,
    )
    model_input, model_input_mask, model_input_metadata = build_segment_model_input(
        pil_image=pil_image,
        full_mask=full_mask,
        selected_xyxy=selected_xyxy,
    )

    overlay = pil_image.convert("RGBA")
    green = Image.new("RGBA", pil_image.size, (163, 230, 53, 95))
    overlay = Image.composite(green, overlay, full_mask).convert("RGB")
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(crop_box, outline=(163, 230, 53), width=4)

    segment_dir = job_output_dir(SEGMENT_OUTPUT_DIR, job_id)
    segment_dir.mkdir(parents=True, exist_ok=True)
    original_path = segment_dir / "original.jpg"
    mask_path = segment_dir / "mask.png"
    crop_path = segment_dir / "crop.jpg"
    triposr_crop_path = segment_dir / "triposr_crop.jpg"
    masked_crop_path = segment_dir / "masked_crop.png"
    transparent_crop_path = segment_dir / "transparent_crop.png"
    model_input_path = segment_dir / "model_input.png"
    model_input_mask_path = segment_dir / "model_input_mask.png"
    overlay_path = segment_dir / "overlay.jpg"

    pil_image.save(original_path, quality=92)
    full_mask.save(mask_path)
    crop.save(crop_path, quality=92)
    triposr_crop.save(triposr_crop_path, quality=94)
    masked_crop.save(masked_crop_path)
    transparent_crop.save(transparent_crop_path)
    model_input.save(model_input_path)
    model_input_mask.save(model_input_mask_path)
    overlay.save(overlay_path, quality=92)

    selected_response = {
        "id": str(selected_index),
        "label": selected_detection["label"],
        "confidence": round(selected_detection["confidence"], 4),
        "bbox": {
            "x": round(x1, 2),
            "y": round(y1, 2),
            "width": round(x2 - x1, 2),
            "height": round(y2 - y1, 2),
        },
    }

    segment_payload = {
        "selected": selected_response,
        "output_dir": str(segment_dir),
        "files": {
            "original": to_relative_url(original_path),
            "mask": to_relative_url(mask_path),
            "crop": to_relative_url(crop_path),
            "triposr_crop": to_relative_url(triposr_crop_path),
            "masked_crop": to_relative_url(masked_crop_path),
            "transparent_crop": to_relative_url(transparent_crop_path),
            "model_input": to_relative_url(model_input_path),
            "model_input_mask": to_relative_url(model_input_mask_path),
            "overlay": to_relative_url(overlay_path),
        },
        "paths": {
            "output_dir": str(segment_dir),
            "original": str(original_path),
            "mask": str(mask_path),
            "crop": str(crop_path),
            "triposr_crop": str(triposr_crop_path),
            "masked_crop": str(masked_crop_path),
            "transparent_crop": str(transparent_crop_path),
            "model_input": str(model_input_path),
            "model_input_mask": str(model_input_mask_path),
            "overlay": str(overlay_path),
        },
        "preprocessing": {
            "legacy_model_input": model_input_metadata,
            "triposr_input": triposr_crop_metadata,
        },
    }
    write_json(segment_dir / "segment_summary.json", segment_payload)
    return segment_payload, triposr_crop_path


def save_reconstruction_artifacts(input_path: Path, job_id: str, label: str | None = None) -> dict:
    try:
        from src.reconstruction.triposr_runner import TripoSRDependencyError
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"TripoSR core dependencies are not ready: {exc}",
        ) from exc

    try:
        result = get_triposr_core().reconstruct_image(
            image_path=input_path,
            output_dir=MODEL_OUTPUT_DIR,
            name=job_id,
            save_preview=TRIPOSR_SAVE_PREVIEW,
        )
    except TripoSRDependencyError as exc:
        raise HTTPException(status_code=503, detail=f"TripoSR dependencies are not ready: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TripoSR reconstruction failed: {exc}") from exc

    mesh_format = result.summary.get("mesh", {}).get("format", TRIPOSR_MODEL_SAVE_FORMAT)
    mesh_url = to_model_url(result.mesh_path)
    colored_mesh_url = to_model_url(result.colored_mesh_ply_path) if result.colored_mesh_ply_path else None
    preview_url = to_model_url(result.preview_path) if result.preview_path else None

    payload = {
        "job_id": job_id,
        "label": label,
        "backend": "triposr",
        "output_dir": str(result.output_dir),
        "input_image": str(input_path),
        "num_points": int(result.points.shape[0]),
        "model": {
            "name": TRIPOSR_MODEL_NAME_OR_PATH,
            "type": "triposr",
            "device": result.summary.get("runtime", {}).get("device"),
            "remove_background": TRIPOSR_REMOVE_BACKGROUND,
            "mc_resolution": TRIPOSR_MC_RESOLUTION,
            "foreground_ratio": TRIPOSR_FOREGROUND_RATIO,
        },
        "mesh": result.summary.get("mesh", {}),
        "files": {
            "input_image": to_model_url(result.input_path),
            "triposr_input": to_model_url(result.processed_input_path),
            "pointcloud_npy": to_model_url(result.pointcloud_npy_path),
            "pointcloud_ply": to_model_url(result.pointcloud_ply_path),
            "mesh": mesh_url,
            "mesh_glb": mesh_url if mesh_format == "glb" else None,
            "mesh_obj": mesh_url if mesh_format == "obj" else None,
            "mesh_colored_ply": colored_mesh_url,
            "preview_png": preview_url,
            "summary_json": to_model_url(result.output_dir / "reconstruction_summary.json"),
            "triposr_summary_json": to_model_url(result.summary_path),
        },
        "paths": {
            "output_dir": str(result.output_dir),
            "input_image": str(input_path),
            "triposr_input": str(result.processed_input_path),
            "pointcloud_npy": str(result.pointcloud_npy_path),
            "pointcloud_ply": str(result.pointcloud_ply_path),
            "mesh": str(result.mesh_path),
            "mesh_glb": str(result.mesh_path) if mesh_format == "glb" else None,
            "mesh_obj": str(result.mesh_path) if mesh_format == "obj" else None,
            "mesh_colored_ply": str(result.colored_mesh_ply_path) if result.colored_mesh_ply_path else None,
            "preview_png": str(result.preview_path) if result.preview_path else None,
            "summary_json": str(result.output_dir / "reconstruction_summary.json"),
            "triposr_summary_json": str(result.summary_path),
        },
    }
    write_json(result.output_dir / "reconstruction_summary.json", payload)
    return payload


@app.on_event("startup")
def warmup_yolo_model():
    try:
        model = get_yolo_model()
        warmup_image = Image.new("RGB", (DETECTION_IMAGE_SIZE, DETECTION_IMAGE_SIZE), (0, 0, 0))
        model.predict(
            warmup_image,
            conf=DETECTION_CONFIDENCE,
            imgsz=DETECTION_IMAGE_SIZE,
            max_det=DETECTION_MAX_OBJECTS,
            iou=DETECTION_IOU,
            device=get_yolo_device(),
            verbose=False,
        )
    except Exception as exc:
        print(f"YOLO warmup skipped: {exc}")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "yolo_weights_exists": YOLO_WEIGHTS.is_file(),
        "yolo_device": get_yolo_device(),
        "detector": {
            "imgsz": DETECTION_IMAGE_SIZE,
            "conf": DETECTION_CONFIDENCE,
            "max_det": DETECTION_MAX_OBJECTS,
            "iou": DETECTION_IOU,
        },
        "reconstruction": {
            "backend": "triposr",
            "model_name_or_path": TRIPOSR_MODEL_NAME_OR_PATH,
            "repo_dir": TRIPOSR_REPO_DIR,
            "device": TRIPOSR_DEVICE,
            "loaded": _triposr_core is not None,
            "remove_background": TRIPOSR_REMOVE_BACKGROUND,
            "mc_resolution": TRIPOSR_MC_RESOLUTION,
            "foreground_ratio": TRIPOSR_FOREGROUND_RATIO,
            "num_points": TRIPOSR_NUM_POINTS,
            "model_save_format": TRIPOSR_MODEL_SAVE_FORMAT,
        },
        "reconstruction_preprocess": {
            "segmented_mode": "yolo_bbox_crop_for_triposr_rembg",
            "crop_margin_ratio": TRIPOSR_CROP_MARGIN_RATIO,
            "crop_min_margin_px": TRIPOSR_CROP_MIN_MARGIN_PX,
            "background_handling": "triposr_rembg" if TRIPOSR_REMOVE_BACKGROUND else "input_background",
            "legacy_mask_artifacts": True,
        },
        "outputs": {
            "segment_outputs": "/segment-outputs",
            "models": "/models",
        },
    }


@app.post("/segment-object")
async def segment_object(
    image: UploadFile = File(...),
    object_id: int | None = Form(default=None),
    bbox_x: float | None = Form(default=None),
    bbox_y: float | None = Form(default=None),
    bbox_width: float | None = Form(default=None),
    bbox_height: float | None = Form(default=None),
):
    started_at = time.perf_counter()
    pil_image = await read_upload_image(image)
    result, selected_detection, _ = detect_and_select_object(
        pil_image,
        object_id=object_id,
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
    )
    job_id = build_job_id(selected_detection["label"])
    segment_payload, _ = save_segment_artifacts(
        pil_image=pil_image,
        result=result,
        selected_detection=selected_detection,
        job_id=job_id,
    )

    return {
        "job_id": job_id,
        "image_width": pil_image.size[0],
        "image_height": pil_image.size[1],
        "processing_ms": round((time.perf_counter() - started_at) * 1000, 1),
        "selected": segment_payload["selected"],
        "output_dir": segment_payload["output_dir"],
        "detector": {
            "imgsz": DETECTION_IMAGE_SIZE,
            "conf": DETECTION_CONFIDENCE,
            "max_det": DETECTION_MAX_OBJECTS,
            "iou": DETECTION_IOU,
            "device": get_yolo_device(),
        },
        "files": segment_payload["files"],
        "paths": segment_payload["paths"],
        "preprocessing": segment_payload["preprocessing"],
    }


@app.post("/detect-frame")
async def detect_frame(image: UploadFile = File(...)):
    started_at = time.perf_counter()
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    image_width, image_height = pil_image.size
    model = get_yolo_model()

    try:
        results = model.predict(
            pil_image,
            conf=DETECTION_CONFIDENCE,
            imgsz=DETECTION_IMAGE_SIZE,
            max_det=DETECTION_MAX_OBJECTS,
            iou=DETECTION_IOU,
            device=get_yolo_device(),
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YOLO detection failed: {exc}") from exc

    detections = []
    result = results[0] if results else None
    if result is not None and result.boxes is not None:
        for index, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = clamp_bbox_xyxy(
                [float(value) for value in box.xyxy[0].tolist()],
                image_width=image_width,
                image_height=image_height,
            )
            detections.append(
                {
                    "id": str(index),
                    "label": model.names.get(cls_id, str(cls_id)),
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x": round(x1, 2),
                        "y": round(y1, 2),
                        "width": round(x2 - x1, 2),
                        "height": round(y2 - y1, 2),
                    },
                }
            )

    return {
        "image_width": image_width,
        "image_height": image_height,
        "processing_ms": round((time.perf_counter() - started_at) * 1000, 1),
        "detector": {
            "imgsz": DETECTION_IMAGE_SIZE,
            "conf": DETECTION_CONFIDENCE,
            "max_det": DETECTION_MAX_OBJECTS,
            "iou": DETECTION_IOU,
            "device": get_yolo_device(),
        },
        "objects": detections,
    }


@app.post("/upload-scan-video")
async def upload_scan_video(
    video: UploadFile = File(...),
    selected_object: str | None = Form(default=None),
):
    return {
        "job_id": "mock_scan_001",
        "status": "queued",
        "filename": video.filename,
        "selected_object": selected_object,
    }


@app.get("/scan-status/{job_id}")
def get_scan_status(job_id: str):
    return {
        "job_id": job_id,
        "status": "done",
        "progress": 100,
        "model_url": "http://localhost:8000/models/mock_scan_001.glb",
    }


@app.post("/reconstruct-object")
async def reconstruct_object(
    image: UploadFile = File(...),
    object_id: int | None = Form(default=None),
    bbox_x: float | None = Form(default=None),
    bbox_y: float | None = Form(default=None),
    bbox_width: float | None = Form(default=None),
    bbox_height: float | None = Form(default=None),
):
    started_at = time.perf_counter()
    pil_image = await read_upload_image(image)
    result, selected_detection, detections = detect_and_select_object(
        pil_image,
        object_id=object_id,
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
    )

    job_id = build_job_id(selected_detection["label"])
    segment_payload, triposr_crop_path = save_segment_artifacts(
        pil_image=pil_image,
        result=result,
        selected_detection=selected_detection,
        job_id=job_id,
    )
    reconstruction = save_reconstruction_artifacts(
        triposr_crop_path,
        job_id,
        label=selected_detection["label"],
    )

    return {
        "job_id": job_id,
        "status": "done",
        "processing_ms": round((time.perf_counter() - started_at) * 1000, 1),
        "image_width": pil_image.size[0],
        "image_height": pil_image.size[1],
        "selected": segment_payload["selected"],
        "detections": [
            {
                "id": str(detection["index"]),
                "label": detection["label"],
                "confidence": round(detection["confidence"], 4),
            }
            for detection in detections
        ],
        "segmentation": segment_payload,
        "reconstruction": reconstruction,
    }


@app.post("/reconstruct-image")
async def reconstruct_image(image: UploadFile = File(...)):
    filename = image.filename or "input.jpg"
    suffix = Path(filename).suffix or ".jpg"
    job_id = build_job_id(Path(filename).stem or "image")
    upload_dir = job_output_dir(UPLOAD_DIR, job_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / f"input{suffix}"
    pil_image = await read_upload_image(image)
    if suffix.lower() in {".jpg", ".jpeg"}:
        pil_image.save(input_path, quality=92)
    else:
        pil_image.save(input_path)

    reconstruction = save_reconstruction_artifacts(input_path, job_id, label=Path(filename).stem or "image")

    return {
        "job_id": job_id,
        "status": "done",
        "backend": reconstruction["backend"],
        "num_points": reconstruction["num_points"],
        "preprocessing": {
            "mode": "triposr_direct_image",
            "background_handling": "triposr_rembg" if TRIPOSR_REMOVE_BACKGROUND else "input_background",
        },
        "input_path": str(input_path),
        "triposr_input_path": reconstruction["paths"]["triposr_input"],
        "pointcloud_npy": reconstruction["paths"]["pointcloud_npy"],
        "pointcloud_ply": reconstruction["paths"]["pointcloud_ply"],
        "mesh": reconstruction["paths"]["mesh"],
        "mesh_glb": reconstruction["paths"]["mesh_glb"],
        "mesh_obj": reconstruction["paths"]["mesh_obj"],
        "mesh_colored_ply": reconstruction["paths"]["mesh_colored_ply"],
        "preview_png": reconstruction["paths"]["preview_png"],
        "model_url": (
            reconstruction["files"].get("mesh_glb")
            or reconstruction["files"].get("mesh")
            or reconstruction["files"].get("mesh_obj")
        ),
        "files": reconstruction["files"],
        "mesh_summary": reconstruction["mesh"],
        "reconstruction": reconstruction,
    }
