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


DETECTION_CONFIDENCE = env_float("YOLO_DETECTION_CONFIDENCE", 0.20)
DETECTION_IMAGE_SIZE = env_int("YOLO_DETECTION_IMAGE_SIZE", 640)
DETECTION_MAX_OBJECTS = env_int("YOLO_DETECTION_MAX_OBJECTS", 20)
DETECTION_IOU = env_float("YOLO_DETECTION_IOU", 0.45)
MODEL_INPUT_IMAGE_SIZE = env_int("RECON_MODEL_INPUT_IMAGE_SIZE", 224)
MODEL_INPUT_MARGIN_RATIO = env_float("RECON_MODEL_INPUT_MARGIN_RATIO", 0.08)
MODEL_INPUT_MIN_MARGIN_PX = env_int("RECON_MODEL_INPUT_MIN_MARGIN_PX", 8)
DEFAULT_BASELINE_CHECKPOINT = (
    PROJECT_DIR
    / "results"
    / "all_categories_resnet50_2048pts_30ep_aug"
    / "outputs"
    / "checkpoints"
    / "best_model.pt"
)
BASELINE_CHECKPOINT = Path(os.environ.get("RECON_BASELINE_CHECKPOINT", DEFAULT_BASELINE_CHECKPOINT))
if not BASELINE_CHECKPOINT.is_absolute():
    BASELINE_CHECKPOINT = PROJECT_DIR / BASELINE_CHECKPOINT
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEGMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/segment-outputs", StaticFiles(directory=SEGMENT_OUTPUT_DIR), name="segment_outputs")
app.mount("/models", StaticFiles(directory=MODEL_OUTPUT_DIR), name="models")

_yolo_model = None
_yolo_model_lock = Lock()


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
    masked_crop_path = segment_dir / "masked_crop.png"
    transparent_crop_path = segment_dir / "transparent_crop.png"
    model_input_path = segment_dir / "model_input.png"
    model_input_mask_path = segment_dir / "model_input_mask.png"
    overlay_path = segment_dir / "overlay.jpg"

    pil_image.save(original_path, quality=92)
    full_mask.save(mask_path)
    crop.save(crop_path, quality=92)
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
            "masked_crop": str(masked_crop_path),
            "transparent_crop": str(transparent_crop_path),
            "model_input": str(model_input_path),
            "model_input_mask": str(model_input_mask_path),
            "overlay": str(overlay_path),
        },
        "preprocessing": model_input_metadata,
    }
    write_json(segment_dir / "segment_summary.json", segment_payload)
    return segment_payload, model_input_path


def save_reconstruction_artifacts(input_path: Path, job_id: str, label: str | None = None) -> dict:
    try:
        from src.inference.baseline_inference import predict_pointcloud
        from src.utils.mesh_export import save_pointcloud_obj
        from src.utils.pointcloud_io import save_pointcloud_npy, save_pointcloud_ply
        from src.utils.visualization import plot_point_cloud
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Inference dependencies are not ready: {exc}",
        ) from exc

    output_dir = job_output_dir(MODEL_OUTPUT_DIR, job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        points, checkpoint = predict_pointcloud(input_path, BASELINE_CHECKPOINT)
        npy_path = save_pointcloud_npy(points, output_dir / "pointcloud.npy")
        ply_path = save_pointcloud_ply(points, output_dir / "pointcloud.ply")
        obj_path, mesh_summary = save_pointcloud_obj(points, output_dir / "model.obj")
        preview_path = plot_point_cloud(points, output_dir / "preview.png", title=label or job_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {exc}") from exc

    payload = {
        "job_id": job_id,
        "label": label,
        "output_dir": str(output_dir),
        "input_image": str(input_path),
        "num_points": int(points.shape[0]),
        "checkpoint": {
            "path": str(BASELINE_CHECKPOINT),
            "model_type": checkpoint.get("model_type"),
            "encoder_name": checkpoint.get("encoder_name"),
            "decoder_type": checkpoint.get("decoder_type", "mlp"),
            "coarse_points": checkpoint.get("coarse_points"),
            "refine_offset_scale": checkpoint.get("refine_offset_scale"),
            "num_points": checkpoint.get("num_points"),
        },
        "mesh": mesh_summary,
        "files": {
            "pointcloud_npy": to_model_url(npy_path),
            "pointcloud_ply": to_model_url(ply_path),
            "mesh_obj": to_model_url(obj_path),
            "preview_png": to_model_url(preview_path),
            "summary_json": to_model_url(output_dir / "reconstruction_summary.json"),
        },
        "paths": {
            "output_dir": str(output_dir),
            "input_image": str(input_path),
            "pointcloud_npy": str(npy_path),
            "pointcloud_ply": str(ply_path),
            "mesh_obj": str(obj_path),
            "preview_png": str(preview_path),
            "summary_json": str(output_dir / "reconstruction_summary.json"),
        },
    }
    write_json(output_dir / "reconstruction_summary.json", payload)
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
        "baseline_checkpoint_exists": BASELINE_CHECKPOINT.is_file(),
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "yolo_weights_exists": YOLO_WEIGHTS.is_file(),
        "yolo_device": get_yolo_device(),
        "detector": {
            "imgsz": DETECTION_IMAGE_SIZE,
            "conf": DETECTION_CONFIDENCE,
            "max_det": DETECTION_MAX_OBJECTS,
            "iou": DETECTION_IOU,
        },
        "reconstruction_preprocess": {
            "image_size": MODEL_INPUT_IMAGE_SIZE,
            "margin_ratio": MODEL_INPUT_MARGIN_RATIO,
            "min_margin_px": MODEL_INPUT_MIN_MARGIN_PX,
            "segmented_mode": "segmented_mask_crop_square_pad",
            "plain_mode": "plain_square_pad",
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
    if not BASELINE_CHECKPOINT.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Baseline checkpoint not found: {BASELINE_CHECKPOINT}",
        )

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
    segment_payload, masked_crop_path = save_segment_artifacts(
        pil_image=pil_image,
        result=result,
        selected_detection=selected_detection,
        job_id=job_id,
    )
    reconstruction = save_reconstruction_artifacts(
        masked_crop_path,
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
    if not BASELINE_CHECKPOINT.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Baseline checkpoint not found: {BASELINE_CHECKPOINT}",
        )

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
    model_input, preprocess_metadata = build_plain_model_input(pil_image)
    model_input_path = upload_dir / "model_input.png"
    model_input.save(model_input_path)

    reconstruction = save_reconstruction_artifacts(model_input_path, job_id, label=Path(filename).stem or "image")

    return {
        "job_id": job_id,
        "status": "done",
        "num_points": reconstruction["num_points"],
        "preprocessing": preprocess_metadata,
        "input_path": str(input_path),
        "model_input_path": str(model_input_path),
        "pointcloud_npy": reconstruction["paths"]["pointcloud_npy"],
        "pointcloud_ply": reconstruction["paths"]["pointcloud_ply"],
        "mesh_obj": reconstruction["paths"]["mesh_obj"],
        "preview_png": reconstruction["paths"]["preview_png"],
        "files": reconstruction["files"],
        "mesh": reconstruction["mesh"],
    }
