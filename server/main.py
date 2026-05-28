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

from src.preprocessing import object_preprocess

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
RECON_BACKEND = os.environ.get("RECON_BACKEND", "triposr").strip().lower()
if RECON_BACKEND in {"baseline", "pointcloud"}:
    RECON_BACKEND = "legacy_pointcloud"
if RECON_BACKEND not in {"triposr", "legacy_pointcloud"}:
    RECON_BACKEND = "triposr"
DEFAULT_MODEL_INPUT_IMAGE_SIZE = 512 if RECON_BACKEND == "triposr" else 224
MODEL_INPUT_IMAGE_SIZE = env_int("RECON_MODEL_INPUT_IMAGE_SIZE", DEFAULT_MODEL_INPUT_IMAGE_SIZE)
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


def needs_baseline_checkpoint() -> bool:
    return RECON_BACKEND == "legacy_pointcloud"


def triposr_readiness() -> dict[str, object]:
    try:
        from src.reconstruction.triposr_runner import TripoSRConfig
    except Exception as exc:
        return {
            "backend": "triposr",
            "available": False,
            "error": f"TripoSR adapter import failed: {exc}",
        }
    return TripoSRConfig.from_env().readiness()


def sam2_readiness() -> dict[str, object]:
    try:
        from src.segmentation.sam2_refiner import SAM2Config
    except Exception as exc:
        return {
            "enabled": False,
            "available": False,
            "error": f"SAM2 adapter import failed: {exc}",
        }
    return SAM2Config.from_env().readiness()


def maybe_refine_mask_with_sam2(
    pil_image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
    fallback_mask: Image.Image,
) -> tuple[Image.Image, dict[str, object]]:
    try:
        from src.segmentation.sam2_refiner import SAM2Config, refine_mask_from_bbox
    except Exception as exc:
        return fallback_mask, {
            "source": "yolo",
            "sam2_enabled": False,
            "sam2_available": False,
            "sam2_error": f"SAM2 adapter import failed: {exc}",
        }

    config = SAM2Config.from_env()
    status = config.readiness()
    if not config.enabled:
        return fallback_mask, {
            "source": "yolo",
            "sam2_enabled": False,
            "sam2_available": status.get("available"),
        }

    try:
        refined_mask, metadata = refine_mask_from_bbox(
            image=pil_image,
            bbox_xyxy=bbox_xyxy,
            config=config,
        )
        metadata["fallback_source"] = "yolo"
        return refined_mask, metadata
    except Exception as exc:
        if config.required:
            raise HTTPException(status_code=503, detail=f"SAM2 mask refinement failed: {exc}") from exc
        return fallback_mask, {
            "source": "yolo",
            "sam2_enabled": True,
            "sam2_available": status.get("available"),
            "sam2_error": str(exc),
            "fallback_reason": "sam2_failed",
        }


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
    return object_preprocess.preprocess_object_image(
        image=pil_image,
        mask=full_mask,
        bbox=selected_xyxy,
        image_size=MODEL_INPUT_IMAGE_SIZE,
        margin_ratio=MODEL_INPUT_MARGIN_RATIO,
        min_margin_px=MODEL_INPUT_MIN_MARGIN_PX,
    )


def build_plain_model_input(pil_image: Image.Image) -> tuple[Image.Image, dict]:
    padded_image, padding = object_preprocess.square_pad_image(pil_image.convert("RGB"), fill=(255, 255, 255))
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
    yolo_mask = make_mask_from_polygon(polygon, image_width, image_height, selected_xyxy)
    full_mask, mask_metadata = maybe_refine_mask_with_sam2(
        pil_image=pil_image,
        bbox_xyxy=selected_xyxy,
        fallback_mask=yolo_mask,
    )

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
        "mask_source": mask_metadata.get("source", "yolo"),
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
        "mask": mask_metadata,
    }
    write_json(segment_dir / "segment_summary.json", segment_payload)
    return segment_payload, model_input_path


def save_legacy_pointcloud_reconstruction_artifacts(input_path: Path, job_id: str, label: str | None = None) -> dict:
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
        "primary_output": "pointcloud_ply",
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


def save_triposr_reconstruction_artifacts(input_path: Path, job_id: str, label: str | None = None) -> dict:
    try:
        from src.postprocessing.mesh_io import copy_mesh_asset, summarize_mesh
        from src.reconstruction.triposr_runner import TripoSRRunner
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"TripoSR dependencies are not ready: {exc}",
        ) from exc

    output_dir = job_output_dir(MODEL_OUTPUT_DIR, job_id)
    runner = TripoSRRunner()

    try:
        runner.validate_ready()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"TripoSR is not configured: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = runner.reconstruct(input_path, output_dir)
        mesh_source_path = Path(str(result["mesh_path"]))
        mesh_extension = mesh_source_path.suffix.lower() or ".glb"
        mesh_path = copy_mesh_asset(mesh_source_path, output_dir / f"model{mesh_extension}")

        texture_path = None
        if result.get("texture_path"):
            texture_path = copy_mesh_asset(result["texture_path"], output_dir / "texture.png")

        render_path = None
        if result.get("render_path"):
            render_path = copy_mesh_asset(result["render_path"], output_dir / "render.mp4")

        triposr_input_path = None
        if result.get("prepared_input_path"):
            triposr_input_path = copy_mesh_asset(result["prepared_input_path"], output_dir / "triposr_input.png")

        preview_path = output_dir / "preview.png"
        Image.open(triposr_input_path or input_path).convert("RGB").save(preview_path)
        mesh_summary = summarize_mesh(mesh_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TripoSR reconstruction failed: {exc}") from exc

    primary_key = "mesh_glb" if mesh_path.suffix.lower() == ".glb" else "mesh_obj"
    files = {
        primary_key: to_model_url(mesh_path),
        "preview_png": to_model_url(preview_path),
        "summary_json": to_model_url(output_dir / "reconstruction_summary.json"),
    }
    paths = {
        "output_dir": str(output_dir),
        "input_image": str(input_path),
        primary_key: str(mesh_path),
        "preview_png": str(preview_path),
        "summary_json": str(output_dir / "reconstruction_summary.json"),
    }
    if texture_path is not None:
        files["texture_png"] = to_model_url(texture_path)
        paths["texture_png"] = str(texture_path)
    if render_path is not None:
        files["render_mp4"] = to_model_url(render_path)
        paths["render_mp4"] = str(render_path)
    if triposr_input_path is not None:
        files["triposr_input"] = to_model_url(triposr_input_path)
        paths["triposr_input"] = str(triposr_input_path)

    payload = {
        "job_id": job_id,
        "label": label,
        "backend": "triposr",
        "primary_output": primary_key,
        "output_dir": str(output_dir),
        "input_image": str(input_path),
        "processing_ms": result.get("processing_ms"),
        "mesh": mesh_summary,
        "triposr": {
            "raw_output_dir": result.get("raw_output_dir"),
            "config": result.get("config"),
            "stdout_tail": result.get("stdout_tail"),
            "stderr_tail": result.get("stderr_tail"),
        },
        "files": files,
        "paths": paths,
    }
    write_json(output_dir / "reconstruction_summary.json", payload)
    return payload


def save_reconstruction_artifacts(input_path: Path, job_id: str, label: str | None = None) -> dict:
    if RECON_BACKEND == "legacy_pointcloud":
        return save_legacy_pointcloud_reconstruction_artifacts(input_path, job_id, label=label)
    return save_triposr_reconstruction_artifacts(input_path, job_id, label=label)


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
    triposr_status = triposr_readiness()
    sam2_status = sam2_readiness()
    return {
        "status": "ok",
        "reconstruction_backend": RECON_BACKEND,
        "baseline_checkpoint_exists": BASELINE_CHECKPOINT.is_file(),
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "triposr": triposr_status,
        "sam2": sam2_status,
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
            "segmented_mode": "crop_mask_square_pad_resize",
            "plain_mode": "plain_square_pad",
            "mask_refinement": "sam2" if sam2_status.get("enabled") else "yolo",
        },
        "outputs": {
            "primary_output": "mesh_glb" if RECON_BACKEND == "triposr" else "pointcloud_ply",
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
        "mask": segment_payload["mask"],
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
    if needs_baseline_checkpoint() and not BASELINE_CHECKPOINT.is_file():
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
    if needs_baseline_checkpoint() and not BASELINE_CHECKPOINT.is_file():
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
        "backend": reconstruction.get("backend", RECON_BACKEND),
        "primary_output": reconstruction.get("primary_output"),
        "num_points": reconstruction.get("num_points"),
        "preprocessing": preprocess_metadata,
        "input_path": str(input_path),
        "model_input_path": str(model_input_path),
        "mesh": reconstruction.get("mesh"),
        "files": reconstruction["files"],
        "paths": reconstruction["paths"],
        "preview_png": reconstruction["paths"].get("preview_png"),
        "mesh_glb": reconstruction["paths"].get("mesh_glb"),
        "mesh_obj": reconstruction["paths"].get("mesh_obj"),
        "pointcloud_npy": reconstruction["paths"].get("pointcloud_npy"),
        "pointcloud_ply": reconstruction["paths"].get("pointcloud_ply"),
    }
