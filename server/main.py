from __future__ import annotations

import io
import shutil
import sys
import time
from threading import Lock
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

app = FastAPI(title="3DRecon API")

SERVER_DIR = Path(__file__).resolve().parent
REPO_DIR = SERVER_DIR.parent
PROJECT_DIR = REPO_DIR / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

UPLOAD_DIR = SERVER_DIR / "uploads"
MODEL_OUTPUT_DIR = SERVER_DIR / "models"
YOLO_WEIGHTS = SERVER_DIR / "weights" / "yolo26n-seg.pt"
DETECTION_CONFIDENCE = 0.60
DETECTION_IMAGE_SIZE = 416
DETECTION_MAX_OBJECTS = 8
BASELINE_CHECKPOINT = PROJECT_DIR / "results" / "chair_baseline" / "outputs" / "checkpoints" / "best_model.pt"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_yolo_model = None
_yolo_model_lock = Lock()


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
            verbose=False,
        )
    except Exception as exc:
        print(f"YOLO warmup skipped: {exc}")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "baseline_checkpoint_exists": BASELINE_CHECKPOINT.is_file(),
        "yolo_weights_exists": YOLO_WEIGHTS.is_file(),
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


@app.post("/reconstruct-image")
async def reconstruct_image(image: UploadFile = File(...)):
    if not BASELINE_CHECKPOINT.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Baseline checkpoint not found: {BASELINE_CHECKPOINT}",
        )

    suffix = Path(image.filename or "input.jpg").suffix or ".jpg"
    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"
    with input_path.open("wb") as file:
        shutil.copyfileobj(image.file, file)

    try:
        from src.inference.baseline_inference import predict_pointcloud
        from src.utils.pointcloud_io import save_pointcloud_npy, save_pointcloud_ply
        from src.utils.visualization import plot_point_cloud
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Inference dependencies are not ready: {exc}",
        ) from exc

    try:
        points, _ = predict_pointcloud(input_path, BASELINE_CHECKPOINT)
        npy_path = save_pointcloud_npy(points, MODEL_OUTPUT_DIR / f"{job_id}.npy")
        ply_path = save_pointcloud_ply(points, MODEL_OUTPUT_DIR / f"{job_id}.ply")
        preview_path = plot_point_cloud(points, MODEL_OUTPUT_DIR / f"{job_id}.png", title=job_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {exc}") from exc

    return {
        "job_id": job_id,
        "status": "done",
        "num_points": int(points.shape[0]),
        "pointcloud_npy": str(npy_path),
        "pointcloud_ply": str(ply_path),
        "preview_png": str(preview_path),
    }
