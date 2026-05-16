from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI(title="3DRecon API")

SERVER_DIR = Path(__file__).resolve().parent
REPO_DIR = SERVER_DIR.parent
PROJECT_DIR = REPO_DIR / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

UPLOAD_DIR = SERVER_DIR / "uploads"
MODEL_OUTPUT_DIR = SERVER_DIR / "models"
BASELINE_CHECKPOINT = PROJECT_DIR / "results" / "baseline" / "outputs" / "checkpoints" / "transformer_pointcloud_net.pt"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "baseline_checkpoint_exists": BASELINE_CHECKPOINT.is_file(),
    }


@app.post("/detect-frame")
async def detect_frame(image: UploadFile = File(...)):
    return {
        "objects": [
            {
                "id": "mock_object_1",
                "label": "object",
                "confidence": 0.92,
                "bbox": {"x": 120, "y": 180, "width": 240, "height": 300},
            }
        ]
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
