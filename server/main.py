from __future__ import annotations

import io
import json
import math
import mimetypes
import os
import re
import shutil
import sys
import time
import subprocess
from datetime import datetime
from threading import Lock, Thread
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

# Internal Imports
from server.services.image_cleaner import clean_object_image
from server.utils.image_crop import ImageCropError, crop_user_bbox
import server.utils.geometry as geo
import server.utils.paths as paths
import server.services.job_service as jobs

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
YOLO_WEIGHTS = SERVER_DIR / "weights" / "yolo11n-seg.pt"


def load_local_env_files() -> None:
    for path in (REPO_DIR / ".env", REPO_DIR / ".env.local"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env_files()


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
RECON_CROP_MARGIN_RATIO = env_float("RECON_CROP_MARGIN_RATIO", 0.16)
RECON_CROP_MIN_MARGIN_PX = env_int("RECON_CROP_MIN_MARGIN_PX", 16)
RECONSTRUCTION_BACKEND = os.environ.get("RECONSTRUCTION_BACKEND", "hunyuan_remote").strip().lower()
HUNYUAN_REMOTE_URL = os.environ.get("HUNYUAN_REMOTE_URL", "").strip().rstrip("/")
HUNYUAN_REMOTE_TIMEOUT_SECONDS = env_int("HUNYUAN_REMOTE_TIMEOUT_SECONDS", 900)
HUNYUAN_REMOTE_OUTPUT_FORMAT = os.environ.get("HUNYUAN_REMOTE_OUTPUT_FORMAT", "glb").strip().lower()
HUNYUAN_REMOTE_ENABLE_TEXTURE = env_bool("HUNYUAN_REMOTE_ENABLE_TEXTURE", False)
IMAGE_CLEANER_BACKEND = os.environ.get("IMAGE_CLEANER_BACKEND", "auto").strip().lower()
ENABLE_REMBG_CLEANER = env_bool("ENABLE_REMBG_CLEANER", True)
CLEAN_IMAGE_MAX_SIDE = env_int("CLEAN_IMAGE_MAX_SIDE", 1536)
CLEAN_IMAGE_PAD_RATIO = env_float("CLEAN_IMAGE_PAD_RATIO", 0.08)
AR_USDZ_ENABLED = env_bool("AR_USDZ_ENABLED", True)
AR_USDZ_TIMEOUT_SECONDS = env_int("AR_USDZ_TIMEOUT_SECONDS", 180)
BLENDER_PATH = os.environ.get("BLENDER_PATH")

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
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_yolo_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    if not YOLO_WEIGHTS.is_file():
        raise HTTPException(status_code=503, detail=f"YOLO weights not found: {YOLO_WEIGHTS}")
    with _yolo_model_lock:
        if _yolo_model is None:
            try:
                from ultralytics import YOLO
                _yolo_model = YOLO(str(YOLO_WEIGHTS))
                _yolo_model.fuse()
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"YOLO failed: {exc}") from exc
    return _yolo_model


def find_blender_executable() -> Path | None:
    candidates = []
    if BLENDER_PATH: candidates.append(Path(BLENDER_PATH))
    blender_on_path = shutil.which("blender")
    if blender_on_path: candidates.append(Path(blender_on_path))
    for env_name in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        root = os.environ.get(env_name)
        if not root: continue
        blender_root = Path(root) / "Blender Foundation"
        if blender_root.is_dir():
            candidates.extend(sorted(blender_root.glob("Blender*\\blender.exe"), reverse=True))
    for candidate in candidates:
        if candidate.is_file(): return candidate.resolve()
    return None


def convert_glb_to_usdz(glb_path: Path) -> tuple[Path | None, dict]:
    status = {"enabled": AR_USDZ_ENABLED, "format": "usdz", "source": str(glb_path), "converted": False}
    if not AR_USDZ_ENABLED or glb_path.suffix.lower() != ".glb" or not glb_path.is_file():
        status["reason"] = "disabled_or_missing"
        return None, status
    blender_path = find_blender_executable()
    if not blender_path:
        status["reason"] = "blender_not_found"
        return None, status

    usdz_path = glb_path.with_suffix(".usdz")
    script_path = glb_path.parent / "_convert_glb_to_usdz.py"
    script_content = "import sys\nfrom pathlib import Path\nimport bpy\n" \
                     "input_path = sys.argv[-2]\noutput_path = sys.argv[-1]\n" \
                     "bpy.ops.object.select_all(action='SELECT')\nbpy.ops.object.delete()\n" \
                     "bpy.ops.import_scene.gltf(filepath=input_path)\n" \
                     "bpy.ops.wm.usd_export(filepath=output_path, export_textures=True, export_materials=True)\n"
    script_path.write_text(script_content, encoding="utf-8")
    
    try:
        subprocess.run([str(blender_path), "--background", "--python", str(script_path), "--", str(glb_path), str(usdz_path)],
                       check=False, capture_output=True, timeout=AR_USDZ_TIMEOUT_SECONDS)
    except Exception as exc:
        status["reason"] = f"crash: {exc}"
        return None, status

    if usdz_path.is_file():
        status["converted"] = True
        return usdz_path, status
    return None, status


async def read_upload_image(image: UploadFile) -> Image.Image:
    try:
        content = await image.read()
        return ImageOps.exif_transpose(Image.open(io.BytesIO(content))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc


def detect_and_select_object(pil_image: Image.Image, object_id: int | None = None,
                             bbox_x: float | None = None, bbox_y: float | None = None,
                             bbox_width: float | None = None, bbox_height: float | None = None):
    image_width, image_height = pil_image.size
    model = get_yolo_model()
    results = model.predict(pil_image, conf=DETECTION_CONFIDENCE, imgsz=DETECTION_IMAGE_SIZE,
                            max_det=DETECTION_MAX_OBJECTS, iou=DETECTION_IOU, device=get_yolo_device(), verbose=False)
    result = results[0] if results else None
    if not result or not result.boxes:
        raise HTTPException(status_code=404, detail="No object detected.")

    detections = []
    for index, box in enumerate(result.boxes):
        x1, y1, x2, y2 = geo.clamp_bbox_xyxy(box.xyxy[0].tolist(), image_width, image_height)
        detections.append({"index": index, "label": model.names.get(int(box.cls[0]), "object"),
                           "confidence": float(box.conf[0]), "bbox_xyxy": (x1, y1, x2, y2)})

    selected_bbox = None
    if None not in (bbox_x, bbox_y, bbox_width, bbox_height):
        selected_bbox = geo.clamp_bbox_xyxy([bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height], image_width, image_height)

    if selected_bbox:
        selected_detection = max(detections, key=lambda d: geo.bbox_iou(d["bbox_xyxy"], selected_bbox))
    elif object_id is not None and 0 <= object_id < len(detections):
        selected_detection = detections[object_id]
    else:
        selected_detection = max(detections, key=lambda d: d["confidence"])

    return result, selected_detection, detections


def compose_masked_crop(pil_image: Image.Image, full_mask: Image.Image, crop_box: tuple[int, int, int, int]):
    crop = pil_image.crop(crop_box)
    mask_crop = full_mask.crop(crop_box)
    masked_crop = Image.new("RGB", crop.size, (255, 255, 255))
    masked_crop.paste(crop, mask=mask_crop)
    transparent_crop = crop.convert("RGBA")
    transparent_crop.putalpha(mask_crop)
    return crop, mask_crop, masked_crop, transparent_crop


def build_segment_model_input(pil_image: Image.Image, full_mask: Image.Image, selected_xyxy: tuple[float, ...]):
    image_width, image_height = pil_image.size
    mask_bbox = full_mask.getbbox()
    base_bbox = geo.union_bbox_xyxy(selected_xyxy, tuple(float(v) for v in mask_bbox) if mask_bbox else None)
    expanded_bbox = geo.expand_bbox_xyxy(base_bbox, image_width, image_height, MODEL_INPUT_MARGIN_RATIO, MODEL_INPUT_MIN_MARGIN_PX)
    model_crop_box = geo.bbox_to_crop_box(expanded_bbox, image_width, image_height)
    _, model_mask_crop, model_masked_crop, _ = compose_masked_crop(pil_image, full_mask, model_crop_box)
    padded_image, padding = geo.square_pad_image(model_masked_crop, fill=(255, 255, 255))
    model_input = padded_image.resize((MODEL_INPUT_IMAGE_SIZE, MODEL_INPUT_IMAGE_SIZE), Image.Resampling.BILINEAR)
    return model_input, {"padding": padding, "model_crop": geo.crop_box_payload(model_crop_box)}


def save_segment_artifacts(pil_image: Image.Image, result, selected_detection: dict, job_id: str):
    image_width, image_height = pil_image.size
    selected_index = selected_detection["index"]
    selected_xyxy = selected_detection["bbox_xyxy"]
    crop_box = geo.bbox_to_crop_box(selected_xyxy, image_width, image_height)

    polygon = result.masks.xy[selected_index] if result.masks is not None and result.masks.xy is not None and selected_index < len(result.masks.xy) else None
    full_mask = geo.make_mask_from_polygon(polygon, image_width, image_height, selected_xyxy)
    crop, mask_crop, masked_crop, transparent_crop = compose_masked_crop(pil_image, full_mask, crop_box)
    
    segment_dir = paths.job_output_dir(SEGMENT_OUTPUT_DIR, job_id)
    segment_dir.mkdir(parents=True, exist_ok=True)
    
    paths_dict = {"original": segment_dir / "original.jpg", "mask": segment_dir / "mask.png", "crop": segment_dir / "crop.jpg"}
    pil_image.save(paths_dict["original"], quality=92)
    full_mask.save(paths_dict["mask"])
    crop.save(paths_dict["crop"], quality=92)
    
    urls = {k: paths.mounted_url(SEGMENT_OUTPUT_DIR, "/segment-outputs", v) for k, v in paths_dict.items()}
    payload = {"selected": selected_detection, "files": urls, "paths": {k: str(v) for k, v in paths_dict.items()}}
    paths.write_json(segment_dir / "segment_summary.json", payload)
    return payload, paths_dict["crop"]


def save_bbox_preprocess_artifacts(pil_image: Image.Image, job_id: str, bbox: dict):
    sample_dir = MODEL_OUTPUT_DIR / job_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    crop_result = crop_user_bbox(pil_image, bbox["x"], bbox["y"], bbox["width"], bbox["height"],
                                 margin_ratio=RECON_CROP_MARGIN_RATIO, min_margin_px=RECON_CROP_MIN_MARGIN_PX)
    clean_result = clean_object_image(crop_result.crop, backend=IMAGE_CLEANER_BACKEND, enable_rembg=ENABLE_REMBG_CLEANER)
    
    input_path = sample_dir / "input.png"
    clean_result.input_image.save(input_path)
    
    payload = {"job_id": job_id, "status": "done", "files": {"input": paths.mounted_url(MODEL_OUTPUT_DIR, "/models", input_path)}}
    paths.write_json(sample_dir / "preprocess_summary.json", payload)
    return payload, input_path


def run_reconstruct_bbox_job(pil_image: Image.Image, job_id: str, bbox: dict):
    try:
        jobs.set_reconstruction_job(job_id, {"status": "running", "stage": "preprocess"})
        preprocess, clean_path = save_bbox_preprocess_artifacts(pil_image, job_id, bbox)
        
        jobs.set_reconstruction_job(job_id, {"stage": "generating_shape"})
        reconstruction = save_hunyuan_remote_artifacts(clean_path, job_id, label="user_bbox")
        
        result = {"job_id": job_id, "status": "done", "reconstruction": reconstruction, "preprocess": preprocess}
        jobs.set_reconstruction_job(job_id, {"status": "completed", "result": result})
    except Exception as exc:
        jobs.set_reconstruction_job(job_id, {"status": "failed", "error": str(exc)})


def save_hunyuan_remote_artifacts(input_path: Path, job_id: str, label: str = "object"):
    if not HUNYUAN_REMOTE_URL: raise HTTPException(status_code=503, detail="Hunyuan URL not configured.")
    sample_dir = MODEL_OUTPUT_DIR / job_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    with input_path.open("rb") as f:
        files = {"image": (input_path.name, f, "image/png")}
        data = {"job_id": job_id, "output_format": "glb"}
        with httpx.Client(timeout=HUNYUAN_REMOTE_TIMEOUT_SECONDS) as client:
            resp = client.post(f"{HUNYUAN_REMOTE_URL}/generate-shape", data=data, files=files, headers={"ngrok-skip-browser-warning": "true"})
            if resp.status_code != 200: raise HTTPException(status_code=502, detail=f"Worker error: {resp.text}")
            content = resp.content

    mesh_path = sample_dir / "mesh.glb"
    mesh_path.write_bytes(content)
    usdz_path, usdz_status = convert_glb_to_usdz(mesh_path)
    
    payload = {
        "job_id": job_id, "backend": "hunyuan_remote",
        "files": {
            "mesh_glb": paths.mounted_url(MODEL_OUTPUT_DIR, "/models", mesh_path),
            "mesh_usdz": paths.mounted_url(MODEL_OUTPUT_DIR, "/models", usdz_path) if usdz_path else None
        },
        "ar": {"usdz": usdz_status}
    }
    paths.write_json(sample_dir / "reconstruction_summary.json", payload)
    return payload


@app.on_event("startup")
def warmup():
    try:
        model = get_yolo_model()
        model.predict(Image.new("RGB", (640, 640)), verbose=False)
    except: pass


@app.get("/health")
def health():
    return {"status": "ok", "yolo": get_yolo_device(), "backend": RECONSTRUCTION_BACKEND}


@app.post("/reconstruct-bbox")
async def reconstruct_bbox(image: UploadFile = File(...), bbox_x: float = Form(...), bbox_y: float = Form(...),
                           bbox_width: float = Form(...), bbox_height: float = Form(...), job_id: str | None = Form(None)):
    pil_image = await read_upload_image(image)
    resolved_id = job_id or paths.build_job_id("user-bbox")
    bbox = {"x": bbox_x, "y": bbox_y, "width": bbox_width, "height": bbox_height}
    
    Thread(target=run_reconstruct_bbox_job, args=(pil_image, resolved_id, bbox), daemon=True).start()
    return {"job_id": resolved_id, "status": "started", "status_url": f"/reconstruction-jobs/{resolved_id}"}


@app.get("/reconstruction-jobs/{job_id}")
async def reconstruction_job_status(job_id: str):
    job = jobs.get_reconstruction_job(job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") == "completed": return job["result"]
    return job


@app.post("/paint-texture")
async def paint_texture(job_id: str = Form(...)):
    # Placeholder for texture logic - similar pattern to shape
    return {"job_id": job_id, "status": "not_implemented_in_refactor_yet"}
