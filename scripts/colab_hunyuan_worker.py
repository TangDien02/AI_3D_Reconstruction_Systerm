from __future__ import annotations

import io
import gc
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

import torch


WORK_DIR = Path(os.environ.get("HUNYUAN_COLAB_WORK_DIR", "/tmp/hunyuan_jobs"))
MODEL_ID = os.environ.get("HUNYUAN_MODEL_ID", "tencent/Hunyuan3D-2")
MODEL_SUBFOLDER = os.environ.get("HUNYUAN_MODEL_SUBFOLDER", "hunyuan3d-dit-v2-0")
TEXGEN_MODEL_ID = os.environ.get("HUNYUAN_TEXGEN_MODEL_ID", MODEL_ID)
INFERENCE_STEPS = int(os.environ.get("HUNYUAN_INFERENCE_STEPS", "20"))
OCTREE_RESOLUTION = int(os.environ.get("HUNYUAN_OCTREE_RESOLUTION", "320"))
NUM_CHUNKS = int(os.environ.get("HUNYUAN_NUM_CHUNKS", "12000"))
SEED = int(os.environ.get("HUNYUAN_SEED", "12345"))
KEEP_SHAPE_PIPELINE = os.environ.get("HUNYUAN_KEEP_SHAPE_PIPELINE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
KEEP_TEXTURE_PIPELINE = os.environ.get("HUNYUAN_KEEP_TEXTURE_PIPELINE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

WORK_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Hunyuan Colab Worker")
_shape_pipeline = None
_texture_pipeline = None


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def get_shape_pipeline():
    global _shape_pipeline
    if _shape_pipeline is None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Hunyuan shape dependencies are not ready. Re-run the install cells and check worker logs.",
            ) from exc

        cleanup_memory()
        _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_ID,
            subfolder=MODEL_SUBFOLDER,
            use_safetensors=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _shape_pipeline


def get_texture_pipeline():
    global _texture_pipeline
    if _texture_pipeline is None:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Hunyuan texture dependencies are not ready. Install the official "
                    "texgen custom_rasterizer and differentiable_renderer modules."
                ),
            ) from exc

        cleanup_memory()
        _texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(TEXGEN_MODEL_ID)
    return _texture_pipeline


def unload_shape_pipeline():
    global _shape_pipeline
    _shape_pipeline = None
    cleanup_memory()


def unload_texture_pipeline():
    global _texture_pipeline
    _texture_pipeline = None
    cleanup_memory()


def read_image(image_bytes: bytes) -> Image.Image:
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGBA")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model_id": MODEL_ID,
        "model_subfolder": MODEL_SUBFOLDER,
        "texgen_model_id": TEXGEN_MODEL_ID,
        "shape_pipeline_loaded": _shape_pipeline is not None,
        "texture_pipeline_loaded": _texture_pipeline is not None,
        "cuda_memory_allocated_mb": (
            round(torch.cuda.memory_allocated() / 1024 / 1024, 1) if torch.cuda.is_available() else None
        ),
        "cuda_memory_reserved_mb": (
            round(torch.cuda.memory_reserved() / 1024 / 1024, 1) if torch.cuda.is_available() else None
        ),
    }


@app.post("/warmup")
def warmup():
    get_shape_pipeline()
    return health()


@app.post("/warmup-texture")
def warmup_texture():
    get_texture_pipeline()
    return health()


async def prepare_job_image(image: UploadFile, job_id: str) -> tuple[Image.Image, Path]:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    pil_image = read_image(image_bytes)
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.png"
    pil_image.save(input_path)
    return pil_image, job_dir


def generate_shape_mesh(pil_image: Image.Image):
    pipeline = get_shape_pipeline()
    with torch.inference_mode():
        return pipeline(
            image=pil_image,
            num_inference_steps=INFERENCE_STEPS,
            octree_resolution=OCTREE_RESOLUTION,
            num_chunks=NUM_CHUNKS,
            generator=torch.manual_seed(SEED),
        )[0]


def validate_glb_output(output_format: str):
    if output_format.lower() != "glb":
        raise HTTPException(status_code=400, detail="Only glb output is supported.")


def load_mesh(mesh_path: Path):
    try:
        import trimesh
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"trimesh is not importable: {exc}") from exc

    mesh = trimesh.load(mesh_path, force="mesh")
    if hasattr(mesh, "geometry"):
        mesh = mesh.dump(concatenate=True)
    return mesh


@app.post("/generate-shape")
async def generate_shape(
    image: UploadFile = File(...),
    job_id: str = Form(...),
    output_format: str = Form(default="glb"),
):
    validate_glb_output(output_format)
    pil_image, job_dir = await prepare_job_image(image, job_id)

    mesh = generate_shape_mesh(pil_image)
    output_path = job_dir / "mesh.glb"
    mesh.export(output_path)
    del mesh
    if not KEEP_SHAPE_PIPELINE:
        unload_shape_pipeline()
    cleanup_memory()
    return FileResponse(output_path, media_type="model/gltf-binary", filename="mesh.glb")


@app.post("/generate-texture")
async def generate_texture(
    image: UploadFile = File(...),
    mesh: UploadFile = File(...),
    job_id: str = Form(...),
    output_format: str = Form(default="glb"),
):
    validate_glb_output(output_format)
    pil_image, job_dir = await prepare_job_image(image, job_id)

    mesh_bytes = await mesh.read()
    if not mesh_bytes:
        raise HTTPException(status_code=400, detail="Uploaded mesh is empty.")

    mesh_path = job_dir / "input_mesh.glb"
    mesh_path.write_bytes(mesh_bytes)
    input_mesh = load_mesh(mesh_path)

    texture_pipeline = get_texture_pipeline()
    try:
        with torch.inference_mode():
            textured_mesh = texture_pipeline(input_mesh, image=pil_image.convert("RGB"))
    except RuntimeError as exc:
        cleanup_memory()
        raise HTTPException(status_code=500, detail=f"Hunyuan texture generation failed: {exc}") from exc

    output_path = job_dir / "mesh.glb"
    textured_mesh.export(output_path)
    del input_mesh
    del textured_mesh
    if not KEEP_TEXTURE_PIPELINE:
        unload_texture_pipeline()
    cleanup_memory()
    return FileResponse(output_path, media_type="model/gltf-binary", filename="mesh.glb")


@app.post("/generate-textured-shape")
async def generate_textured_shape(
    image: UploadFile = File(...),
    job_id: str = Form(...),
    output_format: str = Form(default="glb"),
):
    validate_glb_output(output_format)
    pil_image, job_dir = await prepare_job_image(image, job_id)

    mesh = generate_shape_mesh(pil_image)
    shape_path = job_dir / "shape_mesh.glb"
    mesh.export(shape_path)
    if not KEEP_SHAPE_PIPELINE:
        unload_shape_pipeline()
    cleanup_memory()

    texture_pipeline = get_texture_pipeline()
    try:
        with torch.inference_mode():
            textured_mesh = texture_pipeline(mesh, image=pil_image.convert("RGB"))
    except RuntimeError as exc:
        cleanup_memory()
        raise HTTPException(status_code=500, detail=f"Hunyuan texture generation failed: {exc}") from exc

    output_path = job_dir / "mesh.glb"
    textured_mesh.export(output_path)
    del mesh
    del textured_mesh
    if not KEEP_TEXTURE_PIPELINE:
        unload_texture_pipeline()
    cleanup_memory()
    return FileResponse(output_path, media_type="model/gltf-binary", filename="mesh.glb")
