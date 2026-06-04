from __future__ import annotations

import ctypes
import gc
import io
import os
import sys
import time
import traceback
from pathlib import Path
from threading import Lock, Thread

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

import torch


WORK_DIR = Path(os.environ.get("HUNYUAN_VM_WORK_DIR", "/tmp/hunyuan_jobs"))
MODEL_ID = os.environ.get("HUNYUAN_MODEL_ID", "tencent/Hunyuan3D-2")
MODEL_SUBFOLDER = os.environ.get("HUNYUAN_MODEL_SUBFOLDER", "hunyuan3d-dit-v2-0")
TEXGEN_MODEL_ID = os.environ.get("HUNYUAN_TEXGEN_MODEL_ID", "tencent/Hunyuan3D-2")
INFERENCE_STEPS = int(os.environ.get("HUNYUAN_INFERENCE_STEPS", "25"))
OCTREE_RESOLUTION = int(os.environ.get("HUNYUAN_OCTREE_RESOLUTION", "256"))
NUM_CHUNKS = int(os.environ.get("HUNYUAN_NUM_CHUNKS", "6000"))
SEED = int(os.environ.get("HUNYUAN_SEED", "12345"))
SHAPE_TORCH_DTYPE = os.environ.get("HUNYUAN_SHAPE_TORCH_DTYPE", "float16").strip().lower()
TEXTURE_TORCH_DTYPE = os.environ.get("HUNYUAN_TEXTURE_TORCH_DTYPE", "float16").strip().lower()
MIN_TEXTURE_RAM_FREE_GB = float(os.environ.get("HUNYUAN_MIN_TEXTURE_RAM_FREE_GB", "8.5"))
LOW_CPU_MEM_USAGE = os.environ.get("HUNYUAN_LOW_CPU_MEM_USAGE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
KEEP_SHAPE_PIPELINE = os.environ.get("HUNYUAN_KEEP_SHAPE_PIPELINE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
KEEP_TEXTURE_PIPELINE = os.environ.get("HUNYUAN_KEEP_TEXTURE_PIPELINE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

WORK_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Hunyuan VM Production Worker")
_shape_pipeline = None
_texture_pipeline = None
_busy_lock = Lock()
_busy_job = None
_busy_started_at = None
_jobs_lock = Lock()
_jobs = {}


def start_busy(job_name: str) -> None:
    global _busy_job
    global _busy_started_at
    if not _busy_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail=f"Worker is busy with {_busy_job or 'another job'}. Wait for it to finish before sending another request.",
        )
    _busy_job = job_name
    _busy_started_at = time.time()


def finish_busy() -> None:
    global _busy_job
    global _busy_started_at
    _busy_job = None
    _busy_started_at = None
    _busy_lock.release()


def busy_payload() -> dict:
    return {
        "busy": _busy_lock.locked(),
        "busy_job": _busy_job,
        "busy_seconds": round(time.time() - _busy_started_at, 1) if _busy_started_at else 0,
    }


def update_job(job_id: str, **values) -> None:
    with _jobs_lock:
        job = _jobs.setdefault(job_id, {"job_id": job_id})
        job.update(values)


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def public_job(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {
        "job_id": job_id,
        "kind": job.get("kind"),
        "status": job.get("status"),
        "error": job.get("error"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "mesh_ready": bool(job.get("output_path") and Path(job["output_path"]).is_file()),
        **memory_payload("memory"),
        **busy_payload(),
    }


def start_background_job(job_id: str, kind: str, job_name: str, target, *args) -> dict:
    start_busy(job_name)
    update_job(
        job_id,
        kind=kind,
        status="running",
        error=None,
        output_path=None,
        started_at=time.time(),
        completed_at=None,
    )

    def runner() -> None:
        try:
            output_path = target(*args)
            update_job(job_id, status="done", output_path=str(output_path), completed_at=time.time())
        except BaseException as exc:
            update_job(
                job_id,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
                completed_at=time.time(),
            )
            traceback.print_exc()
        finally:
            finish_busy()

    Thread(target=runner, daemon=True).start()
    return public_job(job_id)


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def memory_payload(prefix: str = "memory") -> dict:
    payload = {}
    try:
        import psutil

        ram = psutil.virtual_memory()
        payload[f"{prefix}_ram_used_gb"] = round(ram.used / 1024**3, 2)
        payload[f"{prefix}_ram_available_gb"] = round(ram.available / 1024**3, 2)
        payload[f"{prefix}_ram_total_gb"] = round(ram.total / 1024**3, 2)
    except Exception:
        pass
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        payload[f"{prefix}_cuda_free_gb"] = round(free / 1024**3, 2)
        payload[f"{prefix}_cuda_total_gb"] = round(total / 1024**3, 2)
        payload[f"{prefix}_cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
        payload[f"{prefix}_cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
    return payload


def ram_available_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None


def require_texture_ram_headroom() -> None:
    cleanup_memory()
    available = ram_available_gb()
    if available is None:
        return
    if available < MIN_TEXTURE_RAM_FREE_GB:
        raise RuntimeError(
            "Not enough VM system RAM to safely load Hunyuan3D Paint. "
            f"Available={available:.2f}GB, required>={MIN_TEXTURE_RAM_FREE_GB:.2f}GB."
        )


def configured_torch_dtype(name: str):
    if name in {"", "none", "auto"}:
        return None
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise RuntimeError(f"Unsupported torch dtype: {name}")


def move_pipeline_to_device(pipeline, device: str):
    if device == "cpu" or not hasattr(pipeline, "to"):
        return pipeline
    moved = pipeline.to(device)
    return moved if moved is not None else pipeline


def load_pipeline_with_fallback(
    pipeline_cls,
    model_id: str,
    dtype_name: str,
    *,
    subfolder: str | None = None,
    use_safetensors: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_kwargs = {}
    if use_safetensors:
        base_kwargs["use_safetensors"] = True
    if subfolder:
        base_kwargs["subfolder"] = subfolder
    minimal_kwargs = {"subfolder": subfolder} if subfolder else {}

    optimized_kwargs = dict(base_kwargs)
    dtype = configured_torch_dtype(dtype_name)
    if dtype is not None:
        optimized_kwargs["torch_dtype"] = dtype
    if LOW_CPU_MEM_USAGE:
        optimized_kwargs["low_cpu_mem_usage"] = True

    attempts = [
        (optimized_kwargs, True),
        ({**optimized_kwargs, "device": device}, False),
        ({**base_kwargs, "device": device}, False),
        (base_kwargs, True),
        ({**minimal_kwargs, "device": device}, False),
        (minimal_kwargs, True),
    ]
    last_exc = None
    for kwargs, move_after_load in attempts:
        try:
            pipeline = pipeline_cls.from_pretrained(model_id, **kwargs)
            if move_after_load:
                pipeline = move_pipeline_to_device(pipeline, device)
            return pipeline
        except (TypeError, RuntimeError, ValueError) as exc:
            last_exc = exc
            cleanup_memory()

    raise RuntimeError(f"Could not load pipeline {model_id}: {last_exc}") from last_exc


def import_texture_dependencies() -> None:
    try:
        import custom_rasterizer  # noqa: F401
        import mesh_processor  # noqa: F401
    except BaseException as exc:
        raise RuntimeError(
            "Texture native extensions are not importable. "
            "Rebuild custom_rasterizer and differentiable_renderer in the active venv. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def get_shape_pipeline():
    global _shape_pipeline
    if _shape_pipeline is None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Hunyuan shape dependencies are not ready: {type(exc).__name__}: {exc}",
            ) from exc

        print(f"Loading shape model {MODEL_ID}/{MODEL_SUBFOLDER} ...", flush=True)
        cleanup_memory()
        _shape_pipeline = load_pipeline_with_fallback(
            Hunyuan3DDiTFlowMatchingPipeline,
            MODEL_ID,
            subfolder=MODEL_SUBFOLDER,
            dtype_name=SHAPE_TORCH_DTYPE,
        )
    return _shape_pipeline


def get_texture_pipeline():
    global _texture_pipeline
    if _texture_pipeline is None:
        try:
            import_texture_dependencies()
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
        except BaseException as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Hunyuan texture dependencies are not ready: {type(exc).__name__}: {exc}",
            ) from exc

        print(f"Loading texture model {TEXGEN_MODEL_ID} ...", flush=True)
        cleanup_memory()
        try:
            _texture_pipeline = load_pipeline_with_fallback(
                Hunyuan3DPaintPipeline,
                TEXGEN_MODEL_ID,
                dtype_name=TEXTURE_TORCH_DTYPE,
                use_safetensors=False,
            )
        except BaseException as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Texture pipeline load failed: {type(exc).__name__}: {exc}") from exc
    return _texture_pipeline


def unload_shape_pipeline() -> None:
    global _shape_pipeline
    _shape_pipeline = None
    cleanup_memory()


def unload_texture_pipeline() -> None:
    global _texture_pipeline
    _texture_pipeline = None
    cleanup_memory()


def read_image(image_bytes: bytes) -> Image.Image:
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGBA")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model_id": MODEL_ID,
        "model_subfolder": MODEL_SUBFOLDER,
        "texgen_model_id": TEXGEN_MODEL_ID,
        "shape_pipeline_loaded": _shape_pipeline is not None,
        "texture_pipeline_loaded": _texture_pipeline is not None,
        "python": sys.version,
        **memory_payload("memory"),
        **busy_payload(),
    }


@app.get("/diagnostics")
def diagnostics() -> dict:
    import importlib.metadata as metadata

    packages = {}
    for name in ["torch", "torchvision", "diffusers", "transformers", "huggingface_hub", "accelerate", "xformers"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    imports = {}
    for module in ["custom_rasterizer", "mesh_processor", "hy3dgen.texgen", "hy3dgen.shapegen"]:
        try:
            __import__(module)
            imports[module] = "ok"
        except BaseException as exc:
            imports[module] = f"{type(exc).__name__}: {exc}"

    return {
        "status": "ok",
        "packages": packages,
        "imports": imports,
        "torch_file": torch.__file__,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        **health(),
    }


@app.post("/cleanup-memory")
def cleanup_memory_endpoint(
    unload_shape: bool = True,
    unload_texture: bool = True,
    clear_jobs: bool = False,
) -> dict:
    global _shape_pipeline
    global _texture_pipeline
    if _busy_lock.locked():
        raise HTTPException(
            status_code=409,
            detail=f"Worker is busy with {_busy_job or 'another job'}. Wait before cleanup.",
        )

    before = memory_payload("before")
    if unload_shape:
        _shape_pipeline = None
    if unload_texture:
        _texture_pipeline = None
    if clear_jobs:
        with _jobs_lock:
            _jobs.clear()
    cleanup_memory()
    after = memory_payload("after")
    return {
        "status": "ok",
        "shape_pipeline_loaded": _shape_pipeline is not None,
        "texture_pipeline_loaded": _texture_pipeline is not None,
        **before,
        **after,
        **busy_payload(),
    }


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return public_job(job_id)


@app.get("/jobs/{job_id}/mesh")
def job_mesh(job_id: str) -> Response:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail=f"Job is not done: {job.get('status')}")
    output_path = Path(job.get("output_path") or "")
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail=f"Mesh output not found for job: {job_id}")
    return glb_response(output_path)


@app.post("/warmup")
def warmup() -> dict:
    start_busy("warmup-shape")
    try:
        get_shape_pipeline()
        return health()
    finally:
        finish_busy()


@app.post("/warmup-texture")
def warmup_texture() -> dict:
    start_busy("warmup-texture")
    try:
        require_texture_ram_headroom()
        get_texture_pipeline()
        return health()
    except HTTPException:
        raise
    except BaseException as exc:
        traceback.print_exc()
        cleanup_memory()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        finish_busy()


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


def validate_glb_output(output_format: str) -> None:
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


def glb_response(output_path: Path, filename: str = "mesh.glb") -> Response:
    content = output_path.read_bytes()
    return Response(
        content=content,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(content)),
        },
    )


def run_shape_job(pil_image: Image.Image, job_dir: Path) -> Path:
    mesh = None
    try:
        mesh = generate_shape_mesh(pil_image)
        output_path = job_dir / "mesh.glb"
        mesh.export(output_path)
        return output_path
    finally:
        if mesh is not None:
            del mesh
        if not KEEP_SHAPE_PIPELINE:
            unload_shape_pipeline()
        else:
            cleanup_memory()


def run_texture_job(pil_image: Image.Image, mesh_path: Path, job_dir: Path) -> Path:
    input_mesh = None
    textured_mesh = None
    try:
        require_texture_ram_headroom()
        input_mesh = load_mesh(mesh_path)
        texture_pipeline = get_texture_pipeline()
        with torch.inference_mode():
            textured_mesh = texture_pipeline(input_mesh, image=pil_image.convert("RGB"))

        output_path = job_dir / "mesh.glb"
        textured_mesh.export(output_path)
        return output_path
    finally:
        if input_mesh is not None:
            del input_mesh
        if textured_mesh is not None:
            del textured_mesh
        if not KEEP_TEXTURE_PIPELINE:
            unload_texture_pipeline()
        else:
            cleanup_memory()


def run_textured_shape_job(pil_image: Image.Image, job_dir: Path) -> Path:
    mesh = None
    textured_mesh = None
    try:
        mesh = generate_shape_mesh(pil_image)
        shape_path = job_dir / "shape_mesh.glb"
        mesh.export(shape_path)
        if not KEEP_SHAPE_PIPELINE:
            unload_shape_pipeline()
        else:
            cleanup_memory()

        require_texture_ram_headroom()
        texture_pipeline = get_texture_pipeline()
        with torch.inference_mode():
            textured_mesh = texture_pipeline(mesh, image=pil_image.convert("RGB"))

        output_path = job_dir / "mesh.glb"
        textured_mesh.export(output_path)
        return output_path
    finally:
        if mesh is not None:
            del mesh
        if textured_mesh is not None:
            del textured_mesh
        if not KEEP_TEXTURE_PIPELINE:
            unload_texture_pipeline()
        else:
            cleanup_memory()


@app.post("/generate-shape")
async def generate_shape(
    image: UploadFile = File(...),
    job_id: str = Form(...),
    output_format: str = Form(default="glb"),
) -> Response:
    validate_glb_output(output_format)
    pil_image, job_dir = await prepare_job_image(image, job_id)

    start_busy(f"shape:{job_id}")
    try:
        output_path = await run_in_threadpool(run_shape_job, pil_image, job_dir)
    finally:
        finish_busy()
    return glb_response(output_path)


@app.post("/start-shape")
async def start_shape(
    image: UploadFile = File(...),
    job_id: str = Form(...),
    output_format: str = Form(default="glb"),
) -> dict:
    validate_glb_output(output_format)
    pil_image, job_dir = await prepare_job_image(image, job_id)
    return start_background_job(job_id, "shape", f"shape:{job_id}", run_shape_job, pil_image, job_dir)


@app.post("/generate-texture")
async def generate_texture(
    image: UploadFile = File(...),
    mesh: UploadFile = File(...),
    job_id: str = Form(...),
    output_format: str = Form(default="glb"),
) -> Response:
    validate_glb_output(output_format)
    pil_image, job_dir = await prepare_job_image(image, job_id)

    mesh_bytes = await mesh.read()
    if not mesh_bytes:
        raise HTTPException(status_code=400, detail="Uploaded mesh is empty.")

    mesh_path = job_dir / "input_mesh.glb"
    mesh_path.write_bytes(mesh_bytes)

    start_busy(f"texture:{job_id}")
    try:
        output_path = await run_in_threadpool(run_texture_job, pil_image, mesh_path, job_dir)
    finally:
        finish_busy()
    return glb_response(output_path)


@app.post("/start-texture")
async def start_texture(
    image: UploadFile = File(...),
    mesh: UploadFile = File(...),
    job_id: str = Form(...),
    output_format: str = Form(default="glb"),
) -> dict:
    validate_glb_output(output_format)
    pil_image, job_dir = await prepare_job_image(image, job_id)

    mesh_bytes = await mesh.read()
    if not mesh_bytes:
        raise HTTPException(status_code=400, detail="Uploaded mesh is empty.")

    mesh_path = job_dir / "input_mesh.glb"
    mesh_path.write_bytes(mesh_bytes)
    return start_background_job(job_id, "texture", f"texture:{job_id}", run_texture_job, pil_image, mesh_path, job_dir)


@app.post("/generate-textured-shape")
async def generate_textured_shape(
    image: UploadFile = File(...),
    job_id: str = Form(...),
    output_format: str = Form(default="glb"),
) -> Response:
    validate_glb_output(output_format)
    pil_image, job_dir = await prepare_job_image(image, job_id)

    start_busy(f"textured-shape:{job_id}")
    try:
        output_path = await run_in_threadpool(run_textured_shape_job, pil_image, job_dir)
    finally:
        finish_busy()
    return glb_response(output_path)


@app.post("/start-textured-shape")
async def start_textured_shape(
    image: UploadFile = File(...),
    job_id: str = Form(...),
    output_format: str = Form(default="glb"),
) -> dict:
    validate_glb_output(output_format)
    pil_image, job_dir = await prepare_job_image(image, job_id)
    return start_background_job(
        job_id,
        "textured-shape",
        f"textured-shape:{job_id}",
        run_textured_shape_job,
        pil_image,
        job_dir,
    )
