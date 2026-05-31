from __future__ import annotations

import importlib
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "fastapi",
    "uvicorn",
    "numpy",
    "PIL",
    "matplotlib",
    "torch",
    "torchvision",
    "ultralytics",
    "trimesh",
    "omegaconf",
    "einops",
    "transformers",
    "huggingface_hub",
    "imageio",
    "xatlas",
    "moderngl",
    "skimage",
    "rembg",
    "onnxruntime",
]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    project_dir = project_root / "project"
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    missing: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        print("Missing imports:", ", ".join(missing))
        return 1

    try:
        from src.reconstruction.triposr_runner import TripoSRConfig, TripoSRCore
    except Exception as exc:
        print(f"Failed to import TripoSR runner: {exc}")
        return 1

    print(f"Python: {sys.version}")
    print(f"TripoSR config default model: {TripoSRConfig().model_name_or_path}")
    print(f"TripoSR core class: {TripoSRCore.__name__}")
    print("Runtime import check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
