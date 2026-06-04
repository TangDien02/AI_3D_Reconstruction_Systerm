"""Repair Hunyuan3D-2 texture loading on the VM.

Run on the Google Cloud VM from the active venv:

    cd ~/work
    source venv/bin/activate
    python /path/to/scripts/fix_hunyuan_texture_vm.py

The main fix is pinning a known-good Hugging Face stack. Newer diffusers can
load the Hunyuan paint UNet under a flattened dynamic module path, and newer
transformers / huggingface_hub / tokenizers combinations can break imports
such as is_offline_mode or tokenizers version checks.
"""

from __future__ import annotations

import importlib.metadata as metadata
import re
import shutil
import subprocess
import sys
from pathlib import Path


HOME = Path.home()
WORK = HOME / "work"
REPO = WORK / "Hunyuan3D-2"
WORKER = WORK / "hunyuan_vm_worker.py"
MULTIVIEW_UTILS = REPO / "hy3dgen" / "texgen" / "utils" / "multiview_utils.py"
DYNAMIC_MODULE_CACHE = HOME / ".cache" / "huggingface" / "modules" / "diffusers_modules"


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.check_call(args)


def strip_trust_remote_code(path: Path) -> None:
    if not path.exists():
        print(f"skip missing {path}")
        return

    text = path.read_text(encoding="utf-8")
    original = text
    text = re.sub(r"\s*trust_remote_code\s*=\s*True\s*,\s*", "", text)
    text = re.sub(r",\s*trust_remote_code\s*=\s*True\s*", "", text)
    text = text.replace("trust_remote_code=True", "")
    text = text.replace("(,", "(").replace(",,", ",")

    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"removed trust_remote_code from {path}")
    else:
        print(f"no trust_remote_code patch found in {path}")


def show_version(package: str) -> None:
    try:
        print(f"{package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: not installed")


def main() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ]
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
            "--no-deps",
            "diffusers==0.31.0",
            "transformers==4.46.3",
            "tokenizers==0.20.3",
            "huggingface_hub==0.26.2",
            "accelerate==1.1.1",
        ]
    )

    strip_trust_remote_code(WORKER)
    strip_trust_remote_code(MULTIVIEW_UTILS)

    if DYNAMIC_MODULE_CACHE.exists():
        shutil.rmtree(DYNAMIC_MODULE_CACHE)
        print(f"removed {DYNAMIC_MODULE_CACHE}")
    else:
        print(f"dynamic module cache not found: {DYNAMIC_MODULE_CACHE}")

    show_version("diffusers")
    show_version("transformers")
    show_version("tokenizers")
    show_version("huggingface_hub")
    show_version("accelerate")
    show_version("torch")
    show_version("torchvision")

    import custom_rasterizer  # noqa: F401
    import mesh_processor  # noqa: F401

    print("texture native extensions import OK")
    print("Restart the uvicorn worker, then run /warmup-texture again.")


if __name__ == "__main__":
    main()
