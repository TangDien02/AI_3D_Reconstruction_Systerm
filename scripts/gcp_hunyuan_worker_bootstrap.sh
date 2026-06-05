#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/work}"
REPO_URL="${REPO_URL:-https://github.com/TangDien02/AI_3D_Reconstruction_Systerm.git}"
REPO_DIR="${REPO_DIR:-$WORK_DIR/AI_3D_Reconstruction_Systerm}"
HUNYUAN_DIR="${HUNYUAN_DIR:-$WORK_DIR/Hunyuan3D-2}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git curl wget tmux build-essential python3-pip python3-venv ninja-build

mkdir -p "$WORK_DIR"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only
fi

if [ ! -d "$HUNYUAN_DIR/.git" ]; then
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$HUNYUAN_DIR"
else
  git -C "$HUNYUAN_DIR" pull --ff-only
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel "setuptools<82"

cd "$HUNYUAN_DIR"
python -m pip install --no-cache-dir -r requirements.txt
# Re-assert CUDA PyTorch after Hunyuan requirements so later dependency
# resolution cannot replace it with a CPU or incompatible wheel.
python -m pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
python -m pip install --no-cache-dir --force-reinstall "setuptools<82"
python -m pip install --no-cache-dir -e .
python -m pip install --no-cache-dir fastapi uvicorn[standard] python-multipart pillow psutil trimesh
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "diffusers==0.31.0" \
  "transformers==4.46.3" \
  "tokenizers==0.20.3" \
  "huggingface_hub==0.26.2" \
  "accelerate==1.1.1"
rm -rf "$HOME/.cache/huggingface/modules/diffusers_modules"

python - <<'PY'
import importlib.metadata as metadata
import torch

expected = {
    "diffusers": "0.31.0",
    "transformers": "4.46.3",
    "tokenizers": "0.20.3",
    "huggingface_hub": "0.26.2",
    "accelerate": "1.1.1",
}
print("Torch:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")
if "+cu" not in torch.__version__:
    raise SystemExit(f"Expected a CUDA PyTorch wheel, got torch=={torch.__version__}")
for package, version in expected.items():
    actual = metadata.version(package)
    print(f"{package}=={actual}")
    if actual != version:
        raise SystemExit(f"Expected {package}=={version}, got {actual}")
PY

TORCH_LIB="$(python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

if command -v nvcc >/dev/null 2>&1; then
  (cd "$HUNYUAN_DIR/hy3dgen/texgen/custom_rasterizer" && python setup.py install)
  (cd "$HUNYUAN_DIR/hy3dgen/texgen/differentiable_renderer" && python setup.py install)
  python - <<'PY'
import ctypes
from pathlib import Path
import torch

torch_lib = Path(torch.__file__).resolve().parent / "lib"
for name in ("libc10.so", "libtorch.so", "libtorch_cpu.so", "libtorch_cuda.so"):
    path = torch_lib / name
    if path.exists():
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

import custom_rasterizer
import mesh_processor
print("Texture native extensions OK")
PY
else
  echo "nvcc was not found. Shape generation can run, but texture extensions need CUDA toolkit/nvcc."
fi

cp "$REPO_DIR/scripts/hunyuan_vm_worker_production.py" "$WORK_DIR/hunyuan_vm_worker.py"

cat > "$WORK_DIR/hunyuan-worker.env" <<EOF
PATH=$VENV_DIR/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
PYTHONPATH=$HUNYUAN_DIR
LD_LIBRARY_PATH=$TORCH_LIB:/usr/local/cuda/lib64
HUNYUAN_VM_WORK_DIR=$WORK_DIR/hunyuan_jobs
HUNYUAN_MODEL_ID=tencent/Hunyuan3D-2
HUNYUAN_MODEL_SUBFOLDER=hunyuan3d-dit-v2-0
HUNYUAN_TEXGEN_MODEL_ID=tencent/Hunyuan3D-2
HUNYUAN_INFERENCE_STEPS=25
HUNYUAN_OCTREE_RESOLUTION=256
HUNYUAN_NUM_CHUNKS=6000
HUNYUAN_KEEP_SHAPE_PIPELINE=0
HUNYUAN_KEEP_TEXTURE_PIPELINE=1
EOF

sudo tee /etc/systemd/system/hunyuan-worker.service >/dev/null <<EOF
[Unit]
Description=Hunyuan3D worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
EnvironmentFile=$WORK_DIR/hunyuan-worker.env
ExecStart=$VENV_DIR/bin/uvicorn hunyuan_vm_worker:app --host 0.0.0.0 --port 8010
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now hunyuan-worker

echo "Hunyuan worker service started."
echo "Check: curl http://127.0.0.1:8010/health"
echo "Logs:  sudo journalctl -u hunyuan-worker -f"
