# Production Setup: Google Cloud Spot VM + Hunyuan3D-2 + Cloudflare + Expo

ASCII-only guide for a fresh VM.

Goal:

```text
Expo mobile app
  -> Windows FastAPI backend :8000
  -> Cloudflare Tunnel
  -> VM Hunyuan worker :8010
  -> Hunyuan3D-2 shape GLB
  -> optional Hunyuan Paint texture GLB
```

Recommended production mode:

```text
HUNYUAN_REMOTE_ENABLE_TEXTURE=false
```

This means:

1. Expo creates shape first.
2. Backend saves `mesh.glb`.
3. User/app calls `/paint-texture`.
4. Backend sends existing shape mesh + image to VM worker `/start-texture`.
5. Backend saves `mesh_textured.glb`.

This is safer on Tesla T4 than doing shape + texture in one long request.

## 1. Recommended VM

For Hunyuan3D-2 full shape + texture:

```text
GPU: NVIDIA Tesla T4 or better
System RAM: 30GB minimum, 50GB+ preferred
Boot disk: 100GB minimum, 150GB+ preferred
OS: Debian 12 or Ubuntu 22.04/24.04
Provisioning model: Spot
```

The commands below assume Debian 12.

## 2. SSH into VM

In Google Cloud Console:

```text
Compute Engine -> VM instances -> SSH
```

Prompt should look like:

```bash
pminhchien2006@instance-name:~$
```

## 2A. After reset or reboot quick start

Use this only when the VM disk already has `~/work`, `venv`, Hunyuan3D-2, and the worker script from a previous setup.

SSH tab 1, restart Jupyter:

```bash
tmux new -s jupyter
cd ~/work
source venv/bin/activate
source ~/.bashrc
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.allow_remote_access=True
```

Detach:

```text
Ctrl+B
D
```

SSH tab 2, expose Jupyter:

```bash
cloudflared tunnel --url http://127.0.0.1:8888
```

SSH tab 3, restart worker:

```bash
tmux new -s worker
cd ~/work
source venv/bin/activate
source ~/.bashrc
TORCH_LIB=$(python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
uvicorn hunyuan_vm_worker:app --host 0.0.0.0 --port 8010
```

Detach:

```text
Ctrl+B
D
```

SSH tab 4, expose worker:

```bash
cloudflared tunnel --url http://127.0.0.1:8010
```

Sanity checks:

```bash
curl http://127.0.0.1:8888
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/diagnostics
```

Important:

```text
Jupyter tunnel URL is only for opening /lab in the browser.
Worker tunnel URL is the only URL for HUNYUAN_REMOTE_URL.
Never send /start-shape to Jupyter port 8888.
```

## 3. Basic packages

```bash
sudo apt update
sudo apt install -y git wget curl tmux unzip build-essential dkms python3-pip python3-venv
```

## 4. NVIDIA driver

Check GPU exists:

```bash
lspci | grep -i nvidia
```

Enable Debian non-free repos:

```bash
sudo sed -i 's/^Components: main$/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources
sudo apt update
```

Install driver:

```bash
sudo apt install -y linux-headers-$(uname -r) nvidia-driver firmware-misc-nonfree
sudo reboot
```

After reboot:

```bash
nvidia-smi
```

## 5. CUDA Toolkit / nvcc

Texture requires building native CUDA extensions. NVIDIA driver alone is not enough.

Install NVIDIA CUDA apt repo:

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

Install CUDA 12 toolkit:

```bash
sudo apt install -y cuda-toolkit-12-6
```

If unavailable:

```bash
apt-cache search '^cuda-toolkit-12'
sudo apt install -y cuda-toolkit-12-8
```

Set env:

```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Check:

```bash
which nvcc
nvcc --version
```

## 6. Workspace and Python venv

```bash
mkdir -p ~/work
cd ~/work
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip wheel setuptools
```

Install PyTorch CUDA 12.6:

```bash
python -m pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Test:

```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")
PY
```

## 7. Install JupyterLab and Cloudflare Tunnel

```bash
cd ~/work
source venv/bin/activate
python -m pip install --no-cache-dir jupyterlab
```

Install `cloudflared`:

```bash
cd ~/work
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
cloudflared --version
```

## 8. Start JupyterLab

SSH tab 1:

```bash
tmux new -s jupyter
cd ~/work
source venv/bin/activate
source ~/.bashrc
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.allow_remote_access=True
```

Copy the Jupyter token.

Detach without stopping Jupyter:

```text
Ctrl+B
D
```

SSH tab 2:

```bash
cloudflared tunnel --url http://127.0.0.1:8888
```

Open on Windows:

```text
https://JUPYTER_TRYCLOUDFLARE_URL/lab?token=JUPYTER_TOKEN
```

## 9. Upload and run production notebook

Upload this notebook to JupyterLab:

```text
notebook/Hunyuan_expo_vm_cloudflare_production.ipynb
```

Run cells from top to bottom.

The notebook will:

- clone official Hunyuan3D-2
- install Python dependencies
- re-assert PyTorch CUDA `cu126` after Hunyuan requirements
- pin the Hugging Face stack:
  `diffusers==0.31.0`, `transformers==4.46.3`, `tokenizers==0.20.3`, `huggingface_hub==0.26.2`, `accelerate==1.1.1`
- clear Hugging Face diffusers dynamic module cache
- build `custom_rasterizer`
- build `mesh_processor`
- write `~/work/hunyuan_vm_worker.py`
- start worker on port `8010`
- expose diagnostics at `/diagnostics`
- optionally warm up texture model

Important known fix:

```bash
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "diffusers==0.31.0" \
  "transformers==4.46.3" \
  "tokenizers==0.20.3" \
  "huggingface_hub==0.26.2" \
  "accelerate==1.1.1"
rm -rf ~/.cache/huggingface/modules/diffusers_modules
```

This avoids Hunyuan Paint / Hugging Face version errors such as:

```text
Expected types for unet ... got diffusers_modules.local.modules.UNet2p5DConditionModel
cannot import name 'is_offline_mode' from 'huggingface_hub'
tokenizers>=0.20,<0.21 is required
```

## 10. Expose VM worker port 8010

After notebook starts the worker and `/health` works, open SSH tab 3:

```bash
cloudflared tunnel --url http://127.0.0.1:8010
```

Copy the worker URL:

```text
https://WORKER_TRYCLOUDFLARE_URL
```

Do not confuse URLs:

```text
Jupyter URL -> port 8888
Worker URL  -> port 8010
```

Worker URL sanity test from Windows browser:

```text
https://WORKER_TRYCLOUDFLARE_URL/health
```

This must show JSON with `status: ok`. If it shows Jupyter, HTML, `_xsrf`, or `Jupyter Server requires Javascript`, you copied the Jupyter tunnel instead of the worker tunnel.

Optional local shape smoke test on the VM:

```bash
cd ~/work
source venv/bin/activate
python - <<'PY'
import time
from pathlib import Path
import requests

base = "http://127.0.0.1:8010"
image_path = Path("/home/pminhchien2006/work/Hunyuan3D-2/assets/demo.png")

print(requests.get(base + "/health", timeout=10).text)
with image_path.open("rb") as f:
    r = requests.post(
        base + "/start-shape",
        files={"image": ("demo.png", f, "image/png")},
        data={"job_id": "local_shape_smoke", "output_format": "glb"},
        timeout=120,
    )
print(r.status_code, r.text[:1000])
r.raise_for_status()
job_id = r.json()["job_id"]

while True:
    s = requests.get(base + f"/jobs/{job_id}", timeout=30)
    print(s.status_code, s.text[:1000])
    s.raise_for_status()
    data = s.json()
    if data["status"] == "done":
        break
    if data["status"] in {"error", "failed"}:
        raise RuntimeError(data)
    time.sleep(5)

mesh = requests.get(base + f"/jobs/{job_id}/mesh", timeout=120)
mesh.raise_for_status()
out = Path("/home/pminhchien2006/work/local_shape_smoke.glb")
out.write_bytes(mesh.content)
print("saved", out, out.stat().st_size)
PY
```

## 11. Start Windows backend

In Windows PowerShell:

```powershell
cd C:/Users/pminh/Desktop/MyProject/AI_3D_Reconstruction_Systerm_TangDien02
.\.venv\Scripts\Activate.ps1
$env:RECONSTRUCTION_BACKEND="hunyuan_remote"
$env:HUNYUAN_REMOTE_URL="https://WORKER_TRYCLOUDFLARE_URL"
$env:HUNYUAN_REMOTE_OUTPUT_FORMAT="glb"
$env:HUNYUAN_REMOTE_ENABLE_TEXTURE="false"
$env:HUNYUAN_REMOTE_TIMEOUT_SECONDS="1800"
$env:HUNYUAN_REMOTE_POLL_INTERVAL_SECONDS="5"
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Check backend:

```text
http://127.0.0.1:8000/health
```

## 12. Start Expo mobile app

Find Windows LAN IP, for example:

```powershell
ipconfig
```

In another Windows PowerShell:

```powershell
cd C:/Users/pminh/Desktop/MyProject/AI_3D_Reconstruction_Systerm_TangDien02/mobile
$env:EXPO_PUBLIC_API_BASE_URL="http://YOUR_WINDOWS_LAN_IP:8000"
npm install
npm start
```

Then use Expo app.

## 13. Expected production flow

Shape:

```text
Expo -> Windows backend /reconstruct -> VM /start-shape -> poll /jobs/{job_id} -> download /jobs/{job_id}/mesh -> backend saves mesh.glb
```

Texture:

```text
Expo paint texture action -> Windows backend /paint-texture -> returns started
Expo polls Windows backend /texture-jobs/{job_id}
Windows backend background job -> VM /start-texture -> poll /jobs/{job_id}_texture -> download textured mesh -> backend saves mesh_textured.glb
```

Texture is intentionally async to avoid long HTTP requests and Cloudflare 524 timeouts.

## 14. Useful VM checks

Local worker health:

```bash
curl http://127.0.0.1:8010/health
```

Diagnostics:

```bash
curl http://127.0.0.1:8010/diagnostics
```

Worker log:

```bash
tail -n 120 /tmp/hunyuan_worker.log
```

GPU:

```bash
nvidia-smi
```

## 15. If texture fails

Check diagnostics first:

```bash
curl http://127.0.0.1:8010/diagnostics
```

Common fixes:

```bash
cd ~/work
source venv/bin/activate
source ~/.bashrc
python -m pip install --no-cache-dir --force-reinstall \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "diffusers==0.31.0" \
  "transformers==4.46.3" \
  "tokenizers==0.20.3" \
  "huggingface_hub==0.26.2" \
  "accelerate==1.1.1"
rm -rf ~/.cache/huggingface/modules/diffusers_modules
```

Rebuild extensions:

```bash
cd ~/work/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
source ~/work/venv/bin/activate
source ~/.bashrc
python setup.py install

cd ~/work/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
python setup.py install
```

Restart worker from notebook cell 7, or from SSH:

```bash
pkill -f 'uvicorn hunyuan_vm_worker:app'
cd ~/work
source venv/bin/activate
source ~/.bashrc
TORCH_LIB=$(python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
export HUNYUAN_KEEP_TEXTURE_PIPELINE=1
uvicorn hunyuan_vm_worker:app --host 0.0.0.0 --port 8010
```

For texture stability, keep `HUNYUAN_KEEP_TEXTURE_PIPELINE=1` on L4/32GB+ VMs. The first texture load may be slow, but later texture jobs reuse the loaded pipeline instead of loading VAE/pipeline components again.

## 16. Spot VM warnings

Spot VM can stop at any time.

Keep:

- Jupyter in `tmux`
- worker in notebook subprocess or tmux
- Cloudflare tunnels in separate SSH tabs
- important GLB outputs downloaded or copied to persistent storage
