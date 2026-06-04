# Google Cloud Spot VM + Cloudflare + Hunyuan3D Setup

This guide is ASCII-only on purpose, so it is safe to open in terminals,
JupyterLab, GitHub, and Windows editors without Vietnamese encoding issues.

Goal:

```text
Windows browser -> Cloudflare Tunnel -> VM JupyterLab :8888
Windows backend -> Cloudflare Tunnel -> VM Hunyuan worker :8010
```

For notebook-only tests, the Hunyuan worker can stay local:

```text
Jupyter notebook on VM -> http://127.0.0.1:8010
```

## 0. Recommended VM

For shape-only, Tesla T4 16GB can work.

For full Hunyuan3D-2 shape plus texture, prefer more system RAM:

```text
GPU: T4 / L4 / A100, depending on quota
RAM: at least 30GB, better 50GB+
Disk: 100GB minimum, 150GB+ if keeping models/outputs
OS: Debian 12 or Ubuntu 22.04/24.04
Provisioning model: Spot
```

If using Debian 12, the commands below match your previous VM.

## 1. SSH into the new VM

In Google Cloud Console:

```text
Compute Engine -> VM instances -> SSH
```

Prompt should look like:

```bash
pminhchien2006@instance-name:~$
```

## 2. Basic packages

```bash
sudo apt update
sudo apt install -y git wget curl tmux unzip build-essential dkms python3-pip python3-venv
```

## 3. NVIDIA driver on Debian 12

Check GPU:

```bash
lspci | grep -i nvidia
```

Enable Debian non-free repos if needed:

```bash
sudo sed -i 's/^Components: main$/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources
sudo apt update
```

Install driver:

```bash
sudo apt install -y linux-headers-$(uname -r) nvidia-driver firmware-misc-nonfree
sudo reboot
```

After reboot, SSH again:

```bash
nvidia-smi
```

Expected:

```text
Tesla T4 / L4 / A100 ...
```

## 4. Install CUDA Toolkit with nvcc

This is required for Hunyuan texture extensions such as `custom_rasterizer`.
The NVIDIA driver alone is not enough.

Install NVIDIA CUDA apt repo for Debian 12:

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

Install CUDA 12 toolkit. Prefer a CUDA 12.x toolkit, not CUDA 13.x, so it
stays compatible with PyTorch CUDA 12 wheels.

```bash
sudo apt install -y cuda-toolkit-12-6
```

If `cuda-toolkit-12-6` is unavailable, list available 12.x versions:

```bash
apt-cache search '^cuda-toolkit-12'
```

Then install one available CUDA 12.x package, for example:

```bash
sudo apt install -y cuda-toolkit-12-8
```

Set environment variables:

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

## 5. Create workspace and Python env

```bash
mkdir -p ~/work
cd ~/work
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

Install PyTorch CUDA. Use CUDA 12.6 wheels when using CUDA 12.x toolkit:

```bash
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
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

## 6. Install JupyterLab and cloudflared

```bash
pip install --no-cache-dir jupyterlab
```

Install Cloudflare Tunnel client:

```bash
cd ~/work
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
cloudflared --version
```

## 7. Start JupyterLab

Use tmux so the server survives closed SSH browser tabs:

```bash
tmux new -s jupyter
cd ~/work
source venv/bin/activate
source ~/.bashrc
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.allow_remote_access=True
```

Copy the Jupyter token from the printed URL.

Detach from tmux:

```text
Ctrl+B
D
```

## 8. Open JupyterLab through Cloudflare Tunnel

Open a second SSH tab:

```bash
cloudflared tunnel --url http://127.0.0.1:8888
```

It prints a URL like:

```text
https://random-name.trycloudflare.com
```

Open on Windows:

```text
https://random-name.trycloudflare.com/lab?token=YOUR_JUPYTER_TOKEN
```

Quick Tunnel URLs change every time you restart `cloudflared`.

## 9. Upload the Cloudflare notebook

Upload this notebook to JupyterLab:

```text
notebook/Hunyuan_workflow_vm_cloudflare_full_ascii.ipynb
```

Run cells from top to bottom.

Important:

```text
Notebook local worker URL: http://127.0.0.1:8010
Jupyter Cloudflare URL:    https://....trycloudflare.com
```

Do not use the Jupyter URL as the Hunyuan worker URL.

## 10. Build Hunyuan texture extensions

The notebook install cell should run these, but if texture later says
`No module named custom_rasterizer`, run manually:

```bash
cd ~/work/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
source ~/work/venv/bin/activate
source ~/.bashrc
python setup.py install

cd ~/work/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
python setup.py install
```

Test:

```bash
python - <<'PY'
import custom_rasterizer
import mesh_processor
print("texture extensions OK")
PY
```

## 10a. Repair Hunyuan Paint diffusers mismatch

If texture loading fails with:

```text
Expected types for unet: (<class 'diffusers_modules.local.unet.modules.UNet2p5DConditionModel'>,), got <class 'diffusers_modules.local.modules.UNet2p5DConditionModel'>.
```

pin `diffusers` to the version known to work with Hunyuan Paint:

```bash
cd ~/work
source venv/bin/activate
python -m pip install --no-cache-dir "diffusers==0.31.0" "huggingface_hub<1.0"
rm -rf ~/.cache/huggingface/modules/diffusers_modules
```

If you previously patched `trust_remote_code=True` into either file, remove it:

```bash
python - <<'PY'
from pathlib import Path
import re

paths = [
    Path.home() / "work/hunyuan_vm_worker.py",
    Path.home() / "work/Hunyuan3D-2/hy3dgen/texgen/utils/multiview_utils.py",
]

for path in paths:
    if not path.exists():
        print("missing", path)
        continue
    text = path.read_text(encoding="utf-8")
    original = text
    text = re.sub(r"\s*trust_remote_code\s*=\s*True\s*,\s*", "", text)
    text = re.sub(r",\s*trust_remote_code\s*=\s*True\s*", "", text)
    text = text.replace("trust_remote_code=True", "")
    text = text.replace("(,", "(").replace(",,", ",")
    if text != original:
        path.write_text(text, encoding="utf-8")
        print("removed trust_remote_code from", path)
    else:
        print("no trust_remote_code patch found in", path)
PY
```

Then restart the VM worker:

```bash
pkill -f 'uvicorn hunyuan_vm_worker:app'
cd ~/work
source venv/bin/activate
source ~/.bashrc
uvicorn hunyuan_vm_worker:app --host 0.0.0.0 --port 8010
```

The project also includes a helper script with the same repair:

```text
scripts/fix_hunyuan_texture_vm.py
```

## 11. Optional: expose Hunyuan worker port 8010

Only needed if Windows backend or mobile app needs to call the VM worker.

First, make sure the notebook started worker on port 8010:

```bash
curl http://127.0.0.1:8010/health
```

Open a third SSH tab:

```bash
cloudflared tunnel --url http://127.0.0.1:8010
```

It prints another URL:

```text
https://another-random-name.trycloudflare.com
```

Use that URL as:

```text
HUNYUAN_REMOTE_URL=https://another-random-name.trycloudflare.com
```

## 12. Windows backend commands

After the worker 8010 Cloudflare URL exists, run this in Windows PowerShell:

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\.venv\Scripts\Activate.ps1
$env:RECONSTRUCTION_BACKEND="hunyuan_remote"
$env:HUNYUAN_REMOTE_URL="https://YOUR_WORKER_8010.trycloudflare.com"
$env:HUNYUAN_REMOTE_OUTPUT_FORMAT="glb"
$env:HUNYUAN_REMOTE_ENABLE_TEXTURE="false"
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Then Expo/mobile points to Windows backend:

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02\mobile
$env:EXPO_PUBLIC_API_BASE_URL="http://YOUR_WINDOWS_LAN_IP:8000"
npm install
npm start
```

## 13. Spot VM warnings

Spot VM can stop at any time.

Do this:

- Keep long-running servers in tmux.
- Save outputs and checkpoints often.
- Download important `.glb` files from JupyterLab.
- Keep source code in GitHub.
- Use a persistent disk or Cloud Storage if outputs matter.

## 14. Quick restart checklist

SSH tab 1:

```bash
tmux attach -t jupyter
```

If no session:

```bash
tmux new -s jupyter
cd ~/work
source venv/bin/activate
source ~/.bashrc
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.allow_remote_access=True
```

SSH tab 2:

```bash
cloudflared tunnel --url http://127.0.0.1:8888
```

SSH tab 3, only if exposing worker:

```bash
cloudflared tunnel --url http://127.0.0.1:8010
```
