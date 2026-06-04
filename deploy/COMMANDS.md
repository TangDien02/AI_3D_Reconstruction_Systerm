# Commands

Day la file lenh van hanh nhanh cho demo va deploy lai.

## VM GPU: check hardware

```bash
nvidia-smi
python3 --version
df -h
free -h
```

## VM GPU: clone va setup worker

```bash
mkdir -p ~/work
cd ~/work
git clone https://github.com/TangDien02/AI_3D_Reconstruction_Systerm.git
cd AI_3D_Reconstruction_Systerm
bash scripts/gcp_hunyuan_worker_bootstrap.sh
```

## VM GPU: worker systemd

```bash
sudo systemctl status hunyuan-worker --no-pager
sudo systemctl restart hunyuan-worker
sudo journalctl -u hunyuan-worker -f
curl http://127.0.0.1:8010/health
```

## VM GPU: worker tmux debug mode

```bash
cd ~/work/AI_3D_Reconstruction_Systerm
bash deploy/scripts/start_worker_tmux.sh
tmux attach -t worker
```

Detach ma khong tat worker:

```text
Ctrl+B, roi D
```

## VM GPU: Cloudflare quick tunnel

```bash
cloudflared tunnel --url http://127.0.0.1:8010
```

Chay trong tmux:

```bash
cd ~/work/AI_3D_Reconstruction_Systerm
bash deploy/scripts/start_tunnel_tmux.sh
tmux attach -t tunnel
```

Copy URL co dang:

```text
https://xxxx.trycloudflare.com
```

Moi lan quick tunnel restart, URL co the doi. Cap nhat backend `.env.local` va restart backend.

## Windows backend: .env.local

```env
RECONSTRUCTION_BACKEND=hunyuan_remote
HUNYUAN_REMOTE_URL=https://YOUR_TUNNEL.trycloudflare.com
HUNYUAN_REMOTE_OUTPUT_FORMAT=glb
HUNYUAN_REMOTE_ENABLE_TEXTURE=false
HUNYUAN_REMOTE_TIMEOUT_SECONDS=1800
HUNYUAN_REMOTE_POLL_INTERVAL_SECONDS=5
IMAGE_CLEANER_BACKEND=auto
ENABLE_REMBG_CLEANER=true
CLEAN_IMAGE_MAX_SIDE=1536
CLEAN_IMAGE_PAD_RATIO=0.08
```

## Windows backend: setup va run

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Run hidden helper:

```powershell
.\deploy\scripts\start_backend_windows.ps1 -HostIp 192.168.1.5
```

## Backend health

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe https://YOUR_TUNNEL.trycloudflare.com/health
```

## Backend local clean smoke test

```powershell
curl.exe -X POST "http://127.0.0.1:8000/preprocess/clean-image" `
  -F "image=@project\samples\chair_demo.png" `
  -F "bbox_x=10" `
  -F "bbox_y=10" `
  -F "bbox_width=400" `
  -F "bbox_height=400" `
  -F "job_id=clean-smoke"
```

## Reconstruct bbox smoke test

```powershell
curl.exe -X POST "http://127.0.0.1:8000/reconstruct-bbox" `
  -F "image=@project\samples\chair_demo.png" `
  -F "bbox_x=10" `
  -F "bbox_y=10" `
  -F "bbox_width=400" `
  -F "bbox_height=400"
```

Poll job:

```powershell
curl.exe "http://127.0.0.1:8000/reconstruction-jobs/JOB_ID"
```

## Expo

```powershell
cd mobile
npm install
$env:EXPO_PUBLIC_API_BASE_URL="http://192.168.1.5:8000"
npm start
```

Android emulator:

```powershell
$env:EXPO_PUBLIC_API_BASE_URL="http://10.0.2.2:8000"
```

## Jupyter log view

```python
import subprocess, time
from IPython.display import clear_output

while True:
    clear_output(wait=True)
    print(subprocess.check_output(
        ["tmux", "capture-pane", "-t", "worker", "-p", "-S", "-120"],
        text=True,
    ))
    time.sleep(2)
```

## Git deploy branch

```bash
git status
git pull --ff-only
```
