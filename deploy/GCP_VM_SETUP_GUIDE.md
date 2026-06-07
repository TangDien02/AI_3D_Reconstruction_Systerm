# Complete GCP VM Setup Guide
## AI 3D Reconstruction System - End-to-End Setup

Hướng dẫn chi tiết setup từ đầu: GCP VM → SSH → Project setup → Windows local

---

## Part 1: Create GCP VM with GPU

### 1.1 Create VM in Google Cloud Console

1. Truy cập [Google Cloud Console](https://console.cloud.google.com)
2. **Compute Engine** → **VM instances** → **Create instance**

#### VM Configuration:
```
Name: endtoend-l4-gpu-v1
Region: asia-east1 (Taiwan)
Zone: asia-east1-b
Machine type: Custom (4 vCPU, 32GB RAM)
GPU: 1x NVIDIA L4 (recommended) hoặc T4
Boot disk: Debian 12 (100GB)
```

#### Network Configuration:
```
Network: default
Subnetwork: default
External IP: Ephemeral (hoặc Static nếu dùng lâu dài)
Firewall: Allow HTTP, Allow HTTPS
```

3. **Create** → Chờ VM start (~2 phút)

### 1.2 Note VM IP

Sau khi VM được tạo, ghi lại:
- **External IP**: `10.140.0.11` (dùng SSH từ ngoài)
- **Internal IP**: `35.201.215.67` (network GCP nội bộ)

---

## Part 2: SSH Into VM & Basic Setup

### 2.1 Open SSH from Cloud Console (Easy)

1. Vào VM instance trong Cloud Console
2. Bấm **SSH** button → Terminal mở

### 2.2 Hoặc SSH từ Local

```bash
# Windows PowerShell hoặc Linux/Mac terminal
gcloud compute ssh admin@endtoend-l4-gpu-v1 --zone=asia-east1-b
```

---

## Part 3: Install NVIDIA Driver & CUDA

### 3.1 Update System
```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential wget curl git
```

### 3.2 Install NVIDIA Driver

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Install CUDA Toolkit + Driver
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6 nvidia-driver-550
```

### 3.3 Verify Driver Installation

```bash
nvidia-smi
```

Output sẽ show NVIDIA GPU info. Nếu success → proceed.

### 3.4 Add NVIDIA to PATH (Optional but Recommended)

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Part 4: Install Python & Dependencies

### 4.1 Install Python 3.11

```bash
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
```

### 4.2 Create Working Directory

```bash
mkdir -p ~/work
cd ~/work
```

---

## Part 5: Clone & Setup Project

### 5.1 Clone Repository

```bash
cd ~/work
git clone https://github.com/TangDien02/AI_3D_Reconstruction_Systerm.git
cd AI_3D_Reconstruction_Systerm
```

### 5.2 Create Python Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 5.3 Install Python Dependencies

```bash
python -m pip install -U pip wheel "setuptools<82"
export MPLBACKEND=Agg
python -m pip install --no-cache-dir -r requirements.txt
```

**Note**: Cài đặt này mất **15-30 phút** (lần đầu download PyTorch lớn)

### 5.4 Verify Runtime

```bash
export MPLBACKEND=Agg
python scripts/verify_runtime.py
```

Expected output:
```
Python: 3.11.2 ...
Reconstruction backend: Hunyuan remote
Runtime import check: OK
```

---

## Part 6: Setup Hunyuan Worker

### 6.1 Create Worker Bootstrap Script

Từ directory `~/work/AI_3D_Reconstruction_Systerm`, chạy:

```bash
cat > hunyuan_worker_setup.sh <<'EOF'
#!/bin/bash
set -euo pipefail

REPO_DIR="$HOME/work/AI_3D_Reconstruction_Systerm"
VENV_DIR="$REPO_DIR/.venv"

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

# Download Hunyuan model (first time only, ~50GB)
python -c "
from diffusers import DiffusionPipeline
import torch
model_id = 'tencent/Hunyuan3D-2'
pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    subfolder='hunyuan3d-dit-v2-0',
    torch_dtype=torch.float16,
    device_map='auto'
)
print('Model downloaded successfully!')
"

# Create Hunyuan worker systemd service
sudo tee /etc/systemd/system/hunyuan-worker.service >/dev/null <<'SRVEOF'
[Unit]
Description=Hunyuan 3D Worker Service
After=network-online.target nvidia-driver.service
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
EnvironmentFile=$REPO_DIR/worker.env
ExecStart=$VENV_DIR/bin/python -m uvicorn server.hunyuan_worker:app --host 0.0.0.0 --port 8010
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SRVEOF

# Create worker environment file
cat > "$REPO_DIR/worker.env" <<'ENVEOF'
MPLBACKEND=Agg
HUNYUAN_ENABLE_SHAPE=true
HUNYUAN_ENABLE_TEXTURE=false
DEVICE_MEMORY_FRACTION=0.95
ENVEOF

sudo systemctl daemon-reload
sudo systemctl enable --now hunyuan-worker
echo "Hunyuan worker service started!"
EOF

chmod +x hunyuan_worker_setup.sh
```

### 6.2 Run Worker Setup (Takes ~20-30 minutes for first-time model download)

```bash
./hunyuan_worker_setup.sh
```

### 6.3 Verify Worker Health

```bash
sleep 10
curl http://127.0.0.1:8010/health | python3 -m json.tool
```

Expected: JSON response with `"status":"ok"` và GPU info.

---

## Part 7: Setup FastAPI Backend

### 7.1 Run Backend Bootstrap Script

```bash
cd ~/work/AI_3D_Reconstruction_Systerm
export HUNYUAN_REMOTE_URL="http://127.0.0.1:8010"
export MPLBACKEND=Agg
bash scripts/gcp_backend_vm_bootstrap.sh
```

### 7.2 Verify Backend Health

```bash
curl http://127.0.0.1:8000/health | python3 -m json.tool
```

Expected: JSON response with `"status":"ok"`.

---

## Part 8: Setup Cloudflare Tunnel for Backend

### 8.1 Create Tunnel Script

```bash
cat > ~/work/start_backend_tunnel.sh <<'EOF'
#!/bin/bash
set -euo pipefail

# Download cloudflared binary
if [ ! -f "$HOME/.local/bin/cloudflared" ]; then
  mkdir -p "$HOME/.local/bin"
  wget -qO "$HOME/.local/bin/cloudflared" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$HOME/.local/bin/cloudflared"
fi

# Run tunnel (shows URL you need to copy)
$HOME/.local/bin/cloudflared tunnel --url http://127.0.0.1:8000
EOF

chmod +x ~/work/start_backend_tunnel.sh
```

### 8.2 Run Tunnel in Separate SSH Session

```bash
# SSH Tab 1 (for backend logs)
ssh user@EXTERNAL_IP

# SSH Tab 2 (for tunnel - keep running)
bash ~/work/start_backend_tunnel.sh
```

Output sẽ show:
```
Your quick tunnel has been created! Visit it at (it may take some time to be reachable):
https://RANDOM_NAME.trycloudflare.com
```

**Copy URL này - dùng cho Expo app.**

---

## Part 9: Setup Windows Local Development

### 9.1 Prerequisites on Windows

- **Node.js 18+**: Download from [nodejs.org](https://nodejs.org)
- **Expo CLI**: `npm install -g expo-cli`
- **Git for Windows** (optional, for git commands)

### 9.2 Clone Mobile App

```powershell
# PowerShell
cd C:\Users\YourUsername\Desktop
git clone https://github.com/TangDien02/AI_3D_Reconstruction_Systerm_TangDien02.git
cd AI_3D_Reconstruction_Systerm_TangDien02\mobile
```

### 9.3 Install Dependencies

```powershell
npm install
```

### 9.4 Start Expo with Backend URL

**Option A: Using LAN IP (if VM and Windows on same network)**

```powershell
$env:EXPO_PUBLIC_API_BASE_URL="http://192.168.1.6:8000"
npm start -- --host lan
```

**Option B: Using Cloudflare Tunnel URL**

```powershell
$env:EXPO_PUBLIC_API_BASE_URL="https://RANDOM_NAME.trycloudflare.com"
npm start -- --host lan
```

### 9.5 Open App on Phone

1. Install **Expo Go** app from App Store / Google Play
2. Quét QR code từ Expo terminal
3. Wait for app to load

---

## Part 10: Verify End-to-End Setup

### 10.1 Test Backend Health

```powershell
# Windows PowerShell
curl http://192.168.1.6:8000/health
# hoặc
curl https://RANDOM_NAME.trycloudflare.com/health
```

### 10.2 Test Preprocess Endpoint

```powershell
# Download test image
$testImage = "C:\Users\YourUsername\Desktop\test_chair.jpg"

# Test preprocessing
curl -X POST "http://192.168.1.6:8000/preprocess/clean-image" `
  -F "image=@$testImage" `
  -F "bbox_x=10" `
  -F "bbox_y=10" `
  -F "bbox_width=400" `
  -F "bbox_height=400" `
  -F "job_id=test-job"
```

### 10.3 Test in Expo App

1. Upload image via app
2. Wait for preprocessing
3. Should see cleaned image → reconstruction job

---

## Part 11: Troubleshooting

### Worker Not Starting

```bash
# Check logs
sudo journalctl -u hunyuan-worker -n 100 --no-pager

# Try restart
sudo systemctl restart hunyuan-worker
```

### Backend Not Reaching Worker

```bash
# Test connectivity from backend
curl http://127.0.0.1:8010/health

# Check backend logs
sudo journalctl -u ai-3d-backend -n 100 --no-pager
```

### Expo App Connection Failed

```powershell
# Test backend is reachable from Windows
curl http://192.168.1.6:8000/health

# Check firewall allows port 8000
# GCP Console → VM → Edit → Allow HTTP/HTTPS
```

### CUDA Out of Memory

```bash
# Check GPU memory
nvidia-smi

# If worker needs more memory, check Hunyuan worker logs
sudo journalctl -u hunyuan-worker -f | grep -i "cuda\|memory"
```

### matplotlib ImportError on VM

If error `Missing imports: matplotlib` occurs:

```bash
export MPLBACKEND=Agg
python scripts/verify_runtime.py
```

---

## Part 12: Quick Reference Commands

### VM SSH
```bash
gcloud compute ssh admin@endtoend-l4-gpu-v1 --zone=asia-east1-b
```

### Check Services Status
```bash
sudo systemctl status hunyuan-worker
sudo systemctl status ai-3d-backend
```

### Restart Services
```bash
sudo systemctl restart hunyuan-worker
sudo systemctl restart ai-3d-backend
```

### View Real-time Logs
```bash
sudo journalctl -u hunyuan-worker -f
sudo journalctl -u ai-3d-backend -f
```

### Stop Services
```bash
sudo systemctl stop hunyuan-worker
sudo systemctl stop ai-3d-backend
```

---

## Part 13: Architecture Overview

```
Expo App (Windows Laptop)
    ↓
    ├─→ Option A: http://192.168.1.6:8000 (LAN)
    └─→ Option B: https://RANDOM_NAME.trycloudflare.com (Tunnel)
        ↓
FastAPI Backend (VM Port 8000)
    ↓
    Local Image Cleaning
    (YOLO bbox crop + Rembg)
    ↓
Hunyuan Worker (VM Port 8010)
    ↓
    3D Model Generation
    ↓
GLB Model Output
    ↓
Back to Expo App
```

---

## Part 14: Performance Tips

1. **L4 GPU**: Best performance for this setup
2. **32GB RAM minimum**: For simultaneous image processing
3. **100GB disk**: Needed for model weights (~50GB) + outputs
4. **Persistent IP**: Use Static IP nếu VM dùng lâu dài
5. **Preemptible VM**: Cheaper nhưng có thể bị stop bất ngờ (dev only)

---

## Notes

- ⏱️ **First-time setup**: ~1-2 hours (model download is slow)
- 💾 **Model size**: ~50GB (stored in VM disk)
- 🔌 **Keep Cloudflare tunnel running**: Use `tmux` or `screen` để keep alive
- 🔐 **Security**: Backend not exposed directly, only through Cloudflare tunnel
- 🌐 **Network**: Ensure Windows and VM on same network for LAN access

---

Generated: 2026-06-07
