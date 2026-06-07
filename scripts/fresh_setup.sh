#!/bin/bash
# Clean setup script - remove old installation and setup fresh

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Fresh Setup: Clean old install + setup from scratch      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================
# Step 1: Stop all services
# ============================================
echo "Step 1: Stopping services..."
sudo systemctl stop hunyuan-worker || true
sudo systemctl stop ai-3d-backend || true
sleep 2
echo "✓ Services stopped"
echo ""

# ============================================
# Step 2: Remove old installation
# ============================================
echo "Step 2: Removing old installation..."
rm -rf ~/work/AI_3D_Reconstruction_Systerm
rm -f ~/work/start_tunnel.sh
echo "✓ Old installation removed"
echo ""

# ============================================
# Step 3: Verify system packages
# ============================================
echo "Step 3: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential wget curl git tmux python3.11 python3.11-venv python3.11-dev python3-pip
echo "✓ System packages ready"
echo ""

# ============================================
# Step 4: Clone fresh repo
# ============================================
echo "Step 4: Cloning fresh repository..."
mkdir -p ~/work
cd ~/work
git clone https://github.com/TangDien02/AI_3D_Reconstruction_Systerm.git
cd AI_3D_Reconstruction_Systerm
git checkout codex/hunyuan-shape-then-paint
echo "✓ Repository cloned from branch: codex/hunyuan-shape-then-paint"
echo ""

# ============================================
# Step 5: Create Python venv
# ============================================
echo "Step 5: Creating Python virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel "setuptools<82"
echo "✓ Python venv ready"
echo ""

# ============================================
# Step 6: Install dependencies (takes 15-30 min)
# ============================================
echo "Step 6: Installing Python dependencies..."
echo "⏱️  This may take 15-30 minutes..."
export MPLBACKEND=Agg
python -m pip install --no-cache-dir -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# ============================================
# Step 7: Verify runtime
# ============================================
echo "Step 7: Verifying runtime imports..."
export MPLBACKEND=Agg
python scripts/verify_runtime.py
echo "✓ Runtime verification passed"
echo ""

# ============================================
# Step 8: Download Hunyuan model (first time ~50GB)
# ============================================
echo "Step 8: Downloading Hunyuan model..."
echo "⏱️  First time: ~50GB download (10-20 minutes)"
echo "    Subsequent runs: much faster (cached)"
export MPLBACKEND=Agg
python -c "
import torch
from diffusers import DiffusionPipeline
print('Device:', 'CUDA' if torch.cuda.is_available() else 'CPU')
model_id = 'tencent/Hunyuan3D-2'
print(f'Loading {model_id}...')
pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    subfolder='hunyuan3d-dit-v2-0',
    torch_dtype=torch.float16,
    device_map='auto'
)
print('✓ Model loaded successfully!')
"
echo "✓ Hunyuan model ready"
echo ""

# ============================================
# Step 9: Setup Hunyuan worker service
# ============================================
echo "Step 9: Setting up Hunyuan worker service..."
REPO_DIR="$HOME/work/AI_3D_Reconstruction_Systerm"
VENV_DIR="$REPO_DIR/.venv"

cat > "$REPO_DIR/worker.env" <<'EOF'
MPLBACKEND=Agg
HUNYUAN_ENABLE_SHAPE=true
HUNYUAN_ENABLE_TEXTURE=false
EOF

sudo tee /etc/systemd/system/hunyuan-worker.service >/dev/null <<EOF
[Unit]
Description=Hunyuan 3D Worker Service
After=network-online.target
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
EOF

sudo systemctl daemon-reload
sudo systemctl enable hunyuan-worker
sudo systemctl start hunyuan-worker
echo "✓ Hunyuan worker service created and started"
echo ""

# ============================================
# Step 10: Setup FastAPI backend service
# ============================================
echo "Step 10: Setting up FastAPI backend service..."

cat > "$REPO_DIR/backend.env" <<'EOF'
RECONSTRUCTION_BACKEND=hunyuan_remote
HUNYUAN_REMOTE_URL=http://127.0.0.1:8010
HUNYUAN_REMOTE_OUTPUT_FORMAT=glb
HUNYUAN_REMOTE_ENABLE_TEXTURE=false
HUNYUAN_REMOTE_TIMEOUT_SECONDS=1800
HUNYUAN_REMOTE_POLL_INTERVAL_SECONDS=5
IMAGE_CLEANER_BACKEND=auto
ENABLE_REMBG_CLEANER=true
CLEAN_IMAGE_MAX_SIDE=1536
CLEAN_IMAGE_PAD_RATIO=0.08
EOF

sudo tee /etc/systemd/system/ai-3d-backend.service >/dev/null <<EOF
[Unit]
Description=AI 3D Reconstruction FastAPI backend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
EnvironmentFile=$REPO_DIR/backend.env
ExecStart=$VENV_DIR/bin/python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ai-3d-backend
sudo systemctl start ai-3d-backend
echo "✓ FastAPI backend service created and started"
echo ""

# ============================================
# Step 11: Verify services
# ============================================
echo "Step 11: Verifying services..."
sleep 10

echo "Checking Hunyuan worker health..."
for attempt in $(seq 1 30); do
  body=$(curl -fsS --max-time 5 http://127.0.0.1:8010/health 2>/dev/null || true)
  if [ -n "$body" ] && echo "$body" | python3 -m json.tool >/dev/null 2>&1; then
    echo "✓ Hunyuan worker OK"
    break
  fi
  if [ $attempt -eq 30 ]; then
    echo "✗ Worker health check failed!"
    sudo journalctl -u hunyuan-worker -n 30 --no-pager
    exit 1
  fi
  sleep 1
done

echo "Checking FastAPI backend health..."
for attempt in $(seq 1 30); do
  body=$(curl -fsS --max-time 5 http://127.0.0.1:8000/health 2>/dev/null || true)
  if [ -n "$body" ] && echo "$body" | python3 -m json.tool >/dev/null 2>&1; then
    echo "✓ FastAPI backend OK"
    break
  fi
  if [ $attempt -eq 30 ]; then
    echo "✗ Backend health check failed!"
    sudo journalctl -u ai-3d-backend -n 30 --no-pager
    exit 1
  fi
  sleep 1
done

echo ""
echo "✓ All services verified!"
echo ""

# ============================================
# Step 12: Download cloudflared
# ============================================
echo "Step 12: Setting up Cloudflare tunnel..."
if [ ! -f "$HOME/.local/bin/cloudflared" ]; then
  mkdir -p "$HOME/.local/bin"
  wget -qO "$HOME/.local/bin/cloudflared" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$HOME/.local/bin/cloudflared"
fi

cat > "$HOME/work/start_tunnel.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

echo "Starting Cloudflare tunnel on port 8000..."
echo "Press Ctrl+C to stop"
echo ""

$HOME/.local/bin/cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
EOF

chmod +x "$HOME/work/start_tunnel.sh"
echo "✓ Tunnel script ready"
echo ""

# ============================================
# Summary
# ============================================
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🎉 FRESH SETUP COMPLETE!                                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Services running:"
echo "  ✓ Hunyuan Worker (http://127.0.0.1:8010)"
echo "  ✓ FastAPI Backend (http://127.0.0.1:8000)"
echo ""
echo "Next steps:"
echo "  1. Open NEW SSH session"
echo "  2. Run: ~/work/start_tunnel.sh"
echo "  3. Copy the tunnel URL"
echo "  4. Setup Expo app on Windows with tunnel URL or LAN IP"
echo ""
echo "Useful commands:"
echo "  - View worker logs:   sudo journalctl -u hunyuan-worker -f"
echo "  - View backend logs:  sudo journalctl -u ai-3d-backend -f"
echo "  - Restart worker:     sudo systemctl restart hunyuan-worker"
echo "  - Restart backend:    sudo systemctl restart ai-3d-backend"
echo "  - Check GPU:          nvidia-smi"
echo ""
