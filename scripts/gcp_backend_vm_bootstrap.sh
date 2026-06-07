#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/work}"
REPO_URL="${REPO_URL:-https://github.com/TangDien02/AI_3D_Reconstruction_Systerm.git}"
REPO_DIR="${REPO_DIR:-$WORK_DIR/AI_3D_Reconstruction_Systerm}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
HUNYUAN_REMOTE_URL="${HUNYUAN_REMOTE_URL:-}"

if [ -z "$HUNYUAN_REMOTE_URL" ]; then
  echo "Set HUNYUAN_REMOTE_URL before running this script."
  exit 1
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends git curl build-essential python3.11 python3.11-venv python3-pip

mkdir -p "$WORK_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only
fi

cd "$REPO_DIR"
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel "setuptools<82"
python -m pip install --no-cache-dir -r requirements.txt
export MPLBACKEND=Agg
python scripts/verify_runtime.py

cat > "$REPO_DIR/backend.env" <<EOF
RECONSTRUCTION_BACKEND=hunyuan_remote
HUNYUAN_REMOTE_URL=$HUNYUAN_REMOTE_URL
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
sudo systemctl enable --now ai-3d-backend

echo "Backend service started."
echo "Check: curl http://127.0.0.1:8000/health"
echo "Logs:  sudo journalctl -u ai-3d-backend -f"
