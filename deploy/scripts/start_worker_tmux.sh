#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-worker}"
WORK_DIR="${WORK_DIR:-$HOME/work}"
REPO_DIR="${REPO_DIR:-$WORK_DIR/AI_3D_Reconstruction_Systerm}"
HUNYUAN_DIR="${HUNYUAN_DIR:-$WORK_DIR/Hunyuan3D-2}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/venv}"
WORKER_MODULE="${WORKER_MODULE:-$REPO_DIR/scripts/hunyuan_vm_worker_production.py}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash -lc '
set -euo pipefail
cd \"$WORK_DIR\"
source \"$VENV_DIR/bin/activate\"
TORCH_LIB=\$(python - <<\"PY\"
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / \"lib\")
PY
)
export PYTHONPATH=\"$HUNYUAN_DIR:\${PYTHONPATH:-}\"
export LD_LIBRARY_PATH=\"\$TORCH_LIB:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}\"
export HUNYUAN_KEEP_SHAPE_PIPELINE=\"\${HUNYUAN_KEEP_SHAPE_PIPELINE:-0}\"
export HUNYUAN_KEEP_TEXTURE_PIPELINE=\"\${HUNYUAN_KEEP_TEXTURE_PIPELINE:-1}\"
export HUNYUAN_VM_WORK_DIR=\"\${HUNYUAN_VM_WORK_DIR:-$WORK_DIR/hunyuan_jobs}\"
cp \"$WORKER_MODULE\" \"$WORK_DIR/hunyuan_vm_worker.py\"
uvicorn hunyuan_vm_worker:app --host 0.0.0.0 --port 8010
'"

echo "Started worker tmux session '$SESSION'."
echo "Attach: tmux attach -t $SESSION"
