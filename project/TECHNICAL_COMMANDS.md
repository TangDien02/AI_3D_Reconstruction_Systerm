# Technical Commands

Run these commands from `project/` unless noted.

## TripoSR Core

```powershell
py -3.10 -m venv .venv-triposr
.\.venv-triposr\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-triposr.txt
git clone https://github.com/VAST-AI-Research/TripoSR.git external/TripoSR
```

Use an existing TripoSR clone:

```powershell
$env:TRIPOSR_REPO_DIR="C:\path\to\TripoSR"
```

Run one image:

```powershell
python -m src.reconstruction.triposr_runner --image path\to\object.png --output-dir results\triposr_core --device auto --model-save-format glb
```

Run a clean crop without background removal:

```powershell
python -m src.reconstruction.triposr_runner --image path\to\object.png --output-dir results\triposr_core --no-remove-bg
```

Run the small orchestration wrapper:

```powershell
python -m src.pipeline.sequential_3d_pipeline --image path\to\object.png --output-dir results\sequential_3d --model-save-format glb
```

Expected output per job:

```text
input.<ext>
triposr_input.png
mesh.glb or mesh.obj
mesh_colored.ply
pointcloud.npy
pointcloud.ply
preview.png
triposr_summary.json
pipeline_manifest.json
```

## Tests

The unit tests use fake TripoSR objects and do not download weights:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

From repo root:

```powershell
npm test
```

## FastAPI Server

Run from repo root:

```powershell
pip install -r server\requirements.txt
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful environment variables:

```powershell
$env:TRIPOSR_REPO_DIR="C:\path\to\TripoSR"
$env:TRIPOSR_DEVICE="auto"
$env:TRIPOSR_MODEL_SAVE_FORMAT="glb"
$env:TRIPOSR_MC_RESOLUTION="256"
$env:TRIPOSR_NUM_POINTS="2048"
$env:TRIPOSR_REMOVE_BACKGROUND="true"
```

Check server:

```powershell
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/reconstruct-image" -F "image=@path\to\object.png"
```

Object flow with YOLO:

```text
/detect-frame       returns object boxes
/segment-object    saves crop/mask debug artifacts
/reconstruct-object runs YOLO selection + TripoSR reconstruction
```

YOLO weights are expected at:

```text
server/weights/yolo26n-seg.pt
```

## Mobile

Run from repo root:

```powershell
npm --prefix mobile install
$env:EXPO_PUBLIC_API_BASE_URL="http://<server-ip>:8000"
npm --prefix mobile start
```

## Legacy Baseline

Old Pix3D preprocessing/training/evaluation code is still available under
`src/preprocessing`, `src/training`, `src/evaluation`, `src/inference`, and
`src/models`. Treat it as legacy comparison code, not the main reconstruction
path.
