# AI 3D Reconstruction System

Loi hien tai cua repo la **single image / object crop -> TripoSR -> mesh + sampled point cloud**.

Khong con coi ResNet point-cloud baseline la duong chinh. Code baseline cu van nam trong `project/src/training`, `project/src/inference` va `project/src/models` de doi chieu/bao cao, nhung workflow nen chay qua TripoSR.

## Main Flow

```text
image
  -> optional YOLO bbox crop in server
  -> TripoSR background removal / reconstruction
  -> mesh .glb or .obj
  -> sampled pointcloud .npy + .ply
  -> JSON summary
```

## Key Folders

```text
project/
  src/reconstruction/triposr_runner.py      TripoSR core runner
  src/pipeline/sequential_3d_pipeline.py    small single-process orchestration
  tests/                                    core contract tests with fake TripoSR model
server/
  main.py                                   FastAPI: YOLO crop + TripoSR endpoints
mobile/
  App.js                                    Expo client for camera -> server flow
```

Generated data, outputs, weights, virtualenvs, `node_modules`, and local `.env` files are ignored. Large processed Pix3D images/masks were removed from git; regenerate them locally when needed.

## Quick Test

These tests do not download TripoSR weights:

```powershell
python -m unittest discover -s project/tests -p "test_*.py" -v
```

## TripoSR Setup

Use Python 3.10 or 3.11:

```powershell
cd project
py -3.10 -m venv .venv-triposr
.\.venv-triposr\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-triposr.txt
git clone https://github.com/VAST-AI-Research/TripoSR.git external/TripoSR
```

Run core reconstruction:

```powershell
python -m src.reconstruction.triposr_runner --image path\to\object.png --output-dir results\triposr_core --device auto --model-save-format glb
```

For already clean object crops:

```powershell
python -m src.reconstruction.triposr_runner --image path\to\object.png --output-dir results\triposr_core --no-remove-bg
```

## Server

```powershell
pip install -r server\requirements.txt
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:

```text
GET  /health
POST /reconstruct-image
POST /detect-frame
POST /reconstruct-object
```

## Mobile

```powershell
npm --prefix mobile install
$env:EXPO_PUBLIC_API_BASE_URL="http://<server-ip>:8000"
npm --prefix mobile start
```
