# AI 3D Reconstruction System

Repo nay da duoc rut gon de chi giu luong runtime:

`mobile app -> FastAPI backend -> YOLO segmentation -> TripoSR -> mesh / point cloud export`

## Runtime duoc ho tro

- Python: `3.11.x`
- Node.js: `20.x`

Khong ho tro `Python 3.12+` cho setup mac dinh cua repo nay. TripoSR van pin `transformers==4.35.0`, va stack do de vo hon tren 3.12/3.14 vi `tokenizers` co the bi buoc build native.

## Thu muc duoc giu

- `mobile/`: Expo app chup camera va goi backend.
- `server/`: FastAPI backend, YOLO detect, reconstruct API.
- `project/src/reconstruction/`: TripoSR runner.
- `project/src/utils/`: point cloud export va preview.
- `project/external/TripoSR/`: source TripoSR local.
- `project/torchmcubes/`: fallback local cho `torchmcubes`.
- `project/samples/`: input sample cho smoke test.
- `scripts/setup.ps1`: script setup moi truong cho Windows.
- `scripts/verify_runtime.py`: check import runtime.

## Setup tren Windows

Buoc setup chuan:

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\scripts\setup.ps1
```

Script se:

1. Tim `Python 3.11`
2. Tao lai `.venv`
3. Cai `requirements.txt`
4. Chay import check cho backend va TripoSR runner

Neu script bao khong tim thay `Python 3.11`, cai `Python 3.11.x` roi chay lai.

## Chay backend

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\.venv\Scripts\Activate.ps1
$env:TRIPOSR_REPO_DIR="C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02\project\external\TripoSR"
$env:TRIPOSR_MODEL_SAVE_FORMAT="glb"
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Kiem tra health:

```powershell
curl http://127.0.0.1:8000/health
```

## Smoke test reconstruct-image

```powershell
curl -X POST "http://127.0.0.1:8000/reconstruct-image" `
  -F "image=@project\samples\chair_demo.png"
```

Artifact duoc ghi vao:

- `server/models/<job_id>/mesh.glb` hoac `mesh.obj`
- `server/models/<job_id>/pointcloud.ply`
- `server/models/<job_id>/mesh_colored.ply`
- `server/models/<job_id>/preview.png`
- `server/models/<job_id>/reconstruction_summary.json`

## Chay mobile

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02\mobile
npm install
$env:EXPO_PUBLIC_API_BASE_URL="http://<your-lan-ip>:8000"
npm start
```

Neu dung Android Emulator:

```powershell
$env:EXPO_PUBLIC_API_BASE_URL="http://10.0.2.2:8000"
```

## Docker

Docker la fallback cho backend. Build context da tro ve root de backend thay duoc ca `server/` va `project/`.

```powershell
docker compose up --build backend
```
