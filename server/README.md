# FastAPI Backend

Backend nay phuc vu luong runtime duy nhat:

- `POST /detect-frame`
- `POST /segment-object`
- `POST /reconstruct-object`
- `POST /reconstruct-image`
- `GET /health`

## Interpreter

Backend nay duoc chuan hoa de chay voi `Python 3.11.x`.

Dung setup script o root:

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\scripts\setup.ps1
```

## Run

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\.venv\Scripts\Activate.ps1
$env:TRIPOSR_MODEL_SAVE_FORMAT="glb"
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Mac dinh backend se dung source vendored tai `project/external/TripoSR`. Chi set `TRIPOSR_REPO_DIR` neu anh doi vi tri thu muc nay.

## Run voi Hunyuan Colab worker

Neu muon de backend local goi Google Colab worker thay vi TripoSR local:

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\.venv\Scripts\Activate.ps1
$env:RECONSTRUCTION_BACKEND="hunyuan_remote"
$env:HUNYUAN_REMOTE_URL="https://<your-colab-tunnel>"
$env:HUNYUAN_REMOTE_OUTPUT_FORMAT="glb"
$env:HUNYUAN_REMOTE_ENABLE_TEXTURE="false"
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Backend se:

- nhan anh tu Expo
- YOLO detect + crop object
- gui crop sang `POST <HUNYUAN_REMOTE_URL>/generate-shape` neu `HUNYUAN_REMOTE_ENABLE_TEXTURE=false`
- gui crop sang `POST <HUNYUAN_REMOTE_URL>/generate-textured-shape` neu `HUNYUAN_REMOTE_ENABLE_TEXTURE=true`
- luu mesh vao `server/models/<job_id>/mesh.glb`

Tren Colab Free, nen tao shape-only truoc. Sau khi co `job_id`, paint texture rieng:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/paint-texture" -F "job_id=<JOB_ID>"
```

Colab worker code nam tai `scripts/colab_hunyuan_worker.py`.

Worker Colab duoc toi uu cho T4/RAM 12.7GB bang cach unload shape pipeline truoc khi load texture pipeline. Nen de:

```bash
export HUNYUAN_KEEP_SHAPE_PIPELINE="0"
export HUNYUAN_KEEP_TEXTURE_PIPELINE="0"
```

Neu chi can mesh trang de test nhanh, tat texture o backend:

```powershell
$env:HUNYUAN_REMOTE_ENABLE_TEXTURE="false"
```

## Output

Runtime artifact:

- `server/uploads/`
- `server/segment_outputs/`
- `server/models/`
