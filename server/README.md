# FastAPI Backend

Backend nay phuc vu luong runtime:

- `POST /detect-frame` legacy/debug YOLO endpoint
- `POST /segment-object` legacy/debug YOLO crop endpoint
- `POST /reconstruct-object`
- `POST /reconstruct-bbox`
- `POST /reconstruct-image`
- `POST /preprocess/clean-image`
- `POST /paint-texture`
- `GET /reconstruction-jobs/{job_id}`
- `GET /texture-jobs/{job_id}`
- `GET /health`

TripoSR da bi go khoi backend. Reconstruction chi ho tro Hunyuan worker qua:

```text
RECONSTRUCTION_BACKEND=hunyuan_remote
```

Flow chinh moi:

```text
Expo chup anh -> user keo bbox -> /reconstruct-bbox
  -> crop bbox
  -> local clean image (rembg optional, crop_only fallback)
  -> Hunyuan worker start-shape
  -> GLB trả về app
```

## Interpreter

Backend chay voi `Python 3.11.x`.

Dung setup script o root:

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\scripts\setup.ps1
```

## Run

```powershell
cd C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02
.\.venv\Scripts\Activate.ps1
$env:RECONSTRUCTION_BACKEND="hunyuan_remote"
$env:HUNYUAN_REMOTE_URL="https://<your-worker-tunnel>"
$env:HUNYUAN_REMOTE_OUTPUT_FORMAT="glb"
$env:HUNYUAN_REMOTE_ENABLE_TEXTURE="false"
$env:HUNYUAN_REMOTE_TIMEOUT_SECONDS="1800"
$env:HUNYUAN_REMOTE_POLL_INTERVAL_SECONDS="5"
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## Output

Runtime artifact:

- `server/uploads/`
- `server/segment_outputs/`
- `server/models/`
