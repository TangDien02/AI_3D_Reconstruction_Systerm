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
$env:TRIPOSR_REPO_DIR="C:\Users\pminh\Desktop\MyProject\AI_3D_Reconstruction_Systerm_TangDien02\project\external\TripoSR"
$env:TRIPOSR_MODEL_SAVE_FORMAT="glb"
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## Output

Runtime artifact:

- `server/uploads/`
- `server/segment_outputs/`
- `server/models/`
